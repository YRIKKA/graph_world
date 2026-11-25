"""Shared train/eval loaders: one pinned NF4 base, optionally plus GRPO LoRA.

Unsloth is imported before Transformers/PEFT. No model is converted to BF16 in
full and quantized models are never moved using model.to('cuda').
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from .model_defaults import initial_checkpoint, loading_contract


def validate_revision(revision):
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Pin revision to a full immutable commit SHA')


def text_configuration(model):
    config = model.config
    return config.get_text_config() if hasattr(config, 'get_text_config') else getattr(config, 'text_config', config)


def no_dropout(model):
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = 0.0


def package_versions():
    result = {}
    for name in ('torch','transformers','unsloth','unsloth_zoo','peft','bitsandbytes','triton','flash-linear-attention','causal-conv1d'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _nf4_runtime():
    # Setting logits mode before importing Unsloth avoids its training-only
    # empty-logits optimization; our loss needs actual constrained probabilities.
    # Eager rewritten forwards keep no-grad collection and differentiable
    # replay on the same arithmetic path. FLA/Triton kernels remain available.
    os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    os.environ['UNSLOTH_RETURN_HIDDEN_STATES'] = '0'
    try:
        from unsloth import FastModel
        # Fail early with a dependency error instead of a slow, silent kernel
        # fallback appearing only after the 27B checkpoint has loaded.
        from causal_conv1d import causal_conv1d_fn
    except ImportError as error:
        raise RuntimeError('The NF4 backend requires requirements.txt in an isolated CUDA environment; see docs/NF4_SETUP.md') from error
    return FastModel


def _snapshot(spec):
    from huggingface_hub import snapshot_download
    # Load by local immutable snapshot so Unsloth cannot redirect the model name
    # to a different parent checkpoint or use a moving tokenizer revision.
    if Path(spec['model']).is_dir():
        return str(Path(spec['model']).resolve())
    return snapshot_download(spec['model'], revision=spec['revision'],
        allow_patterns=['*.json','*.safetensors','*.jinja','*.model','*.txt'],max_workers=2)


def _official_tokenizer(snapshot):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError('The pinned checkpoint must provide an official chat template')
    return tokenizer


def _validate_nf4(model, contract):
    import torch
    import bitsandbytes as bnb
    quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    if not quantized or not getattr(model, 'is_loaded_in_4bit', False):
        raise RuntimeError('Expected a prequantized NF4 model; refusing a dense or silently requantized fallback')
    dtype = getattr(torch, contract['compute_dtype'])
    for module in quantized:
        state = getattr(module.weight, 'quant_state', None)
        if state is None or state.quant_type != 'nf4' or not state.nested:
            raise RuntimeError('Loaded weights do not match the NF4/double-quant initialization contract')
        # Stored checkpoint compute dtype is FP16. Apply the same explicitly
        # chosen BF16/FP16 compute dtype in training AND both evaluation arms.
        module.compute_dtype = dtype
        module.compute_type_is_set = True
    return len(quantized)


def _adapter_path(spec):
    adapter = spec.get('adapter')
    if not adapter or Path(adapter).is_dir():
        return adapter
    from huggingface_hub import snapshot_download
    validate_revision(spec.get('adapter_revision',''))
    return snapshot_download(adapter, revision=spec['adapter_revision'])


def _verify_adapter_contract(adapter, spec, contract):
    path = Path(adapter)/'mosp_backend.json'
    if not path.exists():
        raise ValueError('NF4 adapter lacks mosp_backend.json initialization provenance')
    metadata = json.loads(path.read_text())
    if metadata['initial_checkpoint'] != dict(model=spec['model'], revision=spec['revision'], loading=contract):
        raise ValueError('Adapter was trained from a different base revision, quantization or loading contract')


def _load_nf4(config, spec, training):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Unsloth NF4 requires CUDA; CPU environment/evaluator tests remain available')
    contract = spec['loading']
    if contract['compute_dtype'] not in ('bfloat16','float16'):
        raise ValueError('NF4 compute_dtype must be bfloat16 or float16')
    if config.get('device','cuda').split(':')[0] != 'cuda':
        raise ValueError('The Unsloth NF4 backend requires a CUDA device')
    device = torch.device(config.get('device','cuda'))
    # A bare 'cuda' follows the active GPU; set_device requires an explicit index.
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    ceiling = config.get('max_gpu_memory_gib',38)
    if ceiling > 0:
        torch.cuda.set_per_process_memory_fraction(min(1.0,ceiling*2**30/total),device)
    minimum = config.get('min_free_gpu_gib',24)
    if free < minimum * 2**30:
        raise RuntimeError(f'NF4 preflight: {free/2**30:.1f} GiB free; require at least {minimum} GiB before loading')
    FastModel = _nf4_runtime()  # Import before Transformers or PEFT.
    snapshot = _snapshot(spec)
    dtype = getattr(torch,contract['compute_dtype'])
    model, _ = FastModel.from_pretrained(model_name=snapshot, use_exact_model_name=True,
        max_seq_length=config['context_limit'], dtype=dtype, load_in_4bit=True,
        load_in_8bit=False, load_in_fp8=False, full_finetuning=False,
        fast_inference=False, text_only=True, offload_embedding=contract['offload_embedding'],
        use_gradient_checkpointing='unsloth', device_map={'':torch.cuda.current_device()},
        trust_remote_code=False, local_files_only=True)
    tokenizer = _official_tokenizer(snapshot)
    if config['context_limit'] > text_configuration(model).max_position_embeddings:
        raise ValueError('Configured context exceeds checkpoint positional support')
    quantized_count = _validate_nf4(model,contract)
    adapter = _adapter_path(spec)
    if training:
        if adapter:
            raise ValueError('Training must initialize from the declared frozen NF4 checkpoint, without an existing GRPO adapter')
        last = config.get('lora_last_layers',8) or None
        layers = text_configuration(model).num_hidden_layers
        if last is not None and not 1 <= last <= layers:
            raise ValueError('lora_last_layers must be within the language model layer count')
        model = FastModel.get_peft_model(model, r=config.get('lora_rank',8),
            lora_alpha=config.get('lora_alpha',16), lora_dropout=0.0, bias='none',
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=config.get('lora_mlp',False),
            finetune_last_n_layers=last, use_gradient_checkpointing='unsloth',
            random_state=config.get('seed',42),
            temporary_location=str(Path(config['output'])/'temporary_adapter_buffers'))
        FastModel.for_training(model,use_gradient_checkpointing='unsloth')
        # Train only LoRA tensors. Check this instead of assuming every backend
        # preparation helper preserved the frozen base and module boundaries.
        names = [n for n,p in model.named_parameters() if p.requires_grad]
        if not names or any('lora_' not in n for n in names):
            raise RuntimeError('Unexpected trainable parameters outside the LoRA adapters')
        import re
        indices = [int(m.group(1)) for n in names if (m:=re.search(r'\.layers\.(\d+)\.',n))]
        if last is not None and (not indices or min(indices) < layers-last):
            raise RuntimeError('Backend ignored the requested last-layer adapter restriction')
    else:
        if adapter:
            _verify_adapter_contract(adapter,spec,contract)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model,adapter,is_trainable=False)
        FastModel.for_inference(model)
        model.requires_grad_(False)
        model.eval()
    no_dropout(model)
    model.config.use_cache = False
    text_configuration(model).use_cache = False
    model._mosp_device = device
    model._mosp_loading = contract
    model._mosp_backend_info = dict(initial_checkpoint=dict(model=spec['model'],revision=spec['revision'],loading=contract),
        snapshot=snapshot, quantized_linear_count=quantized_count, versions=package_versions(),
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model,tokenizer


def _load_transformers(config, spec, training):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,PeftModel,get_peft_model
    model = AutoModelForCausalLM.from_pretrained(spec['model'],revision=spec['revision'],
        torch_dtype=getattr(torch,spec['loading']['compute_dtype']),attn_implementation='sdpa').to(config.get('device','cuda'))
    tokenizer = AutoTokenizer.from_pretrained(spec['model'],revision=spec['revision'])
    if config['context_limit'] > text_configuration(model).max_position_embeddings:
        raise ValueError('Configured context exceeds checkpoint positional support')
    if training:
        model = get_peft_model(model,LoraConfig(r=config.get('lora_rank',16),lora_alpha=32,
            lora_dropout=0.0,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        model.train()
    else:
        if spec.get('adapter'):
            model = PeftModel.from_pretrained(model,spec['adapter'],revision=spec.get('adapter_revision'),is_trainable=False)
        model.requires_grad_(False)
        model.eval()
    no_dropout(model)
    model.config.use_cache = False
    model._mosp_loading = spec.get('loading',loading_contract(config,spec['model']))
    return model,tokenizer


def load_model(config, spec=None, training=False):
    spec = dict(spec or initial_checkpoint(config))
    validate_revision(spec['revision'])
    contract = spec.get('loading',loading_contract(config,spec['model']))
    if 'loading' in spec and contract != loading_contract({**config,**contract},spec['model']):
        raise ValueError('Unsupported or inconsistent checkpoint loading contract')
    spec['loading'] = contract
    return (_load_nf4 if contract['backend']=='unsloth_nf4' else _load_transformers)(config,spec,training)


def save_backend_manifest(model,directory):
    """Keep the public pinned base identity when saving portable adapter files."""
    metadata = getattr(model,'_mosp_backend_info',None)
    if metadata is None:
        return
    for peft_config in model.peft_config.values():
        peft_config.base_model_name_or_path = metadata['initial_checkpoint']['model']
        peft_config.revision = metadata['initial_checkpoint']['revision']
    Path(directory).mkdir(parents=True,exist_ok=True)
    (Path(directory)/'mosp_backend.json').write_text(json.dumps(metadata,indent=2)+'\n')
