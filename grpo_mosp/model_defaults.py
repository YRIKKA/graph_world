"""Dependency-free defaults and the effective frozen-base loading contract."""

DEFAULT_MODEL = 'unsloth/Qwen3.8-27B-unsloth-bnb-4bit'
DEFAULT_REVISION = '8aa5f05d26b7205477066e1449e0af13f762a299'


def loading_contract(config, model=None):
    """Quantization/template settings belong to initialization identity too.

    Old model/revision-only Qwen2.5 configs remain supported by the legacy loader.
    A change in the NF4 compute dtype or template cannot masquerade as GRPO gain.
    """
    model = model or config.get('model', DEFAULT_MODEL)
    backend = config.get('backend','auto')
    if backend == 'auto':
        backend = 'unsloth_nf4' if model == DEFAULT_MODEL else 'transformers'
    if backend not in ('unsloth_nf4', 'transformers'):
        raise ValueError(f'Unknown model backend: {backend}')
    if backend == 'transformers':
        return dict(backend=backend, compute_dtype=config.get('compute_dtype','bfloat16'), quantization='none',
                    enable_thinking=config.get('enable_thinking', False))
    return dict(backend=backend, quantization='nf4', double_quant=True,
        compute_dtype=config.get('compute_dtype', 'bfloat16'),
        text_only=True, enable_thinking=False, fast_inference=False, torch_compile=False,
        offload_embedding=config.get('offload_embedding', True))


def initial_checkpoint(config):
    result = dict(model=config['model'], revision=config['revision'])
    contract = loading_contract(config)
    if contract['backend'] == 'unsloth_nf4' or 'backend' in config:
        result['loading'] = contract
    return result
