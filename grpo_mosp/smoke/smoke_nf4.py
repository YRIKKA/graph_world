"""Real pinned-model CUDA integration check, not a performance experiment.

Run with the isolated requirements.txt environment. The three model loads are
sequential: frozen, trainable adapters, then saved adapters frozen for inference.
All artifacts and caches live under the timestamped full_runs directory.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import subprocess
import sys

from ..run_artifacts import artifact_environment, model_slug, new_run_directory, validate_run_directory
from ..model_options import add_thinking_arguments, apply_thinking_arguments, thinking_options
from ..model_defaults import DEFAULT_MODEL, initial_checkpoint


def frozen_check(config, instance, directory):
    import torch
    from ..environment import Environment
    from ..memory import RULES
    from ..model_backends import load_model
    from ..policy import ConstrainedPolicy
    model, tokenizer = load_model(config, training=False)
    assert not any(p.requires_grad for p in model.parameters())
    policy = ConstrainedPolicy(model, tokenizer, config['context_limit'], 1., enable_thinking=False,history_format=config.get('history_format','full'), **thinking_options(config))
    env = Environment(instance, k=1, budget=1)
    messages = [dict(role='system',content=RULES),dict(role='user',content=env.observation())]
    action, decisions = policy.action(messages,env)
    with torch.no_grad():
        logps = policy.replay(decisions[0]).cpu()
    info = dict(action=action,backend=model._mosp_backend_info,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'smoke.json').write_text(json.dumps(info,indent=2)+'\n')
    # Returning CPU values alone allows the model allocation to be released.
    return decisions[0], logps, info


def training_check(config, instance, decision, frozen_logps):
    import torch
    from ..grpo import update
    from ..model_backends import load_model, save_backend_manifest
    from ..policy import ConstrainedPolicy
    from ..rollout import group
    model, tokenizer = load_model(config, training=True)
    policy = ConstrainedPolicy(model,tokenizer,config['context_limit'],1.,enable_thinking=False,history_format=config.get('history_format','full'), **thinking_options(config))
    from unsloth import FastModel
    # Compare both frozen policies in inference mode. Training/inference kernels
    # may differ numerically in BF16; that cross-mode difference is not GRPO gain.
    FastModel.for_inference(model)
    with torch.no_grad():
        zero_eval = policy.replay(decision).cpu()
    drift = (zero_eval-frozen_logps).abs().max().item()
    assert drift < .01, f'Zero-adapter/frozen inference log-probability drift: {drift}'
    FastModel.for_training(model,use_gradient_checkpointing='unsloth')
    current = policy.replay(decision)
    with torch.no_grad():
        sampled = policy.replay(decision).cpu()
    with model.disable_adapter(),torch.no_grad():
        reference = policy.replay(decision).cpu()
    replay_drift = (current.detach().cpu()-sampled).abs().max().item()
    reference_drift = (sampled-reference).abs().max().item()
    assert replay_drift < .01, f'Collection/replay log-probability drift: {replay_drift}'
    assert reference_drift < .0001, f'Zero-adapter/reference drift: {reference_drift}'
    branches = [i for i,a in enumerate(decision.allowed) if len(a)>1]
    assert branches, 'Smoke fixture must exercise a genuine stochastic decision'
    # Gradient connectivity check only: this diagnostic NEVER steps an optimizer.
    (-current[branches].mean()).backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum()>0 for g in gradients)
    model.zero_grad(set_to_none=True)
    del current, gradients
    tasks = group(policy,instance,k=1,size=2,budget=1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.)
    # Only genuine terminal task rewards drive this actual GRPO optimizer step.
    # Equal rewards legitimately produce a zero-advantage/KL-only group.
    result = update(policy,tasks,optimizer,config)
    assert all(not r.env.failure for r in tasks)
    saved = Path(config['output'])/'adapter'
    save_backend_manifest(model,saved)
    model.save_pretrained(saved,save_embedding_layers=False)
    tokenizer.save_pretrained(saved)
    FastModel.for_inference(model)
    with torch.no_grad():
        trained_logps = policy.replay(decision).cpu()
    report = dict(zero_adapter_max_logp_delta=drift,collection_replay_logp_delta=replay_drift,
        zero_adapter_reference_logp_delta=reference_drift,rewards=[r.reward for r in tasks],grpo=result,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
    return saved,trained_logps,report


def reload_check(config, saved, decision, trained_logps):
    import torch
    from ..model_backends import load_model
    from ..policy import ConstrainedPolicy
    model,tokenizer = load_model(config,dict(initial_checkpoint(config),adapter=str(saved)),training=False)
    assert not any(p.requires_grad for p in model.parameters())
    policy = ConstrainedPolicy(model,tokenizer,config['context_limit'],1.,enable_thinking=False,history_format=config.get('history_format','full'), **thinking_options(config))
    with torch.no_grad():
        actual = policy.replay(decision).cpu()
    drift = (actual-trained_logps).abs().max().item()
    assert drift < .01, f'Saved adapter reload log-probability drift: {drift}'
    return dict(max_logp_delta=drift,backend=model._mosp_backend_info)


def worker(stage, root, config):
    """Separate CUDA processes also release backend caches between stages."""
    import torch
    from ..policy import Decision
    from ..tests.helpers import fixture
    torch.manual_seed(17)
    runtime = root/'runtime'
    runtime.mkdir(exist_ok=True)
    if stage == 'frozen':
        decision,logps,info = frozen_check(config,fixture(),root/f'{model_slug(DEFAULT_MODEL)}_not_trained'/'smoke')
        record = dict(decision=asdict(decision),logps=logps.tolist(),info=info)
    else:
        frozen = json.loads((runtime/'smoke-frozen.json').read_text())
        decision = Decision(**frozen['decision'])
        if stage == 'training':
            saved,logps,info = training_check(config,fixture(),decision,torch.tensor(frozen['logps']))
            record = dict(adapter=str(saved),logps=logps.tolist(),info=info)
        else:
            trained = json.loads((runtime/'smoke-training.json').read_text())
            record = reload_check(config,Path(trained['adapter']),decision,torch.tensor(trained['logps']))
    (runtime/f'smoke-{stage}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'NF4 {stage} check passed',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='New full_runs/run_YYYYMMDDHHMMSS directory')
    parser.add_argument('--stage',choices=['frozen','training','reload'],help=argparse.SUPPRESS)
    add_thinking_arguments(parser)
    args = parser.parse_args()
    root = validate_run_directory(args.output) if args.output else new_run_directory(Path(__file__).resolve().parents[1]/'full_runs')
    if (root/'smoke_nf4.json').exists():
        raise FileExistsError('Completed smoke reports are immutable; choose a new run')
    with artifact_environment(root):
        config = json.loads(Path(__file__).with_name('smoke.json').read_text())
        apply_thinking_arguments(config, args)
        config.update(output=str(root/f'{model_slug(DEFAULT_MODEL)}_trained'/'smoke'),seed=17)
        Path(config['output']).mkdir(parents=True,exist_ok=True)
        if args.stage:
            worker(args.stage,root,config)
            return
        from ..graph_images import export_graph_image
        from ..tests.helpers import fixture
        (Path(config['output'])/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        instance = fixture()  # Explicit fixed plumbing diagnostic, never benchmark data.
        data = root/'data';data.mkdir(exist_ok=True)
        (data/'smoke_graph.json').write_text(json.dumps(instance.record(),indent=2)+'\n')
        export_graph_image(instance,data/'smoke_graph.png')
        started = time.perf_counter()
        # The launcher also trains and evaluates in separate CUDA processes.
        # Do not retain Unsloth training graph/offload caches during evaluation.
        for stage in ('frozen','training','reload'):
            subprocess.run([sys.executable,'-m',__package__+'.smoke_nf4',
                '--output',str(root),'--stage',stage,
                '--reasoning-effort',config['reasoning_effort'],
                '--preserve-thinking',str(config['preserve_thinking']).lower()],check=True)
        records = {stage:json.loads((root/'runtime'/f'smoke-{stage}.json').read_text())
                   for stage in ('frozen','training','reload')}
        report = dict(kind='real_checkpoint_integration_only',frozen=records['frozen']['info'],
            training=records['training']['info'],reload=records['reload'],
            decision=records['frozen']['decision'],wall_seconds=time.perf_counter()-started)
        (root/'smoke_nf4.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
