"""Single-GPU LoRA GRPO entry point; never updates while collecting a group."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from .dataset import load
from .training_options import add_training_arguments, apply_training_arguments
from .early_stopping import Plateau, validation_instances, validate
from .model_options import add_thinking_arguments, apply_thinking_arguments, thinking_options
from .policy import load_policy
from .rollout import group
from .grpo import update
from .model_defaults import initial_checkpoint
from .model_backends import save_backend_manifest, package_versions
from .reporting import Progress, Totals, training_metrics, task_tables, table
from .run_artifacts import console_log
from .pipeline_timing import timed_component
from .graph_sampling import GraphSampler, add_sampling_arguments, sampling_options, SAMPLING_DEFAULTS


def traversal_budget(config, k):
    """Resolve the entire task's attempt budget, not a per-move episode limit."""
    budget = config.get('traversal_budget')
    if budget is None:
        budget = config.get('attempt_multiplier', 8) * k
    if not isinstance(budget, int) or budget < k:
        raise ValueError('Traversal budget must be an integer >= k')
    return budget


def main(config_path, args=None):
    config = json.loads(Path(config_path).read_text())
    apply_thinking_arguments(config, args)
    apply_training_arguments(config, args)
    if args is not None:
        config.update({key:getattr(args,key) for key in SAMPLING_DEFAULTS if hasattr(args,key)})
    if args is not None and getattr(args,"validation_dataset",None):
        config["validation_dataset"] = args.validation_dataset
    with console_log(Path(config['output'])/'log.txt'):
        try:
            return train(config)
        except KeyboardInterrupt:
            print('Training interrupted',flush=True)
            raise SystemExit(130) from None
        except Exception:
            import traceback
            traceback.print_exc()
            raise SystemExit(1) from None


def train(config):
    sampling_options(config)
    config.update(thinking_options(config))
    apply_training_arguments(config)
    if config.get('shaping') is not None:
        from .shaping import options
        options(config['shaping'])
    if config.get('moo_rewards') is not None:
        from .experiment_moo_rewards.moo_rewards import validate as validate_moo_rewards
        validate_moo_rewards(config['moo_rewards'])
        if config.get('shaping') is not None:
            raise ValueError('MOO terminal rewards cannot be combined with discovery shaping')
    actual_initial = initial_checkpoint(config)
    if config.get('initial_checkpoint', actual_initial) != actual_initial:
        raise ValueError('This trainer initializes from model/revision; initial_checkpoint must describe those actual weights')
    config['initial_checkpoint'] = actual_initial
    started = time.perf_counter()
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    instances = load(config['dataset'])
    if not instances:
        raise ValueError('Empty dataset')
    graph_pools=None
    if config.get('graph_curriculum') is not None:
        from .graph_curriculum import TrainingPools
        graph_pools=TrainingPools(config,instances)
    sampler=graph_pools or GraphSampler(instances,config,config['seed'] ^ 0x53414D50)
    validation = validation_instances(config,instances) if config['early_stopping'] else []
    torch.cuda.reset_peak_memory_stats()
    print(f'\nModel: {config["model"]} | training seed: {config["seed"]} | objective: {config["objective"]}',flush=True)
    if config.get('shaping') is not None:
        print(f'Shaping condition: {config.get("shaping_condition","custom")} | '
              f'{json.dumps(config["shaping"],sort_keys=True)}',flush=True)
    if config.get('moo_rewards') is not None:
        print(f'MOO terminal reward: {json.dumps(config["moo_rewards"], sort_keys=True)}', flush=True)
    progress = Progress(f'{config["model"]} | seed {config["seed"]} | training iterations',config['steps'])
    totals,loss_sum = Totals(),0.
    with timed_component(f'Training model loading | seed {config["seed"]}') as loading_time:
        policy = load_policy(config)
        if policy.device.type=='cuda':
            torch.cuda.synchronize(policy.device)
    optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=0.0)
    output = Path(config['output'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    zero_groups = context_limit_failures = 0
    generated_tokens = edge_traversals = 0
    rollout_seconds = optimization_seconds = 0.
    optimizer_steps_completed=0
    (output/'backend.json').write_text(json.dumps(getattr(policy.model,'_mosp_backend_info',{}),indent=2)+'\n')
    monitor = Plateau(config['early_stopping_patience'],config['early_stopping_min_delta'])
    checks, stopped, completed = [], False, 0
    validation_seconds = 0.

    def export_adapter():
        # Export only LoRA tensors; base embeddings/vocabulary remain unchanged.
        with timed_component('Selected LoRA adapter and optimizer/RNG export'):
            save_backend_manifest(policy.model,output/'adapter')
            policy.model.save_pretrained(output/'adapter',save_embedding_layers=False)
            policy.tokenizer.save_pretrained(output/'adapter')
            # The optimizer/RNG snapshot belongs to this exact selected adapter.
            torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),graph_sampler=sampler.state(),
                            **({'graph_rng':graph_pools.rng.getstate()} if graph_pools else {})),output/'training_state.pt')

    def check_validation(iteration):
        nonlocal validation_seconds
        start = time.perf_counter()
        with timed_component(f'Early-stopping validation | iteration {iteration}'):
            result = validate(policy,validation,config)
        improved = monitor.observe(result['mean_S'],iteration)
        if improved:
            export_adapter()
        record = dict(iteration=iteration,**result,best_step=monitor.best_step,best_S=monitor.best,
                      stale_checks=monitor.stale,adapter_exported=improved)
        checks.append(record)
        with (output/'validation_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        validation_seconds += time.perf_counter()-start
        table('Early stopping | validation at B=8k',['Iteration','Mean S','Best S','Best iteration','Stale checks'],
              [[iteration,f"{result['mean_S']:.4f}",f'{monitor.best:.4f}',monitor.best_step,
                f"{monitor.stale}/{config['early_stopping_patience']}"]])

    if config['early_stopping']:
        # Include the initialization: training may never improve validation.
        check_validation(0)
    with (output/'metrics.jsonl').open('w') as log:
        last_graph_stage=None
        for step in range(config['steps']):
            step_started = time.perf_counter()
            k = config['curriculum'][min(len(config['curriculum'])-1,step*len(config['curriculum'])//config['steps'])]
            graph_info={}
            if graph_pools:
                instance,graph_info=graph_pools.choose(step)
                if graph_info['graph_stage']!=last_graph_stage:
                    last_graph_stage=graph_info['graph_stage']
                    # Earlier-stage plateaus must not immediately stop training
                    # when a harder graph stage begins. Keep the best adapter,
                    # but restart patience against the same fixed validation S.
                    monitor.stale=0
                    monitor.reference=monitor.best
                    print(f'Entering graph curriculum stage {last_graph_stage+1}/{len(graph_pools.pools)}; '
                          'early-stopping patience reset.',flush=True)
            else:
                instance,sampling_info = sampler.choose()
            # A traversal attempt is NOT a GRPO episode. Every rollout gets B
            # attempts on this same fixed graph, with persistent task memory.
            budget = traversal_budget(config, k)
            rollout_started = time.perf_counter()
            tasks = group(policy,instance,k,config['group_size'],budget=budget,objective=config['objective'],shaping=config.get('shaping'))
            if config.get('moo_rewards') is not None:
                from .experiment_moo_rewards.moo_rewards import apply_rewards
                apply_rewards(tasks, instance, k, config['moo_rewards'])
            if policy.device.type=='cuda':
                torch.cuda.synchronize(policy.device)
            rollout_elapsed = time.perf_counter()-rollout_started
            rollout_seconds += rollout_elapsed
            generated_tokens += sum(len(d.tokens) for r in tasks for d in r.decisions)
            edge_traversals += sum(r.env.moves for r in tasks)
            optimization_started = time.perf_counter()
            stats = update(policy,tasks,optimizer,config)
            stats['optimizer_step_before']=optimizer_steps_completed
            optimizer_steps_completed+=stats['optimizer_steps']
            stats['optimizer_step']=optimizer_steps_completed
            if policy.device.type=='cuda':
                torch.cuda.synchronize(policy.device)
            optimization_elapsed = time.perf_counter()-optimization_started
            optimization_seconds += optimization_elapsed
            stats.update(rollout_seconds=rollout_elapsed,optimization_seconds=optimization_elapsed)
            completed = step+1
            batch = Totals()
            for task in tasks:
                measured = training_metrics(task)
                batch.add(measured)
                totals.add(measured)
            loss_sum += stats['loss']
            stats.update(tasks=len(tasks),total_reward=batch.sums['reward'],cumulative_reward=totals.sums['reward'],
                         performance=batch.record()['means'],elapsed_seconds=time.perf_counter()-step_started)
            if config.get('moo_rewards') is not None:
                for field in ('reward_components', 'reward_contributions'):
                    stats[field] = {name: sum(task.shaping[field][name] for task in tasks) / len(tasks)
                                    for name in config['moo_rewards']['functions']}
            zero_groups += stats['zero_variance_group']
            context_limit_failures += sum(r.status=='context_limit' for r in tasks)
            stats.update(step=step,k=k,B=budget,seed=instance.seed,zero_variance_group_rate=zero_groups/(step+1),
                         protocol_failure_rate=sum(r.env.failure for r in tasks)/len(tasks),
                         context_limit_failure_rate=sum(r.status=='context_limit' for r in tasks)/len(tasks),
                         max_context_tokens=max([len(d.context)+len(d.tokens) for r in tasks for d in r.decisions]+
                             [r.limit_details['tokens'] for r in tasks if r.limit_details]+[0]))
            terminal_scores=[(r.shaping or {}).get('terminal_reward',r.reward) for r in tasks]
            sampling_feedback=sampler.observe(instance,[r.reward for r in tasks],terminal_scores)
            stats['sampling_feedback']=sampling_feedback
            stats.update(graph_info if graph_pools else sampling_info)
            log.write(json.dumps(stats)+'\n')
            log.flush()
            progress.advance()
            task_tables(f'{config["model"]} | seed {config["seed"]} | iteration {step+1}/{config["steps"]} | k={k}, B={budget}, graph seed={instance.seed}',
                        {'This iteration':batch.record(),'Run so far':totals.record()},inference=False)
            table('Optimization / resource metrics',['Metric','Value'],[
                ['Rollout generation seconds',f'{rollout_elapsed:.2f}'],
                ['GRPO optimization seconds',f'{optimization_elapsed:.2f}'],
                ['GRPO loss',f'{stats["loss"]:.6f}'],['Zero-variance groups',f'{stats["zero_variance_group_rate"]:.1%}'],
                ['Adaptive next graph weight',f"{sampling_feedback['sampling_next_weight']:.3f}"],
                ['Graph zero-variance EMA',f"{sampling_feedback['sampling_zero_variance_ema']:.3f}"],
                ['Maximum context tokens',stats['max_context_tokens']],['Iteration seconds',f'{stats["elapsed_seconds"]:.2f}']])
            if graph_info:
                table('Graph curriculum',['Scheduled stage','Sampled pool','Earlier-stage rehearsal'],
                      [[graph_info['graph_stage']+1,graph_info['graph_pool_stage']+1,graph_info['rehearsal']]])
            if stats['context_limit_failure_rate']:
                print('Context-limited tasks retained with reward 0; see rollout limit_details. '
                      'Use state_actions histories or reduce the graph/traversal budget if this rate persists.',flush=True)
            # Persist complete trajectories, grammar states and old probabilities for audit.
            from dataclasses import asdict
            (output/f'rollout-{step}.json').write_text(json.dumps([
                dict(reward=r.reward,shaping=r.shaping,status=r.status,limit_details=r.limit_details,messages=r.messages,decisions=[asdict(d) for d in r.decisions]) for r in tasks]))
            if config['early_stopping'] and (completed % config['early_stopping_interval']==0 or completed==config['steps']):
                check_validation(completed)
                # Give every curriculum stage a chance before stopping. The
                # metric itself always covers the same fixed validation k set.
                final_stage = min(len(config['curriculum'])-1,step*len(config['curriculum'])//config['steps']) == len(config['curriculum'])-1
                final_stage=final_stage and (not graph_pools or graph_info['graph_stage']==len(graph_pools.pools)-1)
                if monitor.stale >= config['early_stopping_patience'] and final_stage and completed < config['steps']:
                    stopped = True
                    print(f'Early stopping at iteration {completed}: validation plateau. '
                          f'Exported best adapter from iteration {monitor.best_step} (S={monitor.best:.4f}).',flush=True)
                    break
    if not config['early_stopping']:
        export_adapter()
    stopping = dict(enabled=config['early_stopping'],stopped_early=stopped,completed_steps=completed,
        planned_steps=config['steps'],stop_reason='validation_plateau' if stopped else 'maximum_steps' if checks else 'early_stopping_disabled',
        selected_step=monitor.best_step if checks else completed,
        best_validation_S=monitor.best if checks else None,validation_seconds=validation_seconds,
        validation_dataset=config.get('validation_dataset'),
        criterion='maximum mean validation S at B=8k; earliest iteration wins ties',
        validation_dataset_sha256=hashlib.sha256(Path(config['validation_dataset']).read_bytes()).hexdigest() if checks else None,
        graph_ids=sorted({t['graph_id'] for t in checks[0]['tasks']}) if checks else [],
        checks=[{key:value for key,value in r.items() if key!='tasks'} for r in checks])
    (output/'early_stopping.json').write_text(json.dumps(stopping,indent=2)+'\n')
    if checks:
        print(f'Validation-selected adapter: iteration {monitor.best_step}, S={monitor.best:.4f}; '
              f'completed {completed}/{config["steps"]} training iterations.',flush=True)
    (output/'graph_sampling.json').write_text(json.dumps(sampler.audit(),indent=2)+'\n')
    # Record training cost separately from every downstream inference measurement.
    torch.cuda.synchronize()
    compute = dict(
        wall_seconds=time.perf_counter()-started, gpu=torch.cuda.get_device_name(),
        gpu_count=1, steps=completed, planned_steps=config['steps'], optimizer_steps=optimizer_steps_completed,complete_tasks=completed*config['group_size'],
        model_loading_seconds=loading_time['seconds'],rollout_seconds=rollout_seconds,optimization_seconds=optimization_seconds,
        validation_seconds=validation_seconds,validation_tasks=sum(len(c['tasks']) for c in checks),
        validation_generated_tokens=sum(t['metrics']['generated_tokens'] for c in checks for t in c['tasks']),
        validation_edge_traversals=sum(t['metrics']['edge_traversals'] for c in checks for t in c['tasks']),
        generated_tokens=generated_tokens, edge_traversals=edge_traversals,
        context_limit_failures=context_limit_failures,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        runtime=package_versions(), initial_checkpoint=actual_initial,
        scope='model loading, training rollouts, optimization, validation, reporting, rollout serialization and selected checkpoint exports; excludes downstream evaluation')
    (output/'training_compute.json').write_text(json.dumps(compute,indent=2)+'\n')
    summary = dict(totals.record(),model=config['model'],training_seed=config['seed'],objective=config['objective'],
                   iterations=completed,optimizer_steps=optimizer_steps_completed,planned_iterations=config['steps'],early_stopping=stopping,mean_loss=loss_sum/completed,wall_seconds=compute['wall_seconds'],
                   zero_variance_group_rate=zero_groups/completed,training_compute=compute)
    (output/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    task_tables(f'{config["model"]} | training run complete | seed {config["seed"]}',{'Final run totals':summary},inference=False)
    print(f'Training seconds: {compute["wall_seconds"]:.2f}; peak GPU allocation: {compute["peak_allocated_bytes"]/2**30:.2f} GiB.\n'
          'These are changing-policy training rollout statistics; held-out checkpoint evaluation is reported separately.',flush=True)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='jm_test/graph_world/grpo_mosp/configs/default.json')
    add_thinking_arguments(parser)
    add_sampling_arguments(parser)
    add_training_arguments(parser)
    parser.add_argument("--validation-dataset",help="Fixed validation JSONL required when early stopping is enabled")
    args = parser.parse_args()
    main(args.config, args)
