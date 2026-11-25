"""Small held-out shaping comparison: one k, B=8k, shared data, no budget sweep.

Reuse the original graph suite, constrained task evaluator, cost metrics and
family averaging. The frozen base is evaluated once. Conditions are experiments,
not independent training seeds of a single pooled training procedure.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import gc
import json
from pathlib import Path
import time

import numpy as np

from . import eval_baselines as full
from .dataset import Instance
from .model_defaults import initial_checkpoint
from .model_options import add_thinking_arguments, apply_thinking_arguments
from .pipeline_timing import timed_component
from .reporting import Progress, Totals, task_tables, table
from .run_artifacts import console_log

MODES=('independent_exploration','shared_transcript_selection')
SCOPES=[(d,[d]) for d in full.DOMAINS]+[('distribution_shift_macro',list(full.DOMAINS[1:]))]


def validate_config(config):
    full.validate_config(config)
    if len(config['ks'])!=1 or config['budget_multipliers']!=[8] or config['relabelings']!=1:
        raise ValueError('Light evaluation requires one k, budget [8] and one relabeling')
    if not config['modes'] or len(set(config['modes']))!=len(config['modes']) or set(config['modes'])-set(MODES):
        raise ValueError('Invalid lightweight evaluation modes')
    from .shaping import signal_names
    if not config['conditions']:raise ValueError('Invalid shaping conditions')
    if config.get('reward_experiment') == 'moo_rewards':
        from .experiment_moo_rewards.moo_rewards import COMBINATIONS
        if config['conditions'] != list(COMBINATIONS):
            raise ValueError('MOO experiment requires all eight canonical conditions')
    else:
        for condition in config['conditions']:signal_names(condition)
    if len(set(config['conditions']))!=len(config['conditions']):
        raise ValueError('Duplicate shaping condition')
    if not config['condition_seeds'] or len(set(config['condition_seeds']))!=len(config['condition_seeds']):
        raise ValueError('Training seeds must be nonempty and unique')
    for condition in config['conditions']:
        seeds=[r['seed'] for r in config['training_runs'] if r['condition']==condition]
        if sorted(seeds)!=sorted(config['condition_seeds']) or len(set(seeds))!=len(seeds):
            raise ValueError('Every condition must use the same unique training seeds')


def verify_suite(config, output, check_code=False):
    output=Path(output)
    manifest=json.loads((output/'suite_manifest.json').read_text())
    if manifest['config_id']!=full.digest(config):
        raise ValueError('Immutable shaping evaluation configuration changed')
    for name,key in [('suite.jsonl','suite_id'),('transcripts.jsonl','transcripts_id')]:
        if full.file_hash(output/name)!=manifest[key]:raise ValueError(f'Changed {name}')
    for path,expected in manifest['excluded_files'].items():
        if full.file_hash(path)!=expected:raise ValueError('Changed training/validation data')
    for path,expected in manifest['graph_images'].items():
        if full.file_hash(output/path)!=expected:raise ValueError('Changed graph image')
    if check_code:
        full.verify_protocol_sources(
            {**manifest['protocol_sources'], **manifest['shaping_sources']},
            'Changed evaluation protocol source',
        )
    return manifest


def prepare(config,output):
    output=Path(output)
    if (output/'suite_manifest.json').exists():
        print('Reusing fixed lightweight held-out suite',flush=True)
        return verify_suite(config,output,True)
    result=full.prepare(config,output)
    result['shaping_sources']={name:full.file_hash(Path(__file__).parent/name) for name in
        ('eval_shaping.py','shaping.py','rollout.py','train.py','graph_curriculum.py','graph_sampling.py')}
    if config.get('reward_experiment') == 'moo_rewards':
        result['shaping_sources'].update({name: full.file_hash(Path(__file__).parent/name)
            for name in ('experiment_moo_rewards/moo_rewards.py', 'rewards.py',
                         'experiment_moo_rewards/experiment_moo_rewards.py')})
    full.write_json(output/'suite_manifest.json',result)
    return result


def validate_models(config):
    """Check actual completed training provenance before loading any GPU weights."""
    expected=config['initial_checkpoint']
    details={}
    common_training=None
    for run in config['training_runs']:
        training=json.loads(Path(run['training_config']).read_text())
        if initial_checkpoint(training)!=expected or run['post_checkpoint'].get('loading')!=expected.get('loading'):
            raise ValueError('All shaping conditions must initialize from the exact frozen comparison base')
        if training['seed']!=run['seed']:
            raise ValueError('Training seed does not match the declared condition')
        from .shaping import SIGNALS, options, signal_names
        if config.get('reward_experiment') == 'moo_rewards':
            from .experiment_moo_rewards.moo_rewards import options as reward_options, validate as validate_rewards
            actual = training.get('moo_rewards', {})
            expected_rewards = reward_options(run['condition'], config['hypervolume_backend'])
            if actual != expected_rewards or training.get('moo_reward_condition') != run['condition']:
                raise ValueError('Training MOO rewards do not match the declared condition')
            validate_rewards(actual)
            if training.get('shaping') is not None:
                raise ValueError('MOO experiment must not use discovery shaping')
        else:
            if training.get('moo_rewards') is not None:
                raise ValueError('Shaping experiment must not use MOO terminal rewards')
            selected=signal_names(run['condition'])
            actual=options(training['shaping'])
            if any(actual[name]!=(name in selected) for name in SIGNALS):
                raise ValueError('Training shaping condition does not match its label')
        if training['objective']!='pareto' or training['validation_dataset']!=config['validation_dataset']:
            raise ValueError('Training must use Pareto reward and the declared validation dataset')
        for key in ('context_limit','temperature','history_format','reasoning_effort','preserve_thinking'):
            if training.get(key)!=config.get(key):raise ValueError(f'Different train/eval control: {key}')
        if json.loads(Path(run['input_config']).read_text())!=training:
            raise ValueError('Actual training configuration differs from the frozen input configuration')
        directory=Path(training['output'])
        if Path(run['post_checkpoint']['adapter']).resolve()!=(directory/'adapter').resolve():
            raise ValueError('Adapter path does not belong to its declared training run')
        if Path(training['dataset']).resolve()!=Path(config['excluded_datasets'][0]).resolve():
            raise ValueError('Every condition must use the same training dataset')
        # All controls are shared except the deliberately varied factors.
        varied = {'output', 'shaping', 'shaping_condition', 'seed'}
        if config.get('reward_experiment') == 'moo_rewards':
            varied.update(('moo_rewards', 'moo_reward_condition'))
        common={k:v for k,v in training.items() if k not in varied}
        if common_training is not None and common_training!=common:
            raise ValueError('Training hyperparameters differ between shaping conditions')
        common_training=common
        if not (directory/'training_state.pt').is_file() or not (directory/'training_summary.json').is_file():
            raise ValueError('Incomplete shaping training run')
        stopping=json.loads((directory/'early_stopping.json').read_text())
        if stopping['enabled']!=training['early_stopping']:
            raise ValueError('Early-stopping provenance differs from training configuration')
        if stopping['enabled'] and stopping['validation_dataset_sha256']!=full.file_hash(config['validation_dataset']):
            raise ValueError('Validation data changed after checkpoint selection')
        details[run['id']]=dict(training_config=training,selection=stopping,
                               training_compute=json.loads(Path(run['training_compute']).read_text()))
    return details


def run(config,output,loader=None):
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    output=Path(output)
    manifest=verify_suite(config,output,True)
    details=validate_models(config)
    specs=full.checkpoint_specs(config)
    identities={key:full.checkpoint_id(spec) for key,spec in specs.items()}
    if (output/'completed.json').exists():
        checked_results(config,output)
        if json.loads((output/'evaluation_manifest.json').read_text())['checkpoints']!=identities:
            raise ValueError('Checkpoint artifacts changed')
        print('Reusing completed lightweight evaluation; no additional model inference',flush=True)
        return
    if (output/'tasks.jsonl').exists():raise FileExistsError('Incomplete evaluation; choose a fresh experiment')
    suite=full.read_jsonl(output/'suite.jsonl')
    transcripts={tuple(t['key']):t for t in full.read_jsonl(output/'transcripts.jsonl')}
    full.write_json(output/'evaluation_manifest.json',dict(config_id=full.digest(config),suite_id=manifest['suite_id'],
        checkpoints=identities,training=details,unshaped_reward='S',
        controls='same suite, prompts/history, decoder, temperature and resource budgets for every checkpoint',
        note='Lightweight pilot. One training seed does not measure training-procedure variability.'))
    loader=loader or full.load_frozen
    progress=Progress('Lightweight evaluation | checkpoints',len(specs))
    loading=[]
    with (output/'tasks.jsonl').open('w') as combined:
        for identifier,spec in specs.items():
            directory=full.model_result_directory(config,output,identifier)
            directory.mkdir(parents=True,exist_ok=True)
            with console_log(directory/'log.txt'),timed_component(f'Light checkpoint | {identifier}'):
                with timed_component('Model loading'):
                    start=time.perf_counter();policy=loader(spec,config)
                    policy.model.requires_grad_(False);policy.model.eval();full.sync(config['device'])
                    full.write_json(directory/'backend.json',getattr(policy.model,'_mosp_backend_info',{}))
                    load_seconds=time.perf_counter()-start
                # Fixed warmup is separated from per-task inference costs.
                with timed_component('Model warmup'),torch.inference_mode():
                    start=time.perf_counter()
                    tokens=policy.prompt([dict(role='user',content='Ready.')])
                    policy.model(input_ids=torch.tensor([tokens],device=policy.device),use_cache=False,logits_to_keep=1)
                    full.sync(config['device'])
                    warmup_seconds=time.perf_counter()-start
                totals=defaultdict(Totals)
                task_progress=Progress(f'{identifier} | held-out tasks',len(suite)*config['repetitions']*len(config['modes']))
                start=time.perf_counter()
                with (directory/'tasks.jsonl').open('w') as individual:
                    for row in suite:
                        instance=Instance.restore(row['instance'])
                        k=config['ks'][0]
                        for repetition in range(config['repetitions']):
                            key=(row['graph_id'],k,8,repetition)
                            for mode in config['modes']:
                                # Common sampling seed across checkpoints. Every
                                # model still explores independently with fresh memory.
                                seed=int(full.digest(['shaping-eval',config['evaluation_seed'],mode,key])[:15],16)
                                task=full.evaluate_task(policy,instance,k,8,seed,config,
                                    transcripts[key] if mode=='shared_transcript_selection' else None)
                                task.update(model_id=identifier,checkpoint_id=identities[identifier]['checkpoint_id'],
                                    graph_id=row['graph_id'],family_id=row['family_id'],variant=row['variant'],
                                    domain=row['domain'],difficulty=row['difficulty'],k=k,B=8*k,budget_multiplier=8,
                                    repetition=repetition,sampling_seed=seed,mode=mode)
                                line=json.dumps(task)+'\n';combined.write(line);combined.flush();individual.write(line);individual.flush()
                                totals[mode].add(dict(task['metrics'],reward=task['metrics']['S']))
                                task_progress.advance()
                inference_seconds=time.perf_counter()-start
                task_tables(f'{identifier} | unshaped held-out evaluation',{m:t.record() for m,t in totals.items()},raw_hypervolume=False)
                loading.append(dict(model_id=identifier,load_seconds=load_seconds,warmup_seconds=warmup_seconds,evaluation_wall_seconds=inference_seconds))
                full.write_json(directory/'evaluation_manifest.json',dict(checkpoint=identities[identifier],
                    suite_id=manifest['suite_id'],config_id=full.digest(config),timing=loading[-1]))
                full.write_json(directory/'completed.json',dict(tasks_sha256=full.file_hash(directory/'tasks.jsonl')))
                del policy;gc.collect()
                if str(config['device']).startswith('cuda'):torch.cuda.empty_cache()
            progress.advance()
    full.write_json(output/'inference_loading.json',loading)
    full.write_json(output/'completed.json',dict(tasks_sha256=full.file_hash(output/'tasks.jsonl'),
        task_count=len(suite)*config['repetitions']*len(config['modes'])*len(specs)))


def checked_results(config,output):
    output=Path(output);verify_suite(config,output)
    completed=json.loads((output/'completed.json').read_text())
    if full.file_hash(output/'tasks.jsonl')!=completed['tasks_sha256']:raise ValueError('Changed evaluation results')
    rows=full.read_jsonl(output/'tasks.jsonl');suite=full.read_jsonl(output/'suite.jsonl')
    # Reuse the original full cell/metadata/finite-metric validation per mode.
    # Its expected modes are fixed at two, so perform the small cell check here.
    expected={(m,g['graph_id'],q,mode) for m in full.checkpoint_specs(config) for g in suite
              for q in range(config['repetitions']) for mode in config['modes']}
    seen=set();metadata={g['graph_id']:g for g in suite}
    identities=json.loads((output/'evaluation_manifest.json').read_text())['checkpoints']
    for row in rows:
        key=(row['model_id'],row['graph_id'],row['repetition'],row['mode'])
        if key in seen or key not in expected:raise ValueError('Duplicate/unexpected evaluation cell')
        if row['checkpoint_id']!=identities[row['model_id']]['checkpoint_id']:
            raise ValueError('Result checkpoint identity does not match the evaluation manifest')
        if row['k']!=config['ks'][0] or row['B']!=8*row['k'] or row['budget_multiplier']!=8:raise ValueError('Changed task budget')
        if any(row[k]!=metadata[row['graph_id']][k] for k in ('family_id','domain','difficulty','variant')):raise ValueError('Changed graph metadata')
        if not all(np.isfinite(row['metrics'][m]) for m in full.METRICS):raise ValueError('Nonfinite metric')
        seen.add(key)
    if seen!=expected or len(rows)!=completed['task_count']:raise ValueError('Incomplete shaping evaluation')
    return rows,suite


def paired_estimate(means,metadata,post_ids,reference_ids,config,scope):
    """Crossed paired-seed x graph-cluster bootstrap, after repetition averaging.

    Conditions with the same seed are paired; all checkpoints share every graph
    resample. The frozen base is broadcast over seed indices, never counted as
    independent baseline training runs.
    """
    domains=dict(SCOPES)[scope]
    families=sorted(f for f,g in metadata.items() if g['domain'] in domains)
    groups={(d,h):[i for i,f in enumerate(families) if metadata[f]['domain']==d and metadata[f]['difficulty']==h]
            for d in domains for h in ('easy','hard')}
    if any(not values for values in groups.values()):raise ValueError('Missing graph stratum')
    weights={key:config['difficulty_weights'][key[1]]/len(domains) for key in groups}
    post=np.array([[means[m,f] for f in families] for m in post_ids])
    reference=np.array([[means[m,f] for f in families] for m in reference_ids])
    if len(reference)==1:reference=np.repeat(reference,len(post),axis=0)
    if reference.shape!=post.shape:raise ValueError('Reference condition seeds do not pair')
    def estimate(seeds,draw):
        before=sum(weights[g]*reference[seeds][:,indices].mean(axis=(0,1)) for g,indices in draw.items())
        after=sum(weights[g]*post[seeds][:,indices].mean(axis=(0,1)) for g,indices in draw.items())
        return np.stack([before,after,after-before])
    point=estimate(np.arange(len(post)),groups)
    rng=np.random.default_rng(int(full.digest([config['bootstrap_seed'],scope,post_ids,reference_ids])[:15],16))
    draws=[]
    for _ in range(config['bootstrap_samples']):
        seeds=rng.choice(len(post),len(post),replace=True)
        draw={g:rng.choice(ids,len(ids),replace=True) for g,ids in groups.items()}
        draws.append(estimate(seeds,draw))
    low,high=np.quantile(draws,[.025,.975],axis=0)
    return point,low,high


def summarize(config,output,plots=True):
    baseline = 'valid' if config.get('reward_experiment') == 'moo_rewards' else 'none'
    baselines = ('valid_per_path', 'valid') if baseline == 'valid' else ('none',)
    baseline_label = 'own validity baseline' if baseline == 'valid' else 'no shaping'
    output=Path(output);rows,suite=checked_results(config,output)
    metadata={g['family_id']:g for g in suite}
    records=[]
    with timed_component('Lightweight paired comparisons'):
        for mode in config['modes']:
            estimates={}
            means=full.family_means([r for r in rows if r['mode']==mode],config['ks'])
            by_condition={c:[r['id'] for seed in config['condition_seeds'] for r in config['training_runs']
                            if r['condition']==c and r['seed']==seed] for c in config['conditions']}
            entities=[(c,ids) for c,ids in by_condition.items()]+[(r['id'],[r['id']]) for r in config['training_runs']]
            for scope,_ in SCOPES:
                for entity,ids in entities:
                    references=[('initial',['initial'])]
                    for reference_baseline in baselines:
                        if reference_baseline in by_condition and entity!=reference_baseline:
                            seeds=[r['seed'] for r in config['training_runs'] if r['id'] in ids]
                            refs=by_condition[reference_baseline] if entity in by_condition else [r['id'] for r in config['training_runs'] if r['condition']==reference_baseline and r['seed'] in seeds]
                            references.append((reference_baseline,refs))
                    for reference,refs in references:
                        cache_key=(scope,tuple(ids),tuple(refs))
                        if cache_key not in estimates:
                            estimates[cache_key]=paired_estimate(means,metadata,ids,refs,config,scope)
                        point,low,high=estimates[cache_key]
                        for i,metric in enumerate(full.METRICS):
                            if scope=='distribution_shift_macro' and metric=='hypervolume':continue
                            records.append(dict(mode=mode,scope=scope,condition=entity,reference=reference,metric=metric,
                                before=float(point[0,i]),after=float(point[1,i]),difference=float(point[2,i]),
                                ci_low=float(low[2,i]),ci_high=float(high[2,i]),before_ci_low=float(low[0,i]),before_ci_high=float(high[0,i]),
                                after_ci_low=float(low[1,i]),after_ci_high=float(high[1,i]),
                                training_seeds=len(ids),graph_families=sum(g['domain'] in dict(SCOPES)[scope] for g in metadata.values())))
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    headline=[]
    for mode in config['modes']:
        for scope,_ in SCOPES:
            selected=[r for r in records if r['mode']==mode and r['scope']==scope and r['condition'] in config['conditions'] and r['metric']=='S' and r['reference']=='initial']
            lines=[]
            for row in selected:
                own_baseline = 'valid_per_path' if baseline == 'valid' and row['condition'].startswith('valid_per_path') else baseline
                vs_none=next((r for r in records if r['mode']==mode and r['scope']==scope and r['condition']==row['condition'] and r['metric']=='S' and r['reference']==own_baseline),None)
                delta='reference' if row['condition']==own_baseline else f"{100*vs_none['difference']:+.2f} [{100*vs_none['ci_low']:+.2f}, {100*vs_none['ci_high']:+.2f}]" if vs_none else 'N/A'
                lines.append([row['condition'],f"{100*row['before']:.2f}%",f"{100*row['after']:.2f}%",
                    f"{100*row['difference']:+.2f} [{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}]",delta])
            title=f'{config["initial_checkpoint"]["model"]} | {("MOO REWARD" if baseline == "valid" else "SHAPING")} COMPARISON | {mode} | {scope} | k={config["ks"][0]}, B=8k'
            table(title,['Condition','Frozen S','Trained S','Change vs frozen (pp), 95% CI',f'Change vs {baseline_label} (pp), 95% CI'],lines)
            headline.extend([title,'',f'| Condition | Frozen S | Trained S | Change vs frozen (pp), 95% CI | Change vs {baseline_label} (pp), 95% CI |',
                '|---|---:|---:|---:|---:|',*['| '+' | '.join(line)+' |' for line in lines],''])
    note='Unshaped held-out S only. Repetitions are averaged within graphs; graph identities and paired training seeds are bootstrapped separately. One seed cannot measure training-procedure variability. This is a lightweight pilot; do not select shaping weights using these test results.'
    print(note,flush=True);headline.append(note)
    (output/'comparison.md').write_text('\n'.join(headline)+'\n')
    full.write_json(output/'summary.json',dict(comparisons=records,note=note))
    if plots:
        from .figures import evaluation_bars
        with timed_component('All-checkpoint metric bar charts'):
            evaluation_bars(config,records,output)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','run','summarize','all'])
    p.add_argument('--config',required=True);p.add_argument('--output');p.add_argument('--no-plots',action='store_true')
    add_thinking_arguments(p);args=p.parse_args()
    config=apply_thinking_arguments(json.loads(Path(args.config).read_text()),args)
    validate_config(config);output=Path(args.output or config['output'])
    with console_log(output/'log.txt'):
        try:
            for stage in ('prepare','run','summarize') if args.stage=='all' else (args.stage,):
                with timed_component(f'Light shaping evaluation | {stage}'):
                    if stage=='prepare':prepare(config,output)
                    elif stage=='run':run(config,output)
                    else:summarize(config,output,not args.no_plots)
        except KeyboardInterrupt:
            print('Lightweight evaluation interrupted',flush=True)
            raise SystemExit(130) from None
        except Exception:
            import traceback
            traceback.print_exc();raise SystemExit(1) from None


if __name__=='__main__':main()
