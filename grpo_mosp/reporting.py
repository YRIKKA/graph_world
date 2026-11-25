"""Readable, deterministic console reports; authoritative task records stay JSON.

Reward sums are raw sums of actual terminal task rewards. Training summaries are
rollout statistics collected while parameters change. Held-out rates and their
uncertainty come from the existing paired/stratified estimator, never from an
unweighted console average masquerading as the scientific comparison.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time


def model_name(config):
    return config.get('model') or config['initial_checkpoint']['model']


def number(value, percent=False, integer=False):
    if value is None:
        return 'N/A'
    if isinstance(value,str):
        return value
    if percent:
        return f'{100*value:.1f}%'
    if integer:
        return f'{value:,.0f}'
    return f'{value:,.4f}' if abs(value)<1e7 else f'{value:.4g}'


def table(title, headers, rows):
    """Plain text stays legible in terminals, pipes and log.txt without ANSI."""
    rows = [[str(v) for v in row] for row in rows]
    headers = list(map(str,headers))
    widths = [max(len(headers[i]),*(len(row[i]) for row in rows)) for i in range(len(headers))] if rows else list(map(len,headers))
    rule = '+-'+'-+-'.join('-'*w for w in widths)+'-+'
    def line(row):
        return '| '+' | '.join(v.ljust(w) for v,w in zip(row,widths))+' |'
    print(f'\n{title}\n{rule}\n{line(headers)}\n{rule}',flush=True)
    for row in rows:
        print(line(row),flush=True)
    print(rule,flush=True)


class Progress:
    """A line-based progress bar works equally well live and in an append-only log.

    Count every completion. Throttle rendering for fast evaluation loops, but
    always emit the initial and final counts; failed work is never advanced.
    """
    def __init__(self,label,total,min_interval=.5):
        self.label,self.total,self.completed = label,total,0
        self.started = time.monotonic()
        self.last_render = -math.inf
        self.min_interval = min_interval
        self.render()
    def render(self):
        fraction = self.completed/self.total if self.total else 1.
        filled = int(24*fraction)
        elapsed = int(time.monotonic()-self.started)
        print(f'{self.label} | [{"#"*filled}{"-"*(24-filled)}] '
              f'{self.completed}/{self.total} ({fraction:6.1%}) | elapsed {elapsed//60:02d}:{elapsed%60:02d}',flush=True)
        self.last_render = time.monotonic()
    def advance(self):
        if self.completed>=self.total:
            raise ValueError('Progress cannot exceed planned work')
        self.completed += 1
        if self.completed==self.total or time.monotonic()-self.last_render>=self.min_interval:
            self.render()


class Totals:
    """Streaming aggregation avoids keeping evaluation histories/models in RAM."""
    def __init__(self):
        self.count,self.sums = 0,defaultdict(float)
    def add(self,metrics):
        self.count += 1
        for key,value in metrics.items():
            if isinstance(value,(int,float)):
                if not math.isfinite(value):
                    raise ValueError(f'Non-finite reported metric: {key}')
                self.sums[key] += value
    def record(self):
        return dict(tasks=self.count,total_reward=self.sums.get('reward',0.),
                    mean_reward=self.sums.get('reward',0.)/self.count if self.count else None,
                    totals=dict(self.sums),means={k:v/self.count for k,v in self.sums.items()} if self.count else {})


RATE_KEYS = {'complete_success','valid_path_fraction','duplicate_rate','protocol_failure','limit_failure'}
METRIC_LABELS = [('S','Pareto fraction S'),('complete_success','Complete success'),
    ('valid_path_fraction','Valid-path fraction'),('duplicate_rate','Duplicate rate'),
    ('protocol_failure','Protocol failures'),('limit_failure','Resource-limit failures'),
    ('normalized_hypervolume','Normalized hypervolume'),('A','Discovery A'),('L','Selection gap L')]


def task_tables(title,columns,raw_hypervolume=True,inference=True,mean_label='Mean reward'):
    """Columns are task summaries; callers determine scientific versus raw means."""
    def values(key,kind='means',**fmt):
        return [number(s.get(kind,{}).get(key),**fmt) for s in columns.values()]
    rows = [['Tasks',*[number(s['tasks'],integer=True) for s in columns.values()]],
            ['Total reward (raw)',*[number(s['total_reward']) for s in columns.values()]],
            [mean_label,*[number(s['mean_reward']) for s in columns.values()]]]
    if any('terminal_reward' in c.get('totals',{}) for c in columns.values()):
        rows += [[label,*values(key,'totals')] for key,label in
            [('terminal_reward','Terminal reward total (unshaped)'),('edge_bonus','Edge-information bonus total'),('path_bonus','Unique-path bonus total'),
             ('hypervolume_bonus','Discovery hypervolume bonus total'),
             ('cost_diversity_bonus','Cost-vector diversity bonus total'),
             ('path_disjointness_bonus','Path-disjointness bonus total')]]
    rows += [[label,*values(key,percent=key in RATE_KEYS)] for key,label in METRIC_LABELS]
    if raw_hypervolume:
        rows.append(['Hypervolume',*values('hypervolume')])
    rows += [['Generated tokens',*values('generated_tokens','totals',integer=True)],
             ['Edge traversals',*values('edge_traversals','totals',integer=True)]]
    if inference:
        rows += [['Mean inference seconds',*values('latency_seconds')],
                 ['Total inference seconds',*values('latency_seconds','totals')]]
    table(title,['Metric',*columns],rows)


def training_metrics(rollout):
    env = rollout.env
    # Reuse metrics already calculated for the terminal reward. Context-limited
    # tasks have no submission; archive membership still gives exact discovery A.
    scored = getattr(env,'_terminal_metrics',None)
    if scored is None:
        discovered = min(env.k,len(set(env.archive)&env._instance.pareto))/env.k
        scored = dict(pareto_fraction=0.,all_correct=0.,valid_slot_fraction=0.,
                      hypervolume=0.,normalized_hypervolume=0.,discovery_fraction=discovered,selection_gap=discovered)
    slots = env.submission.get('paths',[]) if isinstance(env.submission,dict) else []
    nonempty = [tuple(p) for p in slots if isinstance(p,list) and p and all(isinstance(v,str) for v in p)] if isinstance(slots,list) else []
    shaping = rollout.shaping or {}
    return dict(terminal_reward=shaping.get('terminal_reward',rollout.reward),
        edge_bonus=shaping.get('edge_bonus',0.),path_bonus=shaping.get('path_bonus',0.),
        hypervolume_bonus=shaping.get('hypervolume_bonus',0.),
        cost_diversity_bonus=shaping.get('cost_diversity_bonus',0.),
        path_disjointness_bonus=shaping.get('path_disjointness_bonus',0.),
        reward=rollout.reward,S=scored['pareto_fraction'],complete_success=scored['all_correct'],
        valid_path_fraction=scored['valid_slot_fraction'],duplicate_rate=(len(nonempty)-len(set(nonempty)))/env.k,
        protocol_failure=float(env.failure),limit_failure=float(rollout.status!='ok'),
        hypervolume=scored['hypervolume'],normalized_hypervolume=scored['normalized_hypervolume'],
        A=scored['discovery_fraction'],L=scored['selection_gap'],edge_traversals=env.moves,
        generated_tokens=sum(len(d.tokens) for d in rollout.decisions))


def training_run_record(directory,config,require_complete=True):
    """Read new summaries, or recover only metrics actually saved by older runs."""
    directory = Path(directory)
    if (directory/'training_summary.json').exists():
        return json.loads((directory/'training_summary.json').read_text())
    if require_complete and config.get('early_stopping') and not (directory/'training_summary.json').exists():
        return None
    if not (directory/'metrics.jsonl').exists() or (require_complete and not (directory/'training_state.pt').exists()):
        return None
    # A running worker may be writing its last line. Read completed records only.
    text = (directory/'metrics.jsonl').read_text()
    rows = [json.loads(line) for line in text.splitlines(keepends=True) if line.strip() and line.endswith('\n')]
    if not rows:
        return None
    count = sum(r.get('tasks',len(r.get('generated_lengths',[])) or config['group_size']) for r in rows)
    reward = sum(r.get('total_reward',r['mean_reward']*(r.get('tasks',len(r.get('generated_lengths',[])) or config['group_size']))) for r in rows)
    compute = json.loads((directory/'training_compute.json').read_text()) if (directory/'training_compute.json').exists() else {}
    totals = dict(reward=reward,**{k:compute[k] for k in ('edge_traversals','generated_tokens') if k in compute})
    totals.setdefault('generated_tokens',sum(sum(r.get('generated_lengths',[])) for r in rows))
    means = dict(reward=reward/count)
    fields = set.intersection(*(set(r.get('performance',{})) for r in rows))
    for key in fields:
        means[key] = sum(r['performance'][key]*(r.get('tasks',len(r.get('generated_lengths',[])) or config['group_size'])) for r in rows)/count
    if 'edge_traversals' in means:
        totals.setdefault('edge_traversals',means['edge_traversals']*count)
    # Older files do not contain path-quality metrics: report N/A, never invent 0.
    for source,target in [('protocol_failure_rate','protocol_failure'),('context_limit_failure_rate','limit_failure')]:
        if all(source in r for r in rows):
            means[target] = sum(r[source]*(r.get('tasks',len(r.get('generated_lengths',[])) or config['group_size'])) for r in rows)/count
    return dict(model=config['model'],training_seed=config['seed'],iterations=len(rows),tasks=count,
        total_reward=reward,mean_reward=reward/count,totals=totals,means=means,
        wall_seconds=compute.get('wall_seconds'),legacy_metrics=True)


def training_model_report(model,train_configs,output,persist=True):
    runs = [record for path,config in train_configs if (record:=training_run_record(config['output'],config)) is not None]
    if not runs:
        return
    # Total reward pools actual task sums. Means across runs weight each seed
    # equally, even for historical runs with different numbers of tasks.
    keys = set.intersection(*(set(r['means']) for r in runs))
    sums = {k:sum(r['totals'].get(k,0) for r in runs) for k in set.intersection(*(set(r['totals']) for r in runs))}
    across = dict(tasks=sum(r['tasks'] for r in runs),total_reward=sum(r['total_reward'] for r in runs),
        mean_reward=statistics.mean(r['mean_reward'] for r in runs),totals=sums,
        means={k:statistics.mean(r['means'][k] for r in runs) for k in keys})
    table(f'{model} | completed training runs ({len(runs)}/{len(train_configs)})',
          ['Training seed','Tasks','Total reward','Mean reward','Iterations','Training seconds'],
          [[r['training_seed'],r['tasks'],number(r['total_reward']),number(r['mean_reward']),r['iterations'],number(r.get('wall_seconds'))] for r in runs])
    task_tables(f'{model} | summary across {len(runs)} training runs',{'Across runs':across},inference=False,mean_label='Mean reward (equal seeds)')
    print('Training rates are equal-seed means of changing-policy rollout statistics; totals sum actual tasks.\n',flush=True)
    record = dict(model=model,completed_runs=len(runs),planned_runs=len(train_configs),runs=runs,across_runs=across)
    if persist:
        Path(output).mkdir(parents=True,exist_ok=True)
        (Path(output)/'training_summary.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def checkpoint_report(config,model_id,groups):
    name = model_name(config)
    table(f'{name} | {model_id} | evaluation totals (reward = S)',
          ['Mode','B/k','Tasks','Total reward','Mean reward','Tokens','Traversals'],
          [[mode,budget,s.count,number(s.record()['total_reward']),number(s.record()['mean_reward']),
            number(s.sums['generated_tokens'],integer=True),number(s.sums['edge_traversals'],integer=True)]
           for (mode,budget),s in groups['all'].items()])
    for mode in ('independent_exploration','shared_transcript_selection'):
        columns = {domain:s.record() for (m,b,domain),s in groups['domains'].items() if m==mode and b==8}
        if columns:
            task_tables(f'{name} | {model_id} | {mode} | primary B=8k',columns)
    print('Above: raw task totals/means. Prespecified graph-weighted comparisons and confidence intervals follow in summarize.\n',flush=True)


def primary_comparison_report(config,comparison):
    """Finish with the prespecified paired result in plain model/metric labels.

    Reuse the scientific estimator's rows, including its crossed-run intervals;
    raw pooled task averages must never replace the headline weighted scores.
    """
    scopes = [('in_distribution','In-distribution'),
              ('distribution_shift_macro','Distribution shifts (equal-domain mean)'),
              ('size','Shift: graph size'),('density','Shift: graph density'),
              ('costs','Shift: edge costs'),('objectives','Shift: number of objectives')]
    for reference in ['initial']+(['pre_grpo'] if config.get('pre_grpo_checkpoint') else []):
        for mode,mode_label in [('independent_exploration','independent exploration'),
                                ('shared_transcript_selection','shared transcript selection')]:
            selected = {r['scope']:r for r in comparison if r['comparison']=='training_procedure'
                and r['reference']==reference and r['mode']==mode
                and r['budget_multiplier']==8 and r['difficulty']=='all' and r['k']=='all' and r['metric']=='S'}
            if not selected:
                continue
            baseline = 'Frozen base' if reference=='initial' else 'Pre-GRPO checkpoint'
            rows = []
            for scope,label in scopes:
                if scope not in selected:
                    continue
                row = selected[scope]
                interval = ('CI above zero' if row['ci_low']>0 else
                            'CI below zero' if row['ci_high']<0 else 'CI includes zero')
                rows.append([label,f"{100*row['baseline']:.2f}%",f"{100*row['post']:.2f}%",
                             f"{100*row['difference']:+.2f}",
                             f"[{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}]",interval])
            table(f'{model_name(config)} | TRAINED vs {baseline.upper()} | PRIMARY B=8k | {mode_label}',
                  ['Evaluation domain',baseline+' S','GRPO-trained S','Change (pp)','Paired 95% CI (pp)','Interval assessment'],rows)
            print(f'GRPO-trained scores average {len(config["training_runs"])} independent training runs. '
                  'S is the fraction of requested paths submitted as distinct valid true-Pareto paths. '
                  'Change = trained minus reference; positive values favor GRPO. pp = percentage points.\n'
                  'Intervals are pointwise paired bootstrap intervals. An interval including zero does not '
                  'establish a reliable improvement or degradation. This headline uses held-out evaluation, '
                  'separate from changing-policy training rewards.\n'
                  + ('Both models explore independently within the same traversal budget.\n' if mode=='independent_exploration'
                     else 'Selection-only diagnostic: both models receive identical externally collected transcripts and archives.\n'),flush=True)


def comparison_report(config,rows,comparison):
    """Use existing weighted estimates/intervals; do not recompute uncertainty."""
    from .eval_baselines import DOMAINS,METRICS
    posts = [r['id'] for r in config['training_runs']]
    for reference in ['initial']+(['pre_grpo'] if config.get('pre_grpo_checkpoint') else []):
        for mode in ('independent_exploration','shared_transcript_selection'):
            for scope in ('in_distribution','distribution_shift_macro'):
                selected = [r for r in comparison if r['scope']==scope and r['difficulty']=='all' and r['k']=='all'
                    and r['budget_multiplier']==8 and r['mode']==mode and r['reference']==reference]
                if not selected:
                    continue
                lookup = {(r['comparison'],r['metric']):r for r in selected}
                domains = {DOMAINS[0]} if scope=='in_distribution' else set(DOMAINS[1:])
                columns = {}
                for label,ids,entity,prefix in [(reference,[reference],'training_procedure','baseline'),
                    *[(post,[post],post,'post') for post in posts],
                    (f'Across {len(posts)} runs',posts,'training_procedure','post')]:
                    tasks = [r for r in rows if r['model_id'] in ids and r['domain'] in domains
                             and r['budget_multiplier']==8 and r['mode']==mode]
                    means = {m:lookup[entity,m][prefix] for m in METRICS if (entity,m) in lookup}
                    total = sum(r['metrics']['S'] for r in tasks)
                    columns[label] = dict(tasks=len(tasks),total_reward=total,mean_reward=means['S'],means=means,
                        totals={k:sum(r['metrics'][k] for r in tasks) for k in ('generated_tokens','edge_traversals','latency_seconds')})
                task_tables(f'{model_name(config)} | held-out {scope} | {mode} | B=8k',columns,raw_hypervolume=False,mean_label='Mean reward (weighted)')
                table('Paired improvement in S (95% confidence intervals)',
                      ['Run/comparison','Before S','After S','Difference','95% CI'],
                      [[entity,number(lookup[entity,'S']['baseline']),number(lookup[entity,'S']['post']),
                        f"{lookup[entity,'S']['difference']:+.4f}",
                        f"[{lookup[entity,'S']['ci_low']:+.4f}, {lookup[entity,'S']['ci_high']:+.4f}]"]
                       for entity in [*posts,'training_procedure']])
    print('Reward totals sum raw task scores. Held-out rates use the prespecified graph/stratum weights; across-run rates give each training seed equal weight.\n',flush=True)
    primary_comparison_report(config,comparison)


def saved_run_report(root):
    """Read current/completed metric records; never load or modify model weights."""
    from .run_artifacts import console_log
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    with console_log(root/'log.txt'):
        print(f'Saved metric snapshot: {root}',flush=True)
        for model,paths in manifest['models'].items():
            trained = Path(paths['trained'])
            configs = [(str(p),json.loads(p.read_text())) for p in sorted((trained/'configs').glob('train_seed_*.json'))]
            completed = 0
            for _,config in configs:
                directory = Path(config['output'])
                record = training_run_record(directory,config,require_complete=False)
                finished = (directory/'training_state.pt').exists()
                completed += int(finished)
                progress = Progress(f'{model} | seed {config["seed"]} | saved iterations',config['steps'])
                progress.completed = min(record['iterations'] if record else 0,config['steps'])
                if progress.completed:
                    progress.render()
                if record:
                    task_tables(f'{model} | seed {config["seed"]} | {"complete" if finished else "recorded tasks so far"}',
                                {'Saved metrics':record},inference=False)
            progress = Progress(f'{model} | completed training runs',len(configs))
            progress.completed = completed
            if completed:
                progress.render()
            training_model_report(model,configs,trained,persist=False)
        print('Snapshot reads saved records only. Missing historical metrics are N/A; running iterations appear after their metrics record is flushed.',flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pretty-print saved/active GRPO training metrics without loading models.')
    parser.add_argument('--output',type=Path,required=True,help='Existing full_runs/run_<timestamp> directory')
    saved_run_report(parser.parse_args().output)
