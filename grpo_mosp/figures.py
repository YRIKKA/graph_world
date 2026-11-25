"""Headless pyplot exports from saved metrics; no model loading or GPU required.

Training points are rollout-group means, located at the optimizer-update count
of the policy which collected them. Evaluation points reuse the prespecified
weighted estimates and marginal confidence intervals from the evaluator.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .run_artifacts import model_slug, RUN_PREFIXES

METRICS=[('reward','Mean reward'),('correct_pareto_fraction','Correct Pareto fraction S'),
         ('complete_success','Complete success'),('valid_path_fraction','Valid path fraction'),
         ('duplicate_rate','Duplicate rate')]
EVALUATION_KEYS=dict(reward='S',correct_pareto_fraction='S',complete_success='complete_success',
                     valid_path_fraction='valid_path_fraction',duplicate_rate='duplicate_rate')
SCOPES=[('in_distribution','In-distribution'),('distribution_shift_macro','Distribution-shift macro'),
        ('size','Graph size shift'),('density','Graph density shift'),
        ('costs','Edge cost shift'),('objectives','Number of objectives shift')]


def figure_root(config, location):
    """Find the shared run directory through either training or eval subdirs."""
    path=Path(location).expanduser().resolve()
    for parent in (path,*path.parents):
        if parent.name.startswith(RUN_PREFIXES) and parent.parent.name=='full_runs':
            return parent/'figures'
    # Standalone/test callers outside the launcher still get a local figures dir.
    return path/'figures'


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def save(fig, directory, stem, plt):
    directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for extension in ('png','svg'):
        path=directory/f'{stem}.{extension}'
        fig.savefig(path,dpi=160,bbox_inches='tight')
        paths.append(str(path))
    plt.close(fig)
    return paths


def training_points(config, rows):
    """Legacy rows infer counts from epochs and zero-variance skip semantics."""
    points=[];completed=0;inferred=False
    for row in rows:
        before=row.get('optimizer_step_before',completed)
        if 'optimizer_step' in row:
            completed=row['optimizer_step']
        else:
            inferred=True
            skipped=config.get('beta',.01)==0 and bool(row.get('zero_variance_group',0))
            completed+=row.get('optimizer_steps',0 if skipped else config.get('epochs_per_group',1))
        performance=row.get('performance',{})
        score=performance.get('S')
        if score is None and config.get('objective','pareto')=='pareto':
            score=performance.get('terminal_reward')
            if score is None and not config.get('shaping') and not config.get('moo_rewards'):score=row.get('mean_reward')
        stage=row.get('graph_stage')
        curriculum=config.get('graph_curriculum')
        if stage is None and curriculum:
            stage=min(len(curriculum['stages'])-1,row['step']*len(curriculum['stages'])//config['steps'])
        points.append(dict(group=row['step']+1,optimizer_step=before,optimizer_step_after=completed,
            reward=row.get('mean_reward',performance.get('reward')),
            correct_pareto_fraction=score,complete_success=performance.get('complete_success'),
            valid_path_fraction=performance.get('valid_path_fraction'),duplicate_rate=performance.get('duplicate_rate'),
            graph_stage=stage,graph_pool_stage=row.get('graph_pool_stage'),rehearsal=row.get('rehearsal',False)))
        for name in config.get('moo_rewards', {}).get('functions', []):
            points[-1]['raw_' + name] = row.get('reward_components', {}).get(name)
            points[-1]['contribution_' + name] = row.get('reward_contributions', {}).get(name)
    return points,inferred


def training_curves(config):
    output=Path(config['output'])
    source=output/'metrics.jsonl'
    if not source.exists():return []
    # Reading an active run is safe: ignore its possibly incomplete final line.
    snapshot=source.read_bytes()
    rows=[json.loads(line) for line in snapshot.decode().splitlines(keepends=True)
          if line.strip() and line.endswith('\n')]
    if not rows:return []
    points,inferred=training_points(config,rows)
    model=config['model'];condition=config.get('moo_reward_condition',config.get('shaping_condition','grpo'))
    directory=figure_root(config,output)/model_slug(model)/'training'/model_slug(condition)/f'seed-{config["seed"]}'
    fields=['group','optimizer_step','optimizer_step_after',*[m for m,_ in METRICS],
            'graph_stage','graph_pool_stage','rehearsal']
    reward_names = config.get('moo_rewards', {}).get('functions', [])
    fields += [prefix + name for name in reward_names for prefix in ('raw_', 'contribution_')]
    write_csv(directory/'training_curve_data.csv',points,fields)
    subsets=[('trajectory',points)]
    if config.get('graph_curriculum'):
        for stage in range(len(config['graph_curriculum']['stages'])):
            subsets.append((f'curriculum_stage_{stage+1:02d}',[p for p in points if p['graph_stage']==stage]))
    plt=pyplot();files=[]
    selected_group=selected_update=None
    selection=output/'early_stopping.json'
    if selection.exists():
        selected_group=json.loads(selection.read_text()).get('selected_step')
        if selected_group==0:selected_update=0
        else:selected_update=next((p['optimizer_step_after'] for p in points if p['group']==selected_group),None)
    for name,subset in subsets:
        for metric,label in METRICS:
            fig,ax=plt.subplots(figsize=(9,4.8))
            if subset:
                values=[float(p[metric]) if p[metric] is not None else math.nan for p in subset]
                x=[p['optimizer_step'] for p in subset]
                disaggregated = metric == 'reward' and bool(reward_names)
                ax.plot(x,values,marker='o',markersize=4,
                        label='Aggregate reward' if disaggregated else 'Rollout-group mean',
                        **({'color': 'black', 'linewidth': 2.5} if disaggregated else {}))
                if disaggregated:
                    labels = dict(reward_valid='Valid path fraction', reward_valid_per_path='Unique valid count',
                                  reward_hypervolume='Hypervolume', reward_igd='IGD+')
                    missing = []
                    for term in reward_names:
                        key = 'contribution_' + term
                        if not any(p[key] is not None for p in subset):
                            missing.append(term)
                            continue
                        term_values = [float(p[key]) if p[key] is not None else math.nan for p in subset]
                        weighted = (config['moo_rewards']['weight_by_valid']
                                    and term not in ('reward_valid', 'reward_valid_per_path'))
                        label = labels.get(term, term) + (' × validity' if weighted else '')
                        ax.plot(x,term_values,marker='.',linestyle='--',label=label)
                    if missing:
                        ax.text(.01,.02,'Component terms not recorded in these saved logs',
                                transform=ax.transAxes,fontsize=8)
                replay=[p for p in subset if p['rehearsal'] and p[metric] is not None]
                if replay:
                    ax.scatter([p['optimizer_step'] for p in replay],[p[metric] for p in replay],
                               marker='s',s=50,facecolors='none',edgecolors='darkorange',label='Earlier-stage rehearsal',zorder=3)
                if not any(math.isfinite(value) for value in values):
                    ax.text(.5,.5,'Metric not recorded in these saved logs',ha='center',transform=ax.transAxes)
                if name=='trajectory' and config.get('graph_curriculum'):
                    for stage in sorted({p['graph_stage'] for p in subset})[1:]:
                        start=next(p['optimizer_step'] for p in subset if p['graph_stage']==stage)
                        ax.axvline(start,color='gray',linestyle=':',alpha=.6)
                        ax.text(start,.97,f' Stage {stage+1}',transform=ax.get_xaxis_transform(),fontsize=8,va='top')
                if name=='trajectory' and selected_update is not None:
                    ax.axvline(selected_update,color='seagreen',linestyle='--',alpha=.8,label='Exported adapter')
                ax.legend(loc='best',fontsize=8)
            else:
                ax.text(.5,.5,'Stage not reached before training ended',ha='center',transform=ax.transAxes)
            if metric!='reward':ax.set_ylim(-.03,1.03)
            ax.set(title=f'{model}\n{condition} | seed {config["seed"]} | {name.replace("_"," ")}',
                   xlabel='Optimizer updates completed before rollout collection',
                   ylabel='Mean training return' if metric=='reward' else label)
            ax.grid(alpha=.25)
            fig.text(.01,.01,'One point per rollout group; repeated x values indicate skipped optimizer updates.'+
                     (' Legacy update counts inferred from saved epochs/skip flags.' if inferred else ''),fontsize=8)
            fig.tight_layout(rect=(0,.04,1,1))
            files.extend(save(fig,directory/name,metric,plt))
    (directory/'manifest.json').write_text(json.dumps(dict(model=model,condition=condition,seed=config['seed'],
        source_metrics=str(source.resolve()),source_sha256=hashlib.sha256(snapshot).hexdigest(),
        optimizer_counts_inferred=inferred,groups=len(points),selected_group=selected_group,
        selected_optimizer_update=selected_update,training_objective=config.get('objective','pareto'),
        shaping=config.get('shaping'),moo_rewards=config.get('moo_rewards'),
        reward_component_columns={name: dict(raw='raw_'+name, contribution='contribution_'+name)
                                  for name in reward_names},files=files,
        curriculum_partition='scheduled stage; rehearsal points marked separately'),indent=2)+'\n')
    print(f'Exported {len(files)//2} training figures: {directory}',flush=True)
    return files


def evaluation_points(config, records):
    """Normalize full/light evaluator rows without pooling tasks or conditions."""
    training=config['training_runs']
    ids=['initial']+(['pre_grpo'] if config.get('pre_grpo_checkpoint') else [])+[r['id'] for r in training]
    labels={'initial':'Frozen base','pre_grpo':'Pre-GRPO'}
    labels.update({r['id']:f'{r["condition"]}\nseed {r["seed"]}' if 'condition' in r else r['id'] for r in training})
    result=[]
    def optional_number(row,key):
        value=row.get(key)
        return float(value) if value not in (None,'') else None
    for row in records:
        # The full evaluator includes k/difficulty strata and budget sweeps.
        # These comparison bars use exactly the prespecified primary estimate.
        if str(row.get('budget_multiplier',8))!='8' or str(row.get('k','all'))!='all' or row.get('difficulty','all')!='all':
            continue
        entity=row.get('condition',row.get('comparison'))
        if entity not in ids:continue
        light='condition' in row
        before='before' if light else 'baseline'
        after='after' if light else 'post'
        reference=row['reference']
        if entity in ids[1:] and reference=='initial':
            result.append(dict(model_id=entity,label=labels[entity],mode=row['mode'],scope=row['scope'],
                metric=row['metric'],value=float(row[after]),
                low=optional_number(row,after+'_ci_low'),high=optional_number(row,after+'_ci_high')))
        if entity==training[0]['id'] and reference in ('initial','pre_grpo'):
            result.append(dict(model_id=reference,label=labels[reference],mode=row['mode'],scope=row['scope'],
                metric=row['metric'],value=float(row[before]),
                low=optional_number(row,before+'_ci_low'),high=optional_number(row,before+'_ci_high')))
    # Order every chart identically, starting with the untrained base. Frozen
    # marginal intervals come from the first checkpoint's paired bootstrap.
    return sorted(result,key=lambda point:(point['mode'],point['scope'],point['metric'],ids.index(point['model_id'])))


def evaluation_bars(config, records, output):
    points=evaluation_points(config,records)
    if not points:return []
    model=config.get('model') or config['initial_checkpoint']['model']
    directory=figure_root(config,output)/model_slug(model)/'evaluation'
    write_csv(directory/'comparison_figure_data.csv',points,
              ['model_id','label','mode','scope','metric','value','low','high'])
    plt=pyplot();files=[]
    for mode in sorted({point['mode'] for point in points}):
        scopes=[(scope,label) for scope,label in SCOPES if any(p['mode']==mode and p['scope']==scope for p in points)]
        model_count=len({p['model_id'] for p in points if p['mode']==mode})
        for metric,label in METRICS:
            key=EVALUATION_KEYS[metric]
            columns=2 if len(scopes)>1 else 1
            nrows=math.ceil(len(scopes)/columns)
            fig,axes=plt.subplots(nrows,columns,figsize=(max(8,.65*model_count)*columns,4.4*nrows),squeeze=False)
            for ax,(scope,scope_label) in zip(axes.flat,scopes):
                selected=[p for p in points if p['mode']==mode and p['scope']==scope and p['metric']==key]
                if not selected:
                    ax.text(.5,.5,'Metric not recorded',ha='center',transform=ax.transAxes)
                x=list(range(len(selected)))
                ax.bar(x,[p['value'] for p in selected],color=['#777777' if p['model_id'] in ('initial','pre_grpo') else '#2878b5' for p in selected])
                # Draw exact interval endpoints even if a bootstrap interval
                # does not contain the point estimate. Do not clip it to fit.
                for index,point in enumerate(selected):
                    if point['low'] is not None and point['high'] is not None:
                        ax.vlines(index,point['low'],point['high'],color='black',linewidth=1)
                        ax.hlines([point['low'],point['high']],index-.09,index+.09,color='black',linewidth=1)
                    ax.annotate(f'{100*point["value"]:.1f}%',(index,point['value']),
                                xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
                ax.set_xticks(x,[p['label'] for p in selected],rotation=30,ha='right')
                ax.set(title=scope_label,ylabel='Mean evaluation reward S' if metric=='reward' else label,ylim=(0,1.08))
                ax.grid(axis='y',alpha=.2)
            for ax in list(axes.flat)[len(scopes):]:ax.set_visible(False)
            fig.suptitle(f'{model}\n{label} | {mode.replace("_"," ")} | primary B=8k',fontsize=12)
            fig.text(.01,.01,'Each bar is one evaluated checkpoint. Prespecified graph/stratum weights; marginal 95% intervals when recorded.\n'
                     'Evaluation reward equals S. Smaller duplicate rates are better.',fontsize=9)
            fig.tight_layout(rect=(0,.05,1,.95))
            files.extend(save(fig,directory/mode/'checkpoint_comparison',metric,plt))
    (directory/'manifest.json').write_text(json.dumps(dict(model=model,
        checkpoint_ids=list(dict.fromkeys(p['model_id'] for p in points)),
        source_output=str(Path(output).resolve()),evaluation_reward='S',
        weighting='reuse existing weighted estimates, never pool raw tasks',files=files),indent=2)+'\n')
    print(f'Exported {len(files)//2} evaluation comparison figures: {directory}',flush=True)
    return files


def export_run(run):
    """Backfill figures from completed or flushed saved records, without inference."""
    run=Path(run).expanduser().resolve()
    files=[]
    configs=[*run.glob('*_trained/runs/*/config.json'),
             *run.glob('*_trained/shaping/*/runs/*/config.json'),
             *run.glob('*_trained/moo_rewards/*/runs/*/config.json')]
    for path in sorted(configs):
        files.extend(training_curves(json.loads(path.read_text())))
    outputs=[*run.glob('comparisons/*/comparison.csv'),
             *run.glob('comparisons/*/shaping/summary.json'),
             *run.glob('comparisons/*/moo_rewards/summary.json')]
    for path in sorted(outputs):
        output=path.parent
        config=json.loads((output/'suite_manifest.json').read_text())['config']
        if path.suffix=='.csv':
            with path.open() as stream:records=list(csv.DictReader(stream))
        else:records=json.loads(path.read_text())['comparisons']
        files.extend(evaluation_bars(config,records,output))
    return files


def main():
    from .run_artifacts import console_log,validate_run_directory
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='Existing full_runs/run_<timestamp> directory')
    args=parser.parse_args()
    root=validate_run_directory(args.run)
    with console_log(root/'figures/export.log'):
        files=export_run(root)
        print(f'Exported {len(files)} PNG/SVG files under {root/"figures"}',flush=True)


if __name__=='__main__':main()
