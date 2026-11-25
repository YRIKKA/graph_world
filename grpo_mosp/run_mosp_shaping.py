"""Run paired, independently configurable shaping ablations with the existing GRPO trainer.

Defaults are a development experiment: one seed per condition, 20 test graphs,
one evaluation k, one budget and one repetition. No weights load for --dry-run.
"""
import argparse
from copy import copy, deepcopy
import json
import os
from pathlib import Path
import sys
import time

from . import run_mosp as base
from .model_options import boolean
from .model_defaults import initial_checkpoint
from .pipeline_timing import PipelineTiming
from .run_artifacts import artifact_environment, model_slug
from .reporting import Progress, training_model_report
from .shaping import CONDITIONS, SIGNALS, for_condition, experiment_conditions, condition_name, signal_names
from . import graph_curriculum as graphs

LIGHT_DEFAULTS=dict(train_graphs=16,validation_graphs=8,eval_graphs=20,training_runs=1,
    group_size=4,eval_ks=[3],eval_repetitions=1,eval_relabelings=1,
    eval_budget_multipliers=[8],bootstrap_samples=1000)
SHAPING_DEFAULTS=dict(shaping='ablation',edge_weight=.01,path_weight=.05,edge_cap=0,path_cap=0,
                      selection_diagnostic=True,graph_curriculum=None,shaping_signals=None,shaping_combinations=None,
                      shaping_hypervolume=None,shaping_cost_diversity=None,shaping_path_disjointness=None,
                      shaping_edge=None,shaping_path=None,
                      hypervolume_weight=.05,cost_diversity_weight=.05,path_disjointness_weight=.05)
MODULE=base.MODULE


def parser():
    # Reuse graph ranges, traversal controls, NF4/LoRA settings, thinking flags,
    # and early-stopping arguments without maintaining a second parser copy.
    p=base.parser(curriculum_flag='--curriculum-k')
    p.allow_abbrev=False
    for action in p._actions:
        if action.dest in LIGHT_DEFAULTS and action.help:
            action.help=action.help.replace(f'[default: {base.DEFAULTS[action.dest]}]',
                                            f'[default: {LIGHT_DEFAULTS[action.dest]}]')
        if action.dest=='eval_ks':action.help='Single evaluation k [default: 3]'
        if action.dest=='eval_budget_multipliers':action.help='Light evaluation requires the single value 8'
        if action.dest=='eval_relabelings':action.help='Light evaluation requires 1 (no relabeled copies)'
        if action.dest=='objective':
            action.choices=['pareto']
            action.help='Terminal Pareto S; shaping bonuses are added only during training'
    p.prog='run_mosp_shaping.sh'
    p.description=__doc__
    p.epilog='Light defaults: 16 train / 8 validation / 20 held-out graphs, one training seed, group size 4, evaluation k=3, B=8k, one repetition. All other training defaults match run_mosp.sh.'
    p.add_argument('--shaping',default=argparse.SUPPRESS,
        help='ablation: legacy four conditions; factorial: all subsets of --shaping-signals (default all five); or one + joined signal combination [default: ablation]')
    p.add_argument('--shaping-combinations',nargs='+',default=argparse.SUPPRESS,
        help='Explicit shared-experiment condition list; all enables every shaping signal')
    p.add_argument('--shaping-signals',nargs='+',choices=SIGNALS,default=argparse.SUPPRESS,
        help='Signals for a factorial sweep, or a single custom combination')
    for signal in SIGNALS:
        p.add_argument('--shaping-'+signal.replace('_','-'),type=boolean,nargs='?',const=True,
            default=argparse.SUPPRESS,help='Enable/disable this signal for a single combination')
    for name in ('edge-weight','path-weight','hypervolume-weight','cost-diversity-weight','path-disjointness-weight'):
        p.add_argument('--'+name,type=float,default=argparse.SUPPRESS,
            help=f'Maximum total {name.split("-")[0]} bonus per complete task [default: {SHAPING_DEFAULTS[name.replace("-","_")]}]')
    for name in ('edge-cap','path-cap'):
        p.add_argument('--'+name,type=int,default=argparse.SUPPRESS,
            help='Novel discoveries needed to saturate the bonus; 0 uses B for edges or k for paths')
    p.add_argument('--selection-diagnostic',type=boolean,nargs='?',const=True,default=argparse.SUPPRESS,
                   help='Also evaluate shared-transcript selection [default: true]')
    p.add_argument('--curriculum-graph',dest='graph_curriculum_file',nargs='?',
        const=str(base.HERE/'graph_curriculum.json'),default=argparse.SUPPRESS,metavar='JSON',
        help='Graph curriculum JSON; omit flag for fixed graphs, use bare flag for graph_curriculum.json. Overrides graph shape/objective flags.')
    return p


def resolve(args):
    explicit=vars(args)
    output=base.resolve_path(args.output) if args.output is not None else None
    model=explicit.get('model')
    if model is None and output is not None:
        configs=list(output.glob('*_trained/shaping/launch_config.json'))
        if len(configs)>1:
            raise ValueError('Multiple shaping models share this run; select --model')
        if configs:model=json.loads(configs[0].read_text())['model']
    model=model or base.DEFAULTS['model']
    saved_path=output/f'{model_slug(model)}_trained/shaping/launch_config.json' if output else None
    saved=json.loads(saved_path.read_text()) if saved_path is not None and saved_path.exists() else None
    if saved is not None:
        saved=dict(SHAPING_DEFAULTS,**saved)
        from .graph_sampling import SAMPLING_DEFAULTS
        saved={**SAMPLING_DEFAULTS,**saved}
    if args.mode=='eval' and saved is None:
        raise ValueError('--mode eval requires a saved shaping experiment')
    if saved is None and model!=base.DEFAULTS['model'] and 'revision' not in explicit:
        raise ValueError('Changing --model requires its immutable --revision')
    # The base resolver performs the existing range/seed/revision validation.
    # Its prepare mode is used only for resolution; stage dispatch remains here.
    desired=dict(saved if saved else dict(base.DEFAULTS,**LIGHT_DEFAULTS,**SHAPING_DEFAULTS))
    desired['model']=model
    desired.update({k:v for k,v in explicit.items() if k in base.DEFAULTS or k in SHAPING_DEFAULTS})
    toggles={name:desired['shaping_'+name] for name in SIGNALS if desired['shaping_'+name] is not None}
    if (toggles or 'shaping_signals' in explicit) and 'shaping' not in explicit and saved is None:
        desired['shaping']='none'
    if toggles and desired['shaping'] in ('ablation','factorial'):
        raise ValueError('Individual shaping toggles select one combination; use --shaping-signals for factorial subsets')
    if desired['shaping_signals'] is not None and len(set(desired['shaping_signals']))!=len(desired['shaping_signals']):
        raise ValueError('--shaping-signals must be distinct')
    if desired['shaping'] not in ('ablation','factorial'):
        selected=set(desired['shaping_signals']) if desired['shaping_signals'] is not None else signal_names(desired['shaping'])
        for name,value in toggles.items():
            if value:selected.add(name)
            else:selected.discard(name)
        desired['shaping']=condition_name(selected)
    if desired['shaping_combinations'] is not None:
        if 'shaping' in explicit or toggles or desired['shaping_signals'] is not None:
            raise ValueError('--shaping-combinations cannot be mixed with other shaping selectors')
        combinations=desired['shaping_combinations']
        if not isinstance(combinations,list) or not combinations:
            raise ValueError('Shaping combinations must be a nonempty list')
        canonical=['all' if name=='all' else condition_name(signal_names(name)) for name in combinations]
        enabled=[frozenset(signal_names(name)) for name in canonical]
        if len(set(enabled))!=len(enabled):raise ValueError('Duplicate shaping combinations')
        desired['shaping_combinations']=canonical
    experiment_conditions(desired['shaping'],desired['shaping_signals'])
    if 'graph_curriculum_file' in explicit:
        filename=Path(explicit['graph_curriculum_file']).expanduser()
        if not filename.is_absolute():
            resolved=base.resolve_path(filename)
            if not resolved.exists() and filename.parent==Path('.'):
                resolved=base.HERE/filename
            filename=resolved
        desired['graph_curriculum']=graphs.read_config(filename)
    if desired['graph_curriculum'] is not None:
        # Ignore structural CLI flags before validating them. Operational limits,
        # graph counts/seeds and difficulty strata remain configurable.
        for key in graphs.OVERRIDDEN_GRAPH_KEYS:desired[key]=base.DEFAULTS[key]
        desired.update(desired['graph_curriculum']['stages'][-1])
    # Seed defaults must be regenerated when the caller changes the run count.
    if saved is None and 'training_seeds' not in explicit:desired['training_seeds']=None
    if 'training_seeds' in explicit and 'training_runs' not in explicit:
        desired['training_runs']=len(explicit['training_seeds'])
    if 'attempts_per_graph' in explicit:desired['attempt_multiplier']=None
    elif 'attempt_multiplier' in explicit:desired['attempts_per_graph']=None
    temp=copy(args);temp.mode='prepare'
    forwarded={k:v for k,v in desired.items() if k in base.DEFAULTS}
    for key in ('attempts_per_graph','attempt_multiplier'):
        if forwarded.get(key) is None:forwarded.pop(key,None)
    output,settings,_=base.resolve(temp,forwarded)
    settings.update({k:desired[k] for k in SHAPING_DEFAULTS})
    if settings['graph_curriculum'] is not None:
        stages=len(settings['graph_curriculum']['stages'])
        if settings['train_graphs']<stages:
            raise ValueError('--train-graphs must provide at least one graph per curriculum stage')
        if settings['steps']<max(stages,len(settings['curriculum'])):
            raise ValueError('--steps must give every graph and k curriculum stage at least one group')
    for_condition('both',**{name+'_weight':settings[name+'_weight'] for name in SIGNALS},
                  edge_cap=settings['edge_cap'],path_cap=settings['path_cap'])
    if settings['objective']!='pareto':
        raise ValueError('The shaping ablation uses terminal Pareto S; --objective must be pareto')
    if len(settings['eval_ks'])!=1 or settings['eval_budget_multipliers']!=[8] or settings['eval_relabelings']!=1:
        raise ValueError('Light evaluation uses one --eval-ks value, --eval-budget-multipliers 8, and one relabeling')
    if saved is not None and settings!=saved:
        raise ValueError('Shaping experiment settings are immutable; choose a new run directory')
    return output,settings,saved is not None


def configurations(output,c):
    graph,templates,evaluation=base.configurations(output,c)
    pools=graphs.stage_datasets(output,c) if c['graph_curriculum'] is not None else []
    trained=output/f'{model_slug(c["model"])}_trained'/'shaping'
    conditions=c.get('shaping_combinations') or experiment_conditions(c['shaping'],c['shaping_signals'])
    training,runs=[],[]
    result_dirs={'initial':str(output/f'{model_slug(c["model"])}_not_trained/shaping/evaluation/initial')}
    for condition in conditions:
        for _,template in templates:
            config=deepcopy(template)
            seed=config['seed']
            directory=trained/condition/'runs'/f'seed-{seed}'
            path=trained/'configs'/f'{condition}_seed_{seed}.json'
            config.update(output=str(directory),shaping_condition=condition,shaping=for_condition(condition,
                **{name+'_weight':c[name+'_weight'] for name in SIGNALS},
                edge_cap=c['edge_cap'],path_cap=c['path_cap']))
            if pools:
                config.update(graph_curriculum=deepcopy(c['graph_curriculum']),
                              graph_curriculum_datasets=deepcopy(pools))
            training.append((str(path),config))
            identifier=f'{condition}-seed-{seed}'
            runs.append(dict(id=identifier,condition=condition,seed=seed,
                post_checkpoint=dict(initial_checkpoint(c),adapter=str(directory/'adapter')),
                training_config=str(directory/'config.json'),input_config=str(path),
                training_compute=str(directory/'training_compute.json')))
            result_dirs[identifier]=str(trained/condition/'evaluation'/f'seed-{seed}')
    evaluation.update(model=c['model'],phase='pilot',training_runs=runs,training_run_count=len(runs),
        conditions=conditions,condition_seeds=c['training_seeds'],modes=['independent_exploration']+
        (['shared_transcript_selection'] if c['selection_diagnostic'] else []),
        output=str(output/'comparisons'/model_slug(c['model'])/'shaping'),shared_suite=str(output/'shaping_suite'),
        model_output_dirs=result_dirs)
    if pools:
        # Exclude every training stage and relabeled copies from the held-out
        # suite. Its in-distribution reference is the final curriculum stage.
        evaluation['excluded_datasets'] += [pool['dataset'] for pool in pools]
        evaluation['graph_curriculum']=deepcopy(c['graph_curriculum'])
    return graph,training,evaluation


def launch(args):
    timing=PipelineTiming(os.environ.get('MOSP_PIPELINE_STARTED_AT'))
    with timing.component('Resolve shaping experiment',announce=False):
        output,c,existing=resolve(args)
        graph,training,evaluation=configurations(output,c)
    if args.dry_run:
        print(json.dumps(dict(output=str(output),settings=c,graph_generation=graph,
            training_configs=dict(training),evaluation_config=evaluation,
            note='Shaping changes training rollout returns only. All checkpoints share one held-out suite and unshaped S.'),indent=2))
        return
    output.mkdir(parents=True,exist_ok=True)
    trained=output/f'{model_slug(c["model"])}_trained'/'shaping'
    status='failed'
    with artifact_environment(output,trained/'logs') as environment:
        try:
            with timing.component('Dependencies'):
                base.dependencies(args.mode,args.no_plots,c)
            with timing.component('Shared training/validation graphs and weighted images'):
                if c['graph_curriculum'] is None:
                    base.make_data(output,c,graph)
                else:
                    print('Graph curriculum enabled: graph shape/objective CLI flags are overridden by its JSON stages.',flush=True)
                    graphs.make_data(output,c)
            with timing.component('Shaping configurations'):
                (trained/'configs').mkdir(parents=True,exist_ok=True)
                saved_path=trained/'launch_config.json'
                if not existing:saved_path.write_text(json.dumps(c,indent=2)+'\n')
                if c['graph_curriculum'] is not None:
                    snapshot=trained/'configs'/'graph_curriculum.json'
                    parameters=c['graph_curriculum']['parameters']
                    if snapshot.exists() and json.loads(snapshot.read_text())!=parameters:
                        raise ValueError('Saved graph curriculum snapshot changed')
                    snapshot.write_text(json.dumps(parameters,indent=2)+'\n')
                for path,config in training:
                    p=Path(path)
                    if p.exists() and json.loads(p.read_text())!=config:
                        raise ValueError(f'Changed immutable training config: {path}')
                    p.write_text(json.dumps(config,indent=2)+'\n')
                eval_path=trained/'configs'/'eval_shaping.json'
                if eval_path.exists() and json.loads(eval_path.read_text())!=evaluation:
                    raise ValueError('Changed immutable shaping evaluation config')
                eval_path.write_text(json.dumps(evaluation,indent=2)+'\n')
                (trained/'experiment.json').write_text(json.dumps(dict(model=c['model'],conditions=evaluation['conditions'],
                    paired_training_seeds=c['training_seeds'],evaluation_config=str(eval_path),
                    training_configs=[p for p,_ in training],cache_and_temp_paths=environment),indent=2)+'\n')
            if args.mode in ('all','train'):
                progress=Progress(f'{c["model"]} | shaping condition/seed training runs',len(training))
                for path,config in training:
                    label=Path(path).stem
                    with timing.component(f'GRPO | {label}') as measurement:
                        directory=Path(config['output'])
                        if directory.exists():
                            if not (directory/'training_summary.json').exists() or not (directory/'training_state.pt').exists() or not (directory/'adapter').is_dir() or json.loads((directory/'config.json').read_text())!=config:
                                raise ValueError(f'Incomplete or changed training run: {directory}')
                            measurement['status']='reused'
                            print(f'Reusing completed training run: {directory}',flush=True)
                        else:
                            base.execute([sys.executable,'-m',MODULE+'.train','--config',path],trained/'logs'/f'{label}.log')
                    progress.advance()
                    if not args.no_plots:
                        from .figures import training_curves
                        with timing.component(f'Training metric figures | {label}'):
                            training_curves(config)
                for condition in evaluation['conditions']:
                    subset=[(p,cfg) for p,cfg in training if Path(cfg['output']).parents[1].name==condition]
                    training_model_report(f'{c["model"]} | {condition}',subset,trained/condition)
            if args.mode in ('all','eval'):
                if args.mode=='eval' and not args.no_plots:
                    from .figures import training_curves
                    with timing.component('Saved training metric figures'):
                        for _,config in training:training_curves(config)
                for stage in ('prepare','run','summarize'):
                    with timing.component(f'Lightweight shaping evaluation | {stage}'):
                        command=[sys.executable,'-m',MODULE+'.eval_shaping',stage,'--config',str(eval_path)]
                        if stage=='summarize' and args.no_plots:command.append('--no-plots')
                        base.execute(command,Path(evaluation['output'])/'logs'/f'{stage}.log')
            print(f'Finished shaping {args.mode}: {output}',flush=True)
            status='completed'
        except KeyboardInterrupt:
            status='interrupted'
            print('Shaping pipeline interrupted',file=sys.stderr,flush=True)
            raise SystemExit(130) from None
        except Exception:
            import traceback
            traceback.print_exc()
            raise SystemExit(1) from None
        finally:
            timing.finish(output,c['model'],args.mode,status)


def main():
    p=parser();args=p.parse_args()
    try:launch(args)
    except KeyboardInterrupt:p.exit(130,'Shaping pipeline interrupted\n')
    except (ValueError,FileNotFoundError) as error:p.exit(2,f'run_mosp_shaping.sh: {error}\n')


if __name__=='__main__':main()
