"""Paired GRPO ablation of eight additive multi-objective terminal rewards.

Compare unique-valid count and valid fraction alone, then each with hypervolume,
IGD+, or both. Every augmented setting uses validity weighting. All conditions share data, seeds, a frozen reference and held-out S.
"""
import argparse
from copy import deepcopy
import json
import os
import re
import shlex
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = 'jm_test.graph_world.grpo_mosp.experiment_moo_rewards'

from .. import figures
from .. import run_mosp_shaping as runner
from ..experiment_shaping import experiment_shaping as shaping_experiment
from ..pipeline_timing import PipelineTiming
from ..reporting import Progress, training_model_report
from ..run_artifacts import artifact_environment, model_slug, new_run_directory
from .moo_rewards import COMBINATIONS, options

base, graphs, MODULE = runner.base, runner.graphs, runner.MODULE
# Runtime-limited screening profile, anchored to the previous L40S run:
# 32 groups x 4 rollouts took roughly 4.2-4.4 hours per condition.
PROFILE = dict(shaping_experiment.PROFILE, steps=40, group_size=4,
    train_graphs=16, validation_graphs=4, training_runs=1,
    eval_graphs=20, eval_repetitions=1, bootstrap_samples=500,
    curriculum=[2], eval_ks=[2], attempts_per_graph=16, graph_curriculum=None,
    max_task_tokens=4096, early_stopping=False, adaptive_graph_sampling=False)
# Curricula are opt-in for new runs; saved runs retain their immutable settings.
PROFILE.pop('graph_curriculum_file', None)


def parser():
    # Reuse the exact CLI, including argument types, defaults and aliases.
    command = shaping_experiment.parser()
    for action in command._actions:
        if action.dest in PROFILE and action.help:
            action.help = re.sub(r'\s*\[(?:experiment )?default:.*\]\s*$', '', action.help)
            action.help += f' [experiment default: {PROFILE[action.dest]}]'
    command.prog = 'experiment_moo_rewards.sh'
    command.description = __doc__
    command.epilog = ('Fixed conditions: ' + ', '.join(COMBINATIONS) +
                      '. Defaults: fixed graph settings and k=2; opt in with --curriculum-graph and --curriculum-k 1,2,3. Runtime-limited defaults: 40 groups, group size 4, one seed, 16 train / 4 validation / 20 held-out graphs; use --dry-run to inspect.')
    return command


def resolve(args):
    explicit = vars(args)
    if any(key == 'shaping' or key.startswith('shaping_') for key in explicit):
        raise ValueError('experiment_moo_rewards fixes all eight reward combinations')
    output = base.resolve_path(args.output) if args.output is not None else None
    saved_files = list(output.glob('*_trained/moo_rewards/launch_config.json')) if output else []
    if 'model' in explicit:
        saved_files = [path for path in saved_files if path.parent.parent.name == model_slug(args.model) + '_trained']
    if len(saved_files) > 1:
        raise ValueError('Multiple models share this run; select --model')
    saved = json.loads(saved_files[0].read_text()) if saved_files else None
    if args.mode == 'eval' and saved is None:
        raise ValueError('--mode eval requires a saved MOO reward experiment')
    defaults = dict(saved if saved else PROFILE)
    if not saved:
        if 'attempt_multiplier' in explicit: defaults.pop('attempts_per_graph', None)
        if 'training_seeds' in explicit and 'training_runs' not in explicit: defaults.pop('training_runs', None)
    temp = argparse.Namespace(**{key: value for key, value in explicit.items()
                                 if key not in ('hypervolume_backend',)})
    for key, value in defaults.items():
        if key not in explicit and key not in ('hypervolume_backend',):
            setattr(temp, key, value)
    if output is None:
        temp.output = new_run_directory(base.resolve_path(args.runs_root), prefix='moo_rewards_run_')
    temp.mode = 'prepare'
    # Reuse graph/model validation while keeping the experiment's own saved state.
    temp.shaping = 'none'
    root, settings, _ = runner.resolve(temp)
    irrelevant = set(runner.SHAPING_DEFAULTS) - {'selection_diagnostic', 'graph_curriculum'}
    settings = {key: value for key, value in settings.items() if key not in irrelevant}
    settings['hypervolume_backend'] = explicit.get('hypervolume_backend', defaults.get('hypervolume_backend', 'pymoo'))
    if saved is not None and settings != saved:
        raise ValueError('MOO reward experiment settings are immutable; choose a new run directory')
    return root, settings, saved is not None


def configurations(output, c):
    graph, templates, evaluation = base.configurations(output, c)
    pools = graphs.stage_datasets(output, c) if c['graph_curriculum'] is not None else []
    trained = output / f'{model_slug(c["model"])}_trained' / 'moo_rewards'
    training, runs = [], []
    result_dirs = {'initial': str(output / f'{model_slug(c["model"])}_not_trained/moo_rewards/evaluation/initial')}
    for condition in COMBINATIONS:
        for _, template in templates:
            config = deepcopy(template)
            seed = config['seed']
            directory = trained / condition / 'runs' / f'seed-{seed}'
            path = trained / 'configs' / f'{condition}_seed_{seed}.json'
            config.update(output=str(directory), moo_reward_condition=condition,
                          moo_rewards=options(condition, c['hypervolume_backend']))
            if pools:
                config.update(graph_curriculum=deepcopy(c['graph_curriculum']),
                              graph_curriculum_datasets=deepcopy(pools))
            training.append((str(path), config))
            identifier = f'{condition}-seed-{seed}'
            runs.append(dict(id=identifier, condition=condition, seed=seed,
                post_checkpoint=dict(runner.initial_checkpoint(c), adapter=str(directory / 'adapter')),
                training_config=str(directory / 'config.json'), input_config=str(path),
                training_compute=str(directory / 'training_compute.json')))
            result_dirs[identifier] = str(trained / condition / 'evaluation' / f'seed-{seed}')
    evaluation.update(model=c['model'], phase='pilot', training_runs=runs, training_run_count=len(runs),
        conditions=list(COMBINATIONS), condition_seeds=c['training_seeds'],
        reward_experiment='moo_rewards',
        hypervolume_backend=c['hypervolume_backend'],
        modes=['independent_exploration'] + (['shared_transcript_selection'] if c['selection_diagnostic'] else []),
        output=str(output / 'comparisons' / model_slug(c['model']) / 'moo_rewards'),
        shared_suite=str(output / 'moo_rewards_suite'), model_output_dirs=result_dirs)
    if pools:
        evaluation['excluded_datasets'] += [pool['dataset'] for pool in pools]
        evaluation['graph_curriculum'] = deepcopy(c['graph_curriculum'])
    return graph, training, evaluation



def launch_summary(args, settings, output, training, evaluation, existing, argv=None):
    """Describe explicit CLI inputs and the effective, possibly saved settings."""
    return dict(
        command=shlex.join([sys.executable, '-m', MODULE + '.experiment_moo_rewards.experiment_moo_rewards', *argv])
                if argv is not None else None,
        cli_flags=vars(args),
        output=str(output), mode=args.mode, reused_settings=existing,
        python_executable=sys.executable,
        initial_checkpoint=runner.initial_checkpoint(settings),
        graph_curriculum_enabled=settings['graph_curriculum'] is not None,
        k_curriculum_enabled=len(settings['curriculum']) > 1,
        training_budgets={str(k): settings['attempts_per_graph']
                          if settings['attempts_per_graph'] is not None else settings['attempt_multiplier'] * k
                          for k in settings['curriculum']},
        reward_conditions={config['moo_reward_condition']: config['moo_rewards'] for _, config in training},
        planned_training_runs=len(training),
        planned_training_rollouts=len(training) * settings['steps'] * settings['group_size'],
        planned_evaluation_tasks=(len(training) + 1) * settings['eval_graphs']
                                 * settings['eval_repetitions'] * len(evaluation['modes']),
        evaluation_modes=evaluation['modes'], frozen_base_comparison=True,
        plots_enabled=not args.no_plots,
        resolved_settings=settings,
    )


def launch(args, argv=None):
    timing=PipelineTiming(os.environ.get('MOSP_PIPELINE_STARTED_AT'))
    with timing.component('Resolve MOO reward experiment',announce=False):
        output,c,existing=resolve(args)
        graph,training,evaluation=configurations(output,c)
    summary = launch_summary(args, c, output, training, evaluation, existing, argv)
    if args.dry_run:
        print(json.dumps(dict(output=str(output),settings=c,graph_generation=graph,launch_summary=summary,
            training_configs=dict(training),evaluation_config=evaluation,
            note='Eight additive rewards: two validity baselines and all nonempty hypervolume/IGD+ subsets; normalize=true, weight_by_valid=true.'),indent=2,default=str))
        return
    output.mkdir(parents=True,exist_ok=True)
    trained=output/f'{model_slug(c["model"])}_trained'/'moo_rewards'
    status='failed'
    with artifact_environment(output,trained/'logs') as environment:
        try:
            print('\nMOO reward experiment | launch parameters', flush=True)
            if summary['command'] is not None:
                print(f"Command: {summary['command']}", flush=True)
            print(json.dumps(summary, indent=2, default=str), flush=True)
            with timing.component('Dependencies'):
                if args.mode in ('all', 'train'):
                    from pymoo.indicators.igd_plus import IGDPlus
                    from pymoo.indicators.hv import HV
                    if c['hypervolume_backend'] == 'moocore':
                        import moocore
                base.dependencies(args.mode,args.no_plots,c)
            with timing.component('Shared training/validation graphs and weighted images'):
                if c['graph_curriculum'] is None:
                    base.make_data(output,c,graph)
                else:
                    print('Graph curriculum enabled: graph shape/objective CLI flags are overridden by its JSON stages.',flush=True)
                    graphs.make_data(output,c)
            with timing.component('MOO reward configurations'):
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
                eval_path=trained/'configs'/'eval_moo_rewards.json'
                if eval_path.exists() and json.loads(eval_path.read_text())!=evaluation:
                    raise ValueError('Changed immutable MOO reward evaluation config')
                eval_path.write_text(json.dumps(evaluation,indent=2)+'\n')
                (trained/'experiment.json').write_text(json.dumps(dict(model=c['model'],conditions=evaluation['conditions'],
                    paired_training_seeds=c['training_seeds'],evaluation_config=str(eval_path),
                    training_configs=[p for p,_ in training],cache_and_temp_paths=environment),indent=2)+'\n')
            if args.mode in ('all','train'):
                progress=Progress(f'{c["model"]} | MOO reward condition/seed training runs',len(training))
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
                        with timing.component(f'Training metric figures | {label}'):
                            figures.training_curves(config)
                for condition in evaluation['conditions']:
                    subset=[(p,cfg) for p,cfg in training if Path(cfg['output']).parents[1].name==condition]
                    training_model_report(f'{c["model"]} | {condition}',subset,trained/condition)
            if args.mode in ('all','eval'):
                if args.mode=='eval' and not args.no_plots:
                    with timing.component('Saved training metric figures'):
                        for _,config in training:figures.training_curves(config)
                for stage in ('prepare','run','summarize'):
                    with timing.component(f'Lightweight MOO reward evaluation | {stage}'):
                        command=[sys.executable,'-m',MODULE+'.eval_shaping',stage,'--config',str(eval_path)]
                        if stage=='summarize' and args.no_plots:command.append('--no-plots')
                        base.execute(command,Path(evaluation['output'])/'logs'/f'{stage}.log')
            print(f'Finished MOO reward {args.mode}: {output}',flush=True)
            status='completed'
        except KeyboardInterrupt:
            status='interrupted'
            print('MOO reward pipeline interrupted',file=sys.stderr,flush=True)
            raise SystemExit(130) from None
        except Exception:
            import traceback
            traceback.print_exc()
            raise SystemExit(1) from None
        finally:
            timing.finish(output,c['model'],args.mode,status)



def main():
    command = parser()
    args = command.parse_args()
    try:
        launch(args, argv=sys.argv[1:])
    except KeyboardInterrupt:
        command.exit(130, 'MOO reward experiment interrupted\n')
    except (ValueError, FileNotFoundError, ImportError, AttributeError) as error:
        command.exit(2, f'experiment_moo_rewards.sh: {error}\n')


if __name__ == '__main__':
    main()
