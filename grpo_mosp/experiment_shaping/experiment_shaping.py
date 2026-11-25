"""Seven shaping agents and one frozen reference in a single paired experiment.

All expensive work is delegated to run_mosp_shaping, which prepares shared data,
trains each condition, evaluates the reference once, and writes a combined report.
The small screening profile below can be overridden with the existing CLI flags.
"""
import sys
import re
from pathlib import Path

# Support both `python -m ...experiment_shaping` and the requested script path.
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from grpo_mosp import run_mosp_shaping as runner
else:
    from .. import run_mosp_shaping as runner

COMBINATIONS=('none','edge','path','hypervolume','cost_diversity','path_disjointness','all')
# Preserve the user's most recent screening configuration. The persistent JSON
# avoids relying on /tmp surviving between preparation, training and evaluation.
PROFILE=dict(curriculum=[1,2],attempts_per_graph=16,train_graphs=8,validation_graphs=2,
    training_runs=1,steps=32,group_size=4,epochs_per_group=1,early_stopping=False,
    eval_graphs=10,eval_ks=[2],eval_budget_multipliers=[8],eval_repetitions=1,
    selection_diagnostic=True,difficulty_cutoff=8,bootstrap_samples=200,
    reasoning_effort='low',preserve_thinking=False,max_task_tokens=4096,
    shuffle_graphs=True,balance_graphs=True,adaptive_graph_sampling=True,
    graph_curriculum_file=str(runner.base.HERE/'graph_curriculum.json'))


def parser():
    result=runner.parser()
    result.prog='experiment_shaping.sh'
    result.description=__doc__
    for action in result._actions:
        if action.dest in PROFILE and action.help:
            value=PROFILE[action.dest]
            if isinstance(value,bool):value=str(value).lower()
            action.help=re.sub(r'\s*\[default:.*\]\s*$','',action.help)+f' [experiment default: {value}]'
    result.epilog=('Fixed seven conditions: '+', '.join(COMBINATIONS)+
        '. New experiments default to 32 groups, group size 4, B=16, k curriculum 1,2, '
        '8 training graphs, 10 held-out graphs and both evaluation modes. '
        'All three sampling flags default true. Use --dry-run for resolved settings.')
    return result


def configure(args):
    """Apply profile only for a new experiment; later stages reuse saved settings."""
    explicit=vars(args)
    if any(key in explicit for key in ('shaping','shaping_signals','shaping_combinations')) or any(
            key.startswith('shaping_') for key in explicit):
        raise ValueError('experiment_shaping fixes the seven combinations; use run_mosp_shaping for other selections')
    output=runner.base.resolve_path(args.output) if args.output is not None else None
    # A saved model's launch config is authoritative. In particular, do not
    # restore profile values over deliberate overrides when resuming evaluation.
    saved=list(output.glob('*_trained/shaping/launch_config.json')) if output is not None else []
    if saved and 'model' in explicit:
        selected=output/f'{runner.model_slug(args.model)}_trained/shaping/launch_config.json'
        saved=[selected] if selected.exists() else []
    if not saved:
        for key,value in PROFILE.items():
            if key=='attempts_per_graph' and 'attempt_multiplier' in explicit:continue
            if key=='training_runs' and 'training_seeds' in explicit:continue
            if key not in explicit:setattr(args,key,value)
    args.shaping_combinations=list(COMBINATIONS)
    return args


def launch(args):
    args=configure(args)
    # Fail before data generation/model loading if this path points at a different
    # experiment. This also checks immutable settings on resumed runs.
    _,settings,_=runner.resolve(args)
    if settings['shaping_combinations']!=list(COMBINATIONS):
        raise ValueError('Saved run is not the seven-condition experiment')
    return runner.launch(args)


def main():
    command=parser();args=command.parse_args()
    try:launch(args)
    except KeyboardInterrupt:command.exit(130,'Seven-condition shaping experiment interrupted\n')
    except (ValueError,FileNotFoundError) as error:command.exit(2,f'experiment_shaping.sh: {error}\n')


if __name__=='__main__':main()
