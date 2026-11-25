"""Figures preserve optimizer counts, curriculum phases and weighted estimates."""
import csv
import json
from pathlib import Path

import pytest

from .. import figures


def training_rows():
    return [dict(step=i,mean_reward=.7,zero_variance_group=int(i==0),
        graph_stage=0 if i<2 else 1,graph_pool_stage=0,rehearsal=i==2,
        performance=dict(S=.5,complete_success=.25,valid_path_fraction=.75,duplicate_rate=.125))
        for i in range(3)]


def test_optimizer_axis_counts_skips_and_multiple_epochs():
    config=dict(beta=0.,epochs_per_group=2,objective='pareto')
    points,inferred=figures.training_points(config,training_rows())
    assert inferred
    assert [p['optimizer_step'] for p in points]==[0,0,2]
    assert [p['optimizer_step_after'] for p in points]==[0,2,4]
    assert all(p['reward']==.7 and p['correct_pareto_fraction']==.5 for p in points)
    recorded=training_rows()
    for i,row in enumerate(recorded):
        row.update(optimizer_step_before=i*3,optimizer_step=(i+1)*3)
    points,inferred=figures.training_points(config,recorded)
    assert not inferred and [p['optimizer_step'] for p in points]==[0,3,6]


def test_training_exports_all_metrics_and_each_curriculum_stage(tmp_path):
    root=tmp_path/'full_runs/run_20260101120000'
    output=root/'Plots_trained/shaping/both/runs/seed-17'
    output.mkdir(parents=True)
    source=output/'metrics.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in training_rows()))
    config=dict(model='Synthetic/Plots',seed=17,output=str(output),beta=0.,epochs_per_group=2,
        objective='pareto',shaping_condition='both',graph_curriculum=dict(stages=[{}, {}, {}]))
    files=figures.training_curves(config)
    directory=root/'figures/Plots/training/both/seed-17'
    assert len(files)==40  # 5 metrics x (trajectory + 3 stages) x PNG/SVG.
    for name in ('trajectory','curriculum_stage_01','curriculum_stage_02','curriculum_stage_03'):
        assert {p.stem for p in (directory/name).glob('*.png')}=={key for key,_ in figures.METRICS}
    assert (directory/'trajectory/reward.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert 'Stage not reached' in (directory/'curriculum_stage_03/reward.svg').read_text()
    with (directory/'training_curve_data.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert [float(r['optimizer_step']) for r in rows]==[0,0,2]
    assert rows[-1]['rehearsal']=='True' and rows[-1]['graph_stage']=='1'
    assert source.read_text().count('\n')==3  # Export does not rewrite training records.
    assert json.loads((directory/'manifest.json').read_text())['optimizer_counts_inferred']
    import matplotlib.pyplot as plt
    assert plt.get_fignums()==[]


def evaluation_records():
    records=[]
    for mode in ('independent_exploration','shared_transcript_selection'):
        for scope in ('in_distribution','distribution_shift_macro'):
            for identifier,value in [('none-seed-17',.3),('both-seed-17',.4),('both',.99)]:
                for metric in ('S','complete_success','valid_path_fraction','duplicate_rate'):
                    records.append(dict(mode=mode,scope=scope,condition=identifier,reference='initial',
                        metric=metric,before=.2,after=value,before_ci_low=.1,before_ci_high=.25,
                        after_ci_low=value-.05,after_ci_high=min(1.,value+.05)))
    return records


def evaluation_config():
    return dict(initial_checkpoint=dict(model='Synthetic/Plots'),training_runs=[
        dict(id='none-seed-17',condition='none',seed=17),dict(id='both-seed-17',condition='both',seed=17)])


def test_evaluation_points_include_each_checkpoint_and_exclude_condition_averages():
    points=figures.evaluation_points(evaluation_config(),evaluation_records())
    assert {p['model_id'] for p in points}=={'initial','none-seed-17','both-seed-17'}
    assert all(p['value']!=.99 for p in points)
    assert all(p['value']==.2 for p in points if p['model_id']=='initial')
    # Convert to the full evaluator's schema and verify identical plot values.
    full=[]
    for row in evaluation_records():
        row=dict(row)
        for old,new in [('condition','comparison'),('before','baseline'),('after','post'),
            ('before_ci_low','baseline_ci_low'),('before_ci_high','baseline_ci_high'),
            ('after_ci_low','post_ci_low'),('after_ci_high','post_ci_high')]:row[new]=row.pop(old)
        row.update(k='all',difficulty='all',budget_multiplier=8)
        full.append(row)
    assert figures.evaluation_points(evaluation_config(),full)==points


def test_bar_exports_all_models_modes_and_five_metrics(tmp_path):
    root=tmp_path/'full_runs/run_20260101120000'
    output=root/'comparisons/Plots/shaping'
    config=evaluation_config()
    records=evaluation_records()
    original=json.dumps(records,sort_keys=True)
    files=figures.evaluation_bars(config,records,output)
    directory=root/'figures/Plots/evaluation'
    assert len(files)==20  # 5 metrics x 2 modes x PNG/SVG; each has domain panels.
    for mode in ('independent_exploration','shared_transcript_selection'):
        assert {p.stem for p in (directory/mode/'checkpoint_comparison').glob('*.png')}=={k for k,_ in figures.METRICS}
    with (directory/'comparison_figure_data.csv').open() as stream:points=list(csv.DictReader(stream))
    assert {p['model_id'] for p in points}=={'initial','none-seed-17','both-seed-17'}
    assert all(float(p['value'])!=.99 for p in points)
    assert json.dumps(records,sort_keys=True)==original
    assert json.loads((directory/'manifest.json').read_text())['evaluation_reward']=='S'


def test_shaping_launcher_exports_training_figures_for_each_condition(tmp_path,monkeypatch):
    from .. import run_mosp_shaping as runner
    root=tmp_path/'full_runs/run_20260101120000'
    args=runner.parser().parse_args(['--mode','train','--output',str(root)])
    monkeypatch.setattr(runner.base,'dependencies',lambda *a:None)
    monkeypatch.setattr(runner.base,'make_data',lambda *a:None)
    monkeypatch.setattr(runner.base,'execute',lambda *a:None)
    captured=[]
    monkeypatch.setattr(figures,'training_curves',lambda c:captured.append(c['shaping_condition']))
    runner.launch(args)
    assert captured==['none','edge','path','both']


def test_moo_training_exports_separate_reward_curves_for_all_conditions(tmp_path):
    from ..experiment_moo_rewards.moo_rewards import COMBINATIONS, options

    root = tmp_path/'full_runs/run_20260101120000'
    directories = set()
    # Limit this regression to the reward figure; the other metrics and
    # curriculum exports are covered above.
    original_metrics = figures.METRICS
    figures.METRICS = [('reward', 'Mean reward')]
    try:
        for condition in COMBINATIONS:
            reward_config = options(condition)
            output = root/'Plots_trained/moo_rewards'/condition/'runs/seed-17'
            output.mkdir(parents=True)
            rows = training_rows()
            for row in rows:
                raw = {name: {'reward_valid': .5, 'reward_valid_per_path': 2,
                              'reward_hypervolume': .8, 'reward_igd': .6}[name]
                       for name in reward_config['functions']}
                contributions = {name: value * (.5 if reward_config['weight_by_valid']
                                     and name not in ('reward_valid', 'reward_valid_per_path') else 1)
                                 for name, value in raw.items()}
                row.update(mean_reward=sum(contributions.values()),
                           reward_components=raw, reward_contributions=contributions)
            (output/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            config = dict(model='Synthetic/Plots', seed=17, output=str(output), beta=0.,
                          epochs_per_group=1, moo_reward_condition=condition, moo_rewards=reward_config)
            files = figures.training_curves(config)
            assert len(files) == 2
            directory = Path(files[0]).parent.parent
            directories.add(directory)
            with (directory/'training_curve_data.csv').open() as stream:
                points = list(csv.DictReader(stream))
            for point in points:
                total = sum(float(point['contribution_' + name]) for name in reward_config['functions'])
                assert total == pytest.approx(float(point['reward']))
            svg = (directory/'trajectory/reward.svg').read_text()
            assert 'Aggregate reward' in svg
            if 'reward_hypervolume' in reward_config['functions']:
                assert 'Hypervolume' in svg
                assert float(points[0]['raw_reward_hypervolume']) == .8
            if 'reward_igd' in reward_config['functions']:
                assert 'IGD+' in svg
            manifest = json.loads((directory/'manifest.json').read_text())
            assert manifest['condition'] == condition
            assert manifest['moo_rewards'] == reward_config
        assert len(directories) == 8
    finally:
        figures.METRICS = original_metrics


@pytest.mark.parametrize('skip_plots', [False, True])
def test_moo_launcher_exports_training_figures_for_every_reward(tmp_path, monkeypatch, skip_plots):
    from ..experiment_moo_rewards import experiment_moo_rewards as experiment
    from ..experiment_moo_rewards.moo_rewards import COMBINATIONS

    root = tmp_path/'full_runs/run_20260101120000'
    args = experiment.parser().parse_args(['--mode', 'train', '--output', str(root)]
                                          + (['--no-plots'] if skip_plots else []))
    monkeypatch.setattr(experiment.base, 'dependencies', lambda *a: None)
    monkeypatch.setattr(experiment.graphs, 'make_data', lambda *a: None)
    monkeypatch.setattr(experiment.base, 'make_data', lambda *a: None)
    monkeypatch.setattr(experiment.base, 'execute', lambda *a: None)
    captured = []
    monkeypatch.setattr(figures, 'training_curves', lambda config: captured.append(config['moo_reward_condition']))
    experiment.launch(args)
    assert captured == ([] if skip_plots else list(COMBINATIONS))


def test_saved_export_discovers_every_moo_condition(tmp_path, monkeypatch):
    from ..experiment_moo_rewards.moo_rewards import COMBINATIONS

    root = tmp_path/'full_runs/run_20260101120000'
    for condition in COMBINATIONS:
        path = root/'Plots_trained/moo_rewards'/condition/'runs/seed-17/config.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(moo_reward_condition=condition)))
    captured = []
    monkeypatch.setattr(figures, 'training_curves',
                        lambda config: captured.append(config['moo_reward_condition']) or [])
    figures.export_run(root)
    assert set(captured) == set(COMBINATIONS) and len(captured) == 8
