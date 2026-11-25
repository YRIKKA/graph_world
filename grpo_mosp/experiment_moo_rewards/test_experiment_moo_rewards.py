"""Eight paired conditions must change real GRPO returns, not held-out metrics."""
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
from . import experiment_moo_rewards as experiment
from .. import eval_shaping as evaluation
from .. import eval_baselines as full_evaluation
from .. import dataset as dataset_module
from .moo_rewards import BASES, COMBINATIONS, apply_rewards, functions_for, options, validate
from ..rollout import collect
from ..tests.helpers import fixture


def test_legacy_suite_checks_relocated_source_digest(tmp_path, monkeypatch):
    config = {}
    root = Path(evaluation.__file__).parent
    migrations = json.loads((root/'configs/metadata/source_relocations.json').read_text())
    manifest = {
        'config_id': full_evaluation.digest(config),
        'excluded_files': {},
        'graph_images': {},
        'protocol_sources': {},
        'shaping_sources': {name: record['previous_sha256'][0] for name, record in migrations.items()},
    }
    for filename, key in [('suite.jsonl', 'suite_id'), ('transcripts.jsonl', 'transcripts_id')]:
        path = tmp_path/filename
        path.touch()
        manifest[key] = full_evaluation.file_hash(path)
    (tmp_path/'suite_manifest.json').write_text(json.dumps(manifest))
    assert evaluation.verify_suite(config, tmp_path, check_code=True) == manifest

    hash_file = full_evaluation.file_hash

    def hash_changed_source(path):
        return '0'*64 if Path(path).name == 'moo_rewards.py' else hash_file(path)

    monkeypatch.setattr(full_evaluation, 'file_hash', hash_changed_source)
    with pytest.raises(ValueError, match=r'Changed evaluation protocol source: moo_rewards\.py'):
        evaluation.verify_suite(config, tmp_path, check_code=True)


class Scripted:
    def __init__(self, actions):
        self.actions = iter(actions)

    def action(self, messages, env):
        return json.dumps(next(self.actions)), []


def test_default_plan_has_eight_paired_conditions(tmp_path):
    output = tmp_path/'full_runs/run_20260101120000'
    args = experiment.parser().parse_args(['--dry-run', '--output', str(output),
                                           '--training-seeds', '17,29'])
    root, settings, _ = experiment.resolve(args)
    _, training, config = experiment.configurations(root, settings)
    evaluation.validate_config(config)
    assert config['conditions'] == list(COMBINATIONS)
    assert len(training) == 16 and len(config['model_output_dirs']) == 17
    assert len({cfg['dataset'] for _, cfg in training}) == 1
    assert len({cfg['validation_dataset'] for _, cfg in training}) == 1
    assert len({json.dumps(cfg['initial_checkpoint'], sort_keys=True) for _, cfg in training}) == 1
    assert len({cfg['seed'] for _, cfg in training}) == 2
    for _, cfg in training:
        rewards = cfg['moo_rewards']
        assert rewards['functions'] == functions_for(cfg['moo_reward_condition'])
        assert rewards['functions'][0] in ('reward_valid', 'reward_valid_per_path')
        assert rewards['normalize'] is True
        assert rewards['weight_by_valid'] is True
        assert 'reward_doctrine' not in rewards['functions']
        assert 'shaping' not in cfg
    assert not root.exists()


def test_configuration_reuse_and_immutability(tmp_path):
    output = tmp_path/'full_runs/run_20260101120000'
    flags = ['--dry-run', '--output', str(output), '--steps', '6']
    root, settings, _ = experiment.resolve(experiment.parser().parse_args(flags))
    saved = root/f'{experiment.model_slug(settings["model"])}_trained/moo_rewards/launch_config.json'
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps(settings))
    _, reused, existing = experiment.resolve(experiment.parser().parse_args(
        ['--mode', 'eval', '--dry-run', '--output', str(output)]))
    assert existing and reused == settings
    with pytest.raises(ValueError, match='immutable'):
        experiment.resolve(experiment.parser().parse_args(flags + ['--steps', '7']))


def test_shell_dry_run():
    script = Path(experiment.__file__).with_suffix('.sh')
    result = subprocess.run([str(script), '--python', sys.executable, '--dry-run'],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads(result.stdout)['training_configs']) == 8


def test_additive_returns_reach_rollouts_and_keep_original_score():
    instance = fixture()
    actions = [dict(move_to=v) for v in ('a', 't', 'a', 't')]
    actions.append(dict(paths=[['s', 'a', 't'], ['s', 'a', 't']]))
    task = collect(Scripted(actions), instance, 2, 2)
    assert task.reward == task.env.reward == .5
    apply_rewards([task], instance, 2, options(COMBINATIONS[-1]))
    assert task.reward == pytest.approx(.5 + .5 * (1 + 1))
    assert task.env.reward == task.shaping['terminal_reward'] == .5
    assert all('cost' not in data for _, _, data in instance.graph.edges(data=True))


@pytest.mark.parametrize('condition', COMBINATIONS)
def test_all_conditions_are_failure_safe(condition):
    instance = fixture()
    task = collect(Scripted([dict(move_to='illegal')]), instance, 1, 1)
    apply_rewards([task], instance, 1, options(condition))
    assert task.reward == 0


def test_options_reject_non_normalized_rewards():
    config = options('valid')
    config['normalize'] = False
    with pytest.raises(ValueError, match='normalize=true'):
        validate(config)


def test_cpu_grpo_uses_additive_rewards(tmp_path, monkeypatch):
    import torch
    from .. import train
    from ..policy import ConstrainedPolicy
    from ..tests.helpers import tiny_model, ByteTokenizer

    model, tokenizer = tiny_model(), ByteTokenizer()
    before = model.weight.detach().clone()

    def save(path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(path)/'tiny.pt')

    model.save_pretrained = save
    tokenizer.save_pretrained = lambda path: (Path(path)/'tokenizer.json').write_text('{}')
    monkeypatch.setattr(train, 'load_policy', lambda _: ConstrainedPolicy(model, tokenizer, context_limit=50000))
    for name in ('reset_peak_memory_stats', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU test')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', lambda *a, **kw: None)
    dataset = tmp_path/'train.jsonl'
    dataset.write_text(json.dumps(fixture().record())+'\n')
    output = tmp_path/'training'
    config = dict(model='Synthetic/MooCPU', revision='0'*40, seed=17,
        context_limit=50000, temperature=1., dataset=str(dataset), output=str(output),
        curriculum=[1], steps=2, group_size=4, traversal_budget=1, objective='pareto',
        learning_rate=.001, beta=0., clip=.2, epochs_per_group=1, max_grad_norm=1.,
        early_stopping=False, moo_rewards=options(COMBINATIONS[-1]))
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    train.main(path)
    tasks = [row for p in output.glob('rollout-*.json') for row in json.loads(p.read_text())]
    assert len(tasks) == 8
    assert all(row['reward'] == row['shaping']['total_reward'] for row in tasks)
    for task in tasks:
        assert sum(task['shaping']['reward_contributions'].values()) == pytest.approx(task['reward'])
    metrics = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    for row in metrics:
        assert set(row['reward_components']) == set(config['moo_rewards']['functions'])
        assert sum(row['reward_contributions'].values()) == pytest.approx(row['mean_reward'])
    assert any(row['reward'] > row['shaping']['terminal_reward'] for row in tasks)
    assert not torch.equal(before, model.weight.detach())


def test_cpu_evaluation_pairs_all_conditions_against_validity(tmp_path):
    from ..policy import ConstrainedPolicy
    from ..tests.helpers import tiny_model, ByteTokenizer

    root = tmp_path/'full_runs/run_20260101120000'
    args = experiment.parser().parse_args(['--mode', 'prepare', '--output', str(root),
        '--model', 'Synthetic/Moo', '--revision', '0'*40,
        '--backend', 'transformers', '--train-graphs', '3', '--validation-graphs', '1',
        '--steps', '6', '--curriculum-k', '1', '--eval-ks', '1', '--eval-graphs', '10', '--max-paths', '128',
        '--difficulty-cutoff', '4', '--bootstrap-samples', '3', '--max-task-tokens', '64'])
    root, settings, _ = experiment.resolve(args)
    root.mkdir(parents=True)
    _, training, config = experiment.configurations(root, settings)
    experiment.base.make_data(root, settings, experiment.configurations(root, settings)[0])
    config['device'] = 'cpu'
    for path, cfg in training:
        directory = Path(cfg['output'])
        (directory/'adapter').mkdir(parents=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(cfg))
        (directory/'config.json').write_text(json.dumps(cfg))
        (directory/'adapter/weights.bin').write_bytes(str(path).encode())
        (directory/'training_state.pt').write_bytes(b'CPU evaluation fixture')
        (directory/'training_summary.json').write_text('{}')
        (directory/'early_stopping.json').write_text(json.dumps(dict(enabled=False, selected_step=6)))
        (directory/'training_compute.json').write_text(json.dumps(dict(kind='CPU fixture')))
    output = Path(config['output'])
    evaluation.prepare(config, output)
    loaded = []

    def loader(spec, controls):
        loaded.append(spec)
        return ConstrainedPolicy(tiny_model(), ByteTokenizer(), controls['context_limit'])

    evaluation.run(config, output, loader)
    assert len(loaded) == 9 and sum('adapter' not in spec for spec in loaded) == 1
    records = evaluation.summarize(config, output, plots=False)
    assert all(any(row['condition'] == condition and row['reference'] == 'initial'
                   and row['metric'] == 'S' for row in records) for condition in COMBINATIONS)
    assert all(any(row['condition'] == condition and row['reference'] == 'valid'
                   and row['metric'] == 'S' for row in records) for condition in COMBINATIONS if condition != 'valid')
    assert 'Change vs own validity baseline' in (output/'comparison.md').read_text()


def test_eight_distinct_weighted_reward_settings():
    assert len(COMBINATIONS) == 8
    settings = [options(condition) for condition in COMBINATIONS]
    assert len({json.dumps(config, sort_keys=True) for config in settings}) == 8
    assert sum(len(config['functions']) == 1 for config in settings) == 2
    assert all(config['weight_by_valid'] is True for config in settings)
    for base in BASES:
        for suffix in ('hypervolume', 'igd', 'hypervolume+igd'):
            assert options(f'{base}+{suffix}__weighted')['weight_by_valid'] is True
            with pytest.raises(ValueError, match='Unknown MOO reward condition'):
                options(f'{base}+{suffix}__unweighted')


@pytest.mark.parametrize('base', BASES)
def test_validity_weighting_changes_only_added_rewards(base):
    instance = fixture()
    actions = [dict(move_to=v) for v in ('a', 't', 'a', 't')]
    actions.append(dict(paths=[['s', 'a', 't'], ['s', 'a', 't']]))
    task = collect(Scripted(actions), instance, 2, 2)
    condition = f'{base}+hypervolume+igd__weighted'
    apply_rewards([task], instance, 2, options(condition))
    validity = 1 if base == 'valid_per_path' else .5
    assert task.reward == pytest.approx(validity + 1)


def test_cli_matches_experiment_shaping_exactly():
    from ..experiment_shaping import experiment_shaping

    shaping = experiment_shaping.parser()
    moo = experiment.parser()
    assert set(moo._option_string_actions) == set(shaping._option_string_actions)
    for flag, expected in shaping._option_string_actions.items():
        actual = moo._option_string_actions[flag]
        assert type(actual) is type(expected), flag
        for attribute in ('dest', 'nargs', 'type', 'default', 'const', 'required', 'choices'):
            assert getattr(actual, attribute) == getattr(expected, attribute), (flag, attribute)


@pytest.mark.parametrize('flags', [['--shaping', 'factorial'], ['--shaping-edge', 'true']])
def test_fixed_condition_selectors_rejected_like_shaping(flags):
    from ..experiment_shaping import experiment_shaping

    with pytest.raises(ValueError, match='fixes'):
        experiment_shaping.configure(experiment_shaping.parser().parse_args(['--dry-run', *flags]))
    with pytest.raises(ValueError, match='fixes'):
        experiment.resolve(experiment.parser().parse_args(['--dry-run', *flags]))


def test_default_output_has_moo_rewards_prefix(tmp_path):
    args = experiment.parser().parse_args(['--dry-run', '--runs-root', str(tmp_path/'full_runs')])
    root, settings, _ = experiment.resolve(args)
    assert root.parent == tmp_path/'full_runs'
    assert root.name.startswith('moo_rewards_run_')
    assert len(root.name.removeprefix('moo_rewards_run_')) == 14
    _, training, evaluation = experiment.configurations(root, settings)
    for path, config in training:
        assert Path(path).is_relative_to(root)
        assert Path(config['output']).is_relative_to(root)
    assert Path(evaluation['output']).is_relative_to(root)
    assert not root.exists()


def test_runtime_limited_default_profile():
    _, settings, _ = experiment.resolve(experiment.parser().parse_args(['--dry-run']))
    assert settings['steps'] == 40 and settings['group_size'] == 4
    assert settings['training_runs'] == 1
    assert settings['train_graphs'] == 16 and settings['validation_graphs'] == 4
    assert settings['bootstrap_samples'] == 500
    assert settings['eval_graphs'] == 20 and settings['eval_repetitions'] == 1
    assert settings['curriculum'] == [2] and settings['eval_ks'] == [2]
    assert settings['graph_curriculum'] is None
    assert settings['early_stopping'] is False
    assert settings['adaptive_graph_sampling'] is False
    # MOO runtime changes must not change the shaping experiment defaults.
    assert experiment.shaping_experiment.PROFILE['steps'] == 32


def test_unweighted_configuration_rejected():
    config = options('valid+igd__weighted')
    config['weight_by_valid'] = False
    with pytest.raises(ValueError, match='weight_by_valid=true'):
        validate(config)


@pytest.mark.parametrize('flags,expected_k,graph_enabled', [
    ([], [2], False),
    (['--curriculum-k', '1,3,5'], [1, 3, 5], False),
    (['--curriculum-k', '5'], [5], False),
    (['--curriculum-graph'], [2], True),
    (['--curriculum-graph', '--curriculum-k', '1,3,5'], [1, 3, 5], True),
])
def test_curricula_are_independently_opt_in(flags, expected_k, graph_enabled):
    args = experiment.parser().parse_args(['--dry-run', *flags])
    root, settings, _ = experiment.resolve(args)
    assert settings['curriculum'] == expected_k
    assert (settings['graph_curriculum'] is not None) is graph_enabled
    _, training, _ = experiment.configurations(root, settings)
    for _, cfg in training:
        assert cfg['curriculum'] == expected_k
        assert ('graph_curriculum_datasets' in cfg) is graph_enabled


def test_fixed_graph_flags_are_honored_without_curriculum():
    args = experiment.parser().parse_args(['--dry-run', '--subgraphs', '2',
        '--nodes-per-subgraph', '5', '8', '--outdegree', '3'])
    _, settings, _ = experiment.resolve(args)
    assert settings['graph_curriculum'] is None
    assert settings['nodes_per_subgraph'] == [5, 8]
    assert settings['subgraphs'] == [2, 2] and settings['outdegree'] == [3, 3]


def test_saved_curricula_preserved_when_flags_omitted(tmp_path):
    output = tmp_path/'full_runs/moo_rewards_run_20260101120000'
    args = experiment.parser().parse_args(['--dry-run', '--output', str(output),
                                           '--curriculum-graph', '--curriculum-k', '1,3,5'])
    _, settings, _ = experiment.resolve(args)
    path = output/f'{experiment.model_slug(settings["model"])}_trained/moo_rewards/launch_config.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(settings))
    _, reused, exists = experiment.resolve(experiment.parser().parse_args(
        ['--mode', 'eval', '--dry-run', '--output', str(output)]))
    assert exists and reused == settings


def test_launch_parameters_printed_and_logged_before_work(tmp_path, monkeypatch, capsys):
    root = tmp_path/'a path with spaces/full_runs/moo_rewards_run_20260101120000'
    flags = ['--mode', 'prepare', '--output', str(root), '--steps', '6',
             '--train-graphs', '3', '--validation-graphs', '1', '--no-plots']
    args = experiment.parser().parse_args(flags)
    monkeypatch.setattr(experiment.base, 'dependencies', lambda *a: None)
    monkeypatch.setattr(experiment.base, 'make_data', lambda *a: None)
    experiment.launch(args, argv=flags)
    console = capsys.readouterr().out
    log = (root/'log.txt').read_text()
    for text in (console, log):
        assert 'MOO reward experiment | launch parameters' in text
        assert '--steps 6' in text
        assert "'" + str(root) + "'" in text  # The command preserves shell quoting.
        assert '"graph_curriculum_enabled": false' in text
        assert '"k_curriculum_enabled": false' in text
        assert '"planned_training_runs": 8' in text
        assert '"planned_training_rollouts": 192' in text
        assert '"plots_enabled": false' in text
        assert '"resolved_settings"' in text and '"learning_rate"' in text
        assert text.index('launch parameters') < text.index('[Timing] START Dependencies')


def test_dry_run_includes_parameters_without_creating_output(tmp_path, capsys):
    root = tmp_path/'full_runs/moo_rewards_run_20260101120000'
    flags = ['--dry-run', '--output', str(root), '--curriculum-k', '1,3,5']
    experiment.launch(experiment.parser().parse_args(flags), argv=flags)
    plan = json.loads(capsys.readouterr().out)
    summary = plan['launch_summary']
    assert summary['cli_flags']['curriculum'] == [1, 3, 5]
    assert summary['k_curriculum_enabled'] is True
    assert summary['resolved_settings'] == plan['settings']
    assert summary['frozen_base_comparison'] is True
    assert not root.exists()


@pytest.mark.parametrize('solver', ['dijkstra', 'exhaustive', 'paretoset'])
@pytest.mark.parametrize('curriculum', [False, True])
def test_pareto_solver_reaches_every_generation_spec(solver, curriculum):
    flags = ['--dry-run', '--pareto-solver', solver]
    if curriculum:
        flags.append('--curriculum-graph')
    root, settings, _ = experiment.resolve(experiment.parser().parse_args(flags))
    graph, training, config = experiment.configurations(root, settings)
    assert settings['pareto_solver'] == solver
    assert graph['pareto_solver'] == solver
    assert config['graph_generation']['pareto_solver'] == solver
    for _, cfg in training:
        assert cfg['pareto_solver'] == solver
        for pool in cfg.get('graph_curriculum_datasets', []):
            assert pool['graph_generation']['pareto_solver'] == solver


def test_pareto_solver_default_and_invalid_flag():
    _, settings, _ = experiment.resolve(experiment.parser().parse_args(['--dry-run']))
    assert settings['pareto_solver'] == 'exhaustive'
    with pytest.raises(SystemExit):
        experiment.parser().parse_args(['--pareto-solver', 'unknown'])


@pytest.mark.parametrize('solver', ['dijkstra', 'exhaustive', 'paretoset'])
def test_cost_shift_uses_selected_pareto_solver(monkeypatch, solver):
    monkeypatch.setattr(full_evaluation, 'generate', lambda *a, **kw: deepcopy(fixture()))
    original = full_evaluation.solve_pareto_paths
    calls = []

    def track(*args, **kwargs):
        calls.append(kwargs['pareto_solver'])
        return original(*args, **kwargs)

    monkeypatch.setattr(full_evaluation, 'solve_pareto_paths', track)
    item = full_evaluation.make_candidate(1, 'costs', 1, dict(pareto_solver=solver, max_paths=128))
    assert calls == [solver]
    assert item.pareto == fixture().pareto


@pytest.mark.parametrize('solver', ['dijkstra', 'exhaustive', 'paretoset'])
def test_dataset_passes_solver_to_graph_constructor(monkeypatch, solver):
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('constructor reached')

    monkeypatch.setattr(dataset_module, 'GraphGenerator', capture)
    with pytest.raises(RuntimeError, match='constructor reached'):
        dataset_module.generate(42, graph_config=dict(pareto_solver=solver))
    assert calls[0]['pareto_solver'] == solver
