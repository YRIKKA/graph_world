"""Paired ablation setup and tiny-model evaluation reuse existing controls."""
import json
from pathlib import Path
import subprocess
import sys
from .. import run_mosp_shaping as runner
from .. import run_mosp as base
from ..run_artifacts import model_slug


def test_default_and_single_condition_plans_write_nothing(tmp_path):
    output = tmp_path/'full_runs/run_20260101120000'
    for condition, count in [('ablation', 4), ('none', 1), ('edge', 1), ('path', 1), ('both', 1)]:
        args = runner.parser().parse_args(['--dry-run', '--output', str(output), '--shaping', condition])
        root, settings, _ = runner.resolve(args)
        _, training, evaluation = runner.configurations(root, settings)
        assert len(training) == count and not output.exists()
        assert evaluation['graphs'] == 20 and evaluation['ks'] == [3]
        assert evaluation['budget_multipliers'] == [8] and evaluation['repetitions'] == 1
        assert len({c['dataset'] for _, c in training}) == 1
        assert len({c['validation_dataset'] for _, c in training}) == 1
        assert len({json.dumps(c['initial_checkpoint'], sort_keys=True) for _, c in training}) == 1
        assert len(evaluation['model_output_dirs']) == count + 1


def test_shell_prepare_and_saved_eval_plan(tmp_path):
    script = Path(runner.__file__).parent/'experiment_shaping/run_mosp_shaping.sh'
    output = tmp_path/'full_runs/run_20260101120000'
    arguments = ['--python', sys.executable, '--mode', 'prepare', '--output', str(output),
        '--train-graphs', '2', '--validation-graphs', '2', '--eval-graphs', '10', '--eval-ks', '1',
        '--curriculum-k', '1', '--nodes-per-subgraph', '5', '7', '--objectives', '2',
        '--max-paths', '128', '--difficulty-cutoff', '4', '--training-seeds', '17,29']
    result = subprocess.run([str(script), *arguments], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (output/'log.txt').read_text().splitlines()[-1].startswith('TOTAL PIPELINE ELAPSED TIME:')
    trained = output/f'{model_slug(base.DEFAULTS["model"])}_trained/shaping'
    assert len(list((trained/'configs').glob('*_seed_*.json'))) == 8
    assert len(list((output/'data/images').rglob('*.png'))) == 4
    later = subprocess.run([str(script), '--python', sys.executable, '--mode', 'eval',
                           '--output', str(output), '--dry-run'], capture_output=True, text=True)
    assert later.returncode == 0, later.stderr
    plan = json.loads(later.stdout)
    assert len(plan['training_configs']) == 8 and plan['settings']['training_seeds'] == [17, 29]


def test_light_eval_actual_cpu_loop_and_exports(tmp_path):
    from .. import eval_shaping as light
    from ..policy import ConstrainedPolicy
    from .helpers import tiny_model, ByteTokenizer
    output = tmp_path/'full_runs/run_20260101120000'
    args = runner.parser().parse_args(['--mode', 'prepare', '--output', str(output),
        '--model', 'Synthetic/Tiny', '--revision', '0'*40, '--backend', 'transformers',
        '--train-graphs', '1', '--validation-graphs', '1', '--eval-graphs', '10', '--eval-ks', '1',
        '--curriculum-k', '1', '--nodes-per-subgraph', '5', '7', '--objectives', '2',
        '--max-paths', '128', '--difficulty-cutoff', '4', '--bootstrap-samples', '3',
        '--max-task-tokens', '64', '--early-stopping', 'false',
        '--shaping','factorial','--shaping-signals','edge','path','hypervolume'])
    root, settings, _ = runner.resolve(args)
    root.mkdir(parents=True)
    graph, training, config = runner.configurations(root, settings)
    base.make_data(root, settings, graph)
    config['device'] = 'cpu'
    for path, c in training:
        directory = Path(c['output'])
        (directory/'adapter').mkdir(parents=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(c))
        (directory/'config.json').write_text(json.dumps(c))
        # Synthetic checkpoint fixtures only; no pretrained weights or learning claim.
        (directory/'adapter/weights.bin').write_bytes(str(path).encode())
        (directory/'training_state.pt').write_bytes(b'CPU fixture only')
        (directory/'training_summary.json').write_text('{}')
        (directory/'early_stopping.json').write_text(json.dumps(dict(enabled=False, selected_step=30)))
        (directory/'training_compute.json').write_text(json.dumps(dict(kind='synthetic CPU fixture')))
    light.validate_config(config)
    evaluation = Path(config['output'])
    light.prepare(config, evaluation)
    policies = []
    def loader(spec, controls):
        policy = ConstrainedPolicy(tiny_model(), ByteTokenizer(), controls['context_limit'])
        policies.append(policy)
        return policy
    light.run(config, evaluation, loader)
    rows, suite = light.checked_results(config, evaluation)
    assert len(policies) == 9 and len(rows) == 180
    assert all(not p.model.training and not any(v.requires_grad for v in p.model.parameters()) for p in policies)
    assert all(r['reward'] == r['metrics']['S'] and 0 <= r['reward'] <= 1 for r in rows)
    for graph in suite:
        for mode in config['modes']:
            assert len({r['sampling_seed'] for r in rows if r['graph_id'] == graph['graph_id'] and r['mode'] == mode}) == 1
    summary = light.summarize(config, evaluation, plots=True)
    assert any(r['condition'] == 'both' and r['reference'] == 'none' for r in summary)
    assert (evaluation/'comparison.csv').exists() and (evaluation/'comparison.md').exists()
    assert (root/'figures/Tiny/evaluation/independent_exploration/checkpoint_comparison/correct_pareto_fraction.png').exists()
    light.run(config, evaluation, loader)
    assert len(policies) == 9


def test_crossed_bootstrap_preserves_seed_and_graph_pairing():
    import numpy as np
    from ..eval_shaping import paired_estimate
    metadata = {str(i):dict(domain='in_distribution', difficulty='easy' if i<2 else 'hard') for i in range(4)}
    means = {}
    for seed in (1, 2):
        for i in range(4):
            value = .1 + .1*i + .05*seed
            means[f'none-{seed}', str(i)] = np.array([value])
            means[f'both-{seed}', str(i)] = np.array([value+.1])
    config = dict(bootstrap_samples=40, bootstrap_seed=17, difficulty_weights={'easy':.5, 'hard':.5})
    point, low, high = paired_estimate(means, metadata, ['both-1', 'both-2'], ['none-1', 'none-2'], config, 'in_distribution')
    # Every paired difference is exactly .1 despite variation in both factors.
    assert np.allclose(point[2], .1) and np.allclose(low[2], .1) and np.allclose(high[2], .1)
