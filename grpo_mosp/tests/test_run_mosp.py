"""Launcher checks exercise real argument forwarding and CPU graph generation."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from .. import run_mosp as runner
from ..dataset import load, generate
from ..eval_baselines import IsomorphismIndex

SCRIPT = Path(runner.__file__).with_suffix('.sh')


def run_path(tmp_path, label):
    return tmp_path/label/'full_runs'/'run_20260101120000'


def shell(*args):
    return subprocess.run([str(SCRIPT), '--python', sys.executable, *map(str,args)],
                          capture_output=True, text=True, check=False)


def test_shell_help_and_dry_run_with_complete_controls(tmp_path):
    assert shell('--help').returncode == 0
    output = run_path(tmp_path, "space and apostrophe's run")
    result = shell('--dry-run','--output',output,'--train-graphs','7',
        '--validation-graphs','3','--eval-graphs','10','--subgraphs','2','3',
        '--nodes-per-subgraph','5','8','--outdegree','2','3','--objectives','2',
        '--training-seeds','17,29','--curriculum','1,2','--attempts-per-graph','6',
        '--group-size','4','--steps','2','--learning-rate','0.00002',
        '--eval-repetitions','2','--eval-budget-multipliers','8')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert not output.exists()  # Dry runs do not write even config files.
    assert plan['settings']['training_runs'] == 2
    assert plan['graph_generation']['nodes_per_subgraph'] == [5,8]
    assert plan['evaluation_config']['graph_generation'] == plan['graph_generation']
    assert plan['evaluation_config']['initial_checkpoint']['revision'] == runner.DEFAULTS['revision']
    for config in plan['training_configs'].values():
        assert config['traversal_budget'] == 6 and config['attempt_multiplier'] is None
        assert config['group_size'] == 4 and config['learning_rate'] == .00002
        assert Path(config['output']).is_relative_to(output/'Qwen3.8-27B-unsloth-bnb-4bit_trained')
    assert Path(plan['evaluation_config']['model_output_dirs']['initial']).is_relative_to(output/'Qwen3.8-27B-unsloth-bnb-4bit_not_trained')
    commands = plan['commands']
    assert len(commands) == 6  # Two real training invocations, four eval stages.
    assert commands[0][2].endswith('.train')
    assert [c[3] for c in commands[2:]] == ['select-validation','prepare','run','summarize']


@pytest.mark.parametrize('flags', [
    ['--curriculum','1,3','--attempts-per-graph','2'],
    ['--nodes-per-subgraph','3','6','--outdegree','3'],
    ['--eval-graphs','11'], ['--subgraphs','1'],
    ['--eval-budget-multipliers','1,2'],
    ['--model','another/model'],
])
def test_invalid_cli_fails_before_writing(tmp_path,flags):
    output = run_path(tmp_path, 'invalid')
    result = shell('--dry-run','--output',output,*flags)
    assert result.returncode == 2 and not output.exists()


def prepare_args(output):
    return ['--mode','prepare','--output',str(output),'--train-graphs','2',
        '--validation-graphs','2','--eval-graphs','10','--training-runs','1',
        '--curriculum','1','--nodes-per-subgraph','5','7','--objectives','2',
        '--max-paths','128','--difficulty-cutoff','4','--eval-budget-multipliers','8']


def test_real_prepare_and_later_eval_load_identical_saved_settings(tmp_path):
    output = run_path(tmp_path, 'prepared run')
    result = shell(*prepare_args(output))
    assert result.returncode == 0, result.stderr
    index = IsomorphismIndex()
    for split in ('train','validation'):
        instances = load(output/'data'/f'{split}.jsonl')
        assert len(instances) == 2
        images = list((output/'data'/'images'/split).glob('*.png'))
        assert len(images) == 2 and all(i.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for i in images)
        for item in instances:
            assert 10 <= len(item.graph) <= 14 and len(item.names) == 2
            assert item.names == ('water','time')
            assert index.add_if_new(item)
            assert all('transition_prob' not in d for _,_,d in item.graph.edges(data=True))
    assert (output/'Qwen3.8-27B-unsloth-bnb-4bit_trained'/'logs'/'launcher.log').exists()
    run_manifest = json.loads((output/'run_manifest.json').read_text())
    assert all(Path(value).is_relative_to(output) for key,value in run_manifest['cache_and_temp_paths'].items() if key != 'PYTHONDONTWRITEBYTECODE')
    manifest_before = (output/'data'/'manifest.json').read_bytes()
    later = shell('--mode','eval','--output',output,'--dry-run')
    assert later.returncode == 0, later.stderr
    plan = json.loads(later.stdout)
    assert plan['settings']['nodes_per_subgraph'] == [5,7]
    assert plan['settings']['objectives'] == 2
    assert all(c[2].endswith('.eval_baselines') for c in plan['commands'])
    assert (output/'data'/'manifest.json').read_bytes() == manifest_before
    again = shell('--mode','prepare','--output',output)
    assert again.returncode == 0, again.stderr
    assert (output/'data'/'manifest.json').read_bytes() == manifest_before
    changed = shell('--mode','eval','--output',output,'--steps','999','--dry-run')
    assert changed.returncode == 2 and 'immutable' in changed.stderr


def test_orchestrator_runs_training_then_same_suite_comparison(tmp_path,monkeypatch):
    p = runner.parser()
    args = p.parse_args(prepare_args(run_path(tmp_path, 'orchestrated'))+['--mode','all'])
    calls = []
    monkeypatch.setattr(runner,'dependencies',lambda *a:None)
    monkeypatch.setattr(runner,'make_data',lambda *a:None)
    monkeypatch.setattr(runner,'execute',lambda command,logfile:calls.append(command))
    runner.launch(args,vars(args))
    assert calls[0][2].endswith('.train')
    assert [c[3] for c in calls[1:]] == ['select-validation','prepare','run','summarize']
    assert calls[2][-1] == calls[3][-1] == calls[4][-1]


def test_training_budget_resolves_cli_fixed_and_k_scaled_choices():
    pytest.importorskip('torch')
    from ..train import traversal_budget
    assert traversal_budget({},3) == 24
    assert traversal_budget({'attempt_multiplier':4},3) == 12
    assert traversal_budget({'traversal_budget':9,'attempt_multiplier':None},3) == 9
    with pytest.raises(ValueError): traversal_budget({'traversal_budget':2},3)


def test_default_timestamped_run_directory_and_custom_full_runs_root(tmp_path):
    result = shell('--dry-run','--runs-root',tmp_path/'full_runs','--training-runs','1')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    output = Path(plan['output'])
    assert output.parent == tmp_path/'full_runs'
    assert output.name.startswith('run_')
    assert len(output.name.rsplit('_',1)[-1]) == 14
    assert not output.exists()
    bad = shell('--dry-run','--output',tmp_path/'outside')
    assert bad.returncode == 2 and 'full_runs' in bad.stderr


def test_multiple_models_share_one_run_without_overwriting_settings(tmp_path):
    output = run_path(tmp_path,'multiple')
    first = shell(*prepare_args(output))
    assert first.returncode == 0, first.stderr
    first_settings = output/'Qwen3.8-27B-unsloth-bnb-4bit_trained'/'launch_config.json'
    original = first_settings.read_bytes()
    data = (output/'data'/'manifest.json').read_bytes()
    second = shell('--mode','prepare','--output',output,'--model','Synthetic/SecondQwen',
                   '--revision','1'*40,'--training-runs','1')
    assert second.returncode == 0, second.stderr
    assert first_settings.read_bytes() == original
    assert (output/'data'/'manifest.json').read_bytes() == data
    assert (output/'SecondQwen_trained'/'launch_config.json').exists()
    assert (output/'SecondQwen_not_trained').is_dir()
    registry = json.loads((output/'run_manifest.json').read_text())['models']
    assert len(registry) == 2
    ambiguous = shell('--mode','eval','--output',output,'--dry-run')
    assert ambiguous.returncode == 2 and 'multiple models' in ambiguous.stderr
    chosen = shell('--mode','eval','--output',output,'--model','Synthetic/SecondQwen','--dry-run')
    assert chosen.returncode == 0, chosen.stderr
    plan = json.loads(chosen.stdout)
    assert plan['settings']['objectives'] == 2  # Inherits the shared task design.
    assert plan['evaluation_config']['shared_suite'] == str(output/'suite')
    assert Path(plan['evaluation_config']['output']) == output/'comparisons'/'SecondQwen'


def test_repository_relative_output_is_stable_from_script_or_other_directory(tmp_path):
    relative = 'jm_test/graph_world/grpo_mosp/full_runs/run_20261103155500'
    for cwd in (SCRIPT.parent,tmp_path):
        result = subprocess.run([str(SCRIPT),'--python',sys.executable,'--dry-run','--output',relative],
                                cwd=cwd,capture_output=True,text=True)
        assert result.returncode == 0,result.stderr
        plan = json.loads(result.stdout)
        assert Path(plan['output']) == runner.ROOT/relative
        assert plan['settings']['history_format'] == 'state_actions'
        assert plan['evaluation_config']['history_format'] == 'state_actions'
        assert all(c['history_format']=='state_actions' for c in plan['training_configs'].values())
