"""CLI-to-template controls must match for frozen and adapted evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ..model_options import add_thinking_arguments, apply_thinking_arguments, thinking_options
from .. import run_mosp


@pytest.mark.parametrize('effort', ['xhigh', 'medium', 'low'])
@pytest.mark.parametrize('preserve', [True, False])
def test_cli_overrides_are_shared_by_training_and_comparison(effort, preserve):
    parser = run_mosp.parser()
    args = parser.parse_args(['--dry-run', '--reasoning-effort', effort,
                              '--preserve-thinking', str(preserve).lower()])
    explicit = vars(args)
    output, config, _ = run_mosp.resolve(args, explicit)
    _, training, evaluation = run_mosp.configurations(output, config)
    expected = dict(reasoning_effort=effort, preserve_thinking=preserve)
    assert thinking_options(evaluation) == expected
    assert all(thinking_options(c) == expected for _, c in training)
    # Template controls must not change the exact NF4 weight/loading identity.
    assert evaluation['initial_checkpoint'] == run_mosp.initial_checkpoint(run_mosp.DEFAULTS)


def test_missing_flags_preserve_config_and_aliases():
    p = argparse.ArgumentParser()
    add_thinking_arguments(p)
    assert apply_thinking_arguments({}, p.parse_args([])) == dict(reasoning_effort='xhigh', preserve_thinking=False)
    config = dict(reasoning_effort='low', preserve_thinking=True)
    assert apply_thinking_arguments(config.copy(), p.parse_args([])) == config
    assert apply_thinking_arguments({}, p.parse_args(['--reasoning_effort', 'medium', '--preserve_thinking', 'false'])) == dict(reasoning_effort='medium', preserve_thinking=False)
    assert p.parse_args(['--no-preserve-thinking']).preserve_thinking is False
    assert p.parse_args(['--preserve-thinking']).preserve_thinking is True
    for flags in [['--reasoning-effort', 'high'], ['--preserve-thinking', 'yes']]:
        with pytest.raises(SystemExit):
            p.parse_args(flags)
    with pytest.raises(ValueError, match='boolean'):
        thinking_options(dict(preserve_thinking='false'))


def test_template_kwargs_survive_evaluation_wrapper():
    from ..policy import ConstrainedPolicy
    from ..eval_baselines import limited_policy
    from .helpers import ByteTokenizer, tiny_model

    class RecordingTokenizer(ByteTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = RecordingTokenizer()
    policy = ConstrainedPolicy(tiny_model(), tokenizer, enable_thinking=False,
                               reasoning_effort='low', preserve_thinking=False)
    wrapped = limited_policy(policy, dict(max_action_tokens=512, max_task_tokens=16384))
    wrapped.prompt([dict(role='user', content='choose a path')])
    assert tokenizer.kwargs == dict(tokenize=True, add_generation_prompt=True,
                                   enable_thinking=False, reasoning_effort='low', preserve_thinking=False)
    assert (wrapped.reasoning_effort, wrapped.preserve_thinking) == ('low', False)


@pytest.mark.parametrize('module', ['train', 'eval_baselines', 'evaluate', 'smoke.smoke_nf4'])
def test_model_entry_points_expose_both_flags(module):
    result = subprocess.run([sys.executable, '-m', 'jm_test.graph_world.grpo_mosp.'+module, '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--reasoning-effort' in result.stdout and '--preserve-thinking' in result.stdout


def test_example_configs_record_defaults():
    directory = Path(run_mosp.__file__).parent/'configs'
    from ..run_artifacts import BASELINE_PRESET_DIR

    templates = [*directory.glob('*.json'), *BASELINE_PRESET_DIR.glob('*.json'), directory.parent/'smoke/smoke.json']
    for path in templates:
        config = json.loads(path.read_text())
        assert {k:config[k] for k in ('reasoning_effort', 'preserve_thinking')} == thinking_options({})
