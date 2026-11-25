"""Initialization identity and backend dispatch do not require downloading 27B weights."""
import json

import pytest

from .. import model_backends as backend
from ..model_defaults import DEFAULT_MODEL, DEFAULT_REVISION, initial_checkpoint, loading_contract
from .. import run_mosp


def test_default_plan_pairs_one_nf4_base_with_adapter_variants(tmp_path):
    args = run_mosp.parser().parse_args(['--dry-run', '--runs-root', str(tmp_path/'full_runs')])
    output, settings, _ = run_mosp.resolve(args, vars(args))
    _, configs, evaluation = run_mosp.configurations(output, settings)
    expected = initial_checkpoint(settings)
    assert expected['model'] == DEFAULT_MODEL
    assert expected['revision'] == DEFAULT_REVISION
    assert expected['loading']['quantization'] == 'nf4'
    assert expected['loading']['double_quant'] is True
    assert evaluation['initial_checkpoint'] == expected
    for _, config in configs:
        assert config['initial_checkpoint'] == expected
        assert config['lora_last_layers'] == 8 and config['lora_rank'] == 8
    for run in evaluation['training_runs']:
        assert {key:run['post_checkpoint'][key] for key in expected} == expected
        assert run['post_checkpoint']['adapter']


def test_train_and_eval_dispatch_share_identical_loading_contract(monkeypatch):
    seen = []
    monkeypatch.setattr(backend, '_load_nf4', lambda c,s,t: seen.append((s,t)))
    c = dict(model=DEFAULT_MODEL, revision=DEFAULT_REVISION)
    backend.load_model(c, training=True)
    backend.load_model(c, training=False)
    assert seen[0][0] == seen[1][0] == initial_checkpoint(c)
    assert [t for _,t in seen] == [True,False]
    # Refuse a spec that claims an incompatible quantization or thinking policy.
    wrong = initial_checkpoint(c)
    wrong['loading']['quantization'] = 'fp8'
    with pytest.raises(ValueError, match='loading contract'):
        backend.load_model(c,wrong)


def test_nf4_adapter_provenance_rejects_revision_or_compute_mismatch(tmp_path):
    spec = dict(model=DEFAULT_MODEL, revision=DEFAULT_REVISION)
    contract = loading_contract(spec)
    with pytest.raises(ValueError, match='provenance'):
        backend._verify_adapter_contract(tmp_path,spec,contract)
    metadata = dict(initial_checkpoint=dict(spec,loading=contract))
    (tmp_path/'mosp_backend.json').write_text(json.dumps(metadata))
    backend._verify_adapter_contract(tmp_path,spec,contract)
    for wrong in [dict(spec,revision='f'*40), dict(spec,model='Other/Checkpoint')]:
        with pytest.raises(ValueError, match='different base'):
            backend._verify_adapter_contract(tmp_path,wrong,contract)
    with pytest.raises(ValueError, match='loading contract'):
        backend._verify_adapter_contract(tmp_path,spec,dict(contract,compute_dtype='float16'))


def test_legacy_checkpoint_keeps_legacy_backend():
    c = dict(model='Qwen/Qwen2.5-1.5B-Instruct',revision='0'*40)
    assert loading_contract(c)['backend'] == 'transformers'
    assert initial_checkpoint(c) == c


def test_offloaded_model_uses_recorded_cuda_input_device():
    torch = pytest.importorskip('torch')
    from ..policy import ConstrainedPolicy
    class Model:
        device = torch.device('cpu')
        _mosp_device = torch.device('cuda:0')
    policy = ConstrainedPolicy(Model(),None)
    assert policy.device == torch.device('cuda:0')


def test_policy_loader_can_freeze_base_without_creating_random_adapters(monkeypatch):
    pytest.importorskip('torch')
    from types import SimpleNamespace
    from ..policy import load_policy
    calls = []
    def fake_load(config,spec,training):
        calls.append((spec,training))
        return SimpleNamespace(_mosp_loading=spec['loading']), None
    monkeypatch.setattr(backend,'load_model',fake_load)
    c = dict(model=DEFAULT_MODEL,revision=DEFAULT_REVISION,context_limit=4096,temperature=1.)
    frozen = load_policy(c,training=False)
    load_policy(c,training=True)
    assert calls[0][0] == calls[1][0]
    assert [training for _,training in calls] == [False,True]
    assert frozen.enable_thinking is False
    assert calls[0][0]['loading']['torch_compile'] is False
