"""Validate plateau decisions, exported weights, resource accounting and RNG isolation."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ..early_stopping import Plateau, validation_mode, validation_instances, validate
from ..training_options import TRAINING_DEFAULTS, add_training_arguments, apply_training_arguments
from .. import run_mosp
from ..policy import ConstrainedPolicy
from .helpers import fixture, tiny_model, ByteTokenizer


def test_schedule_defaults_and_cli_overrides():
    parser=argparse.ArgumentParser();add_training_arguments(parser)
    assert apply_training_arguments({},parser.parse_args([]))==TRAINING_DEFAULTS
    assert TRAINING_DEFAULTS['steps']==30 and TRAINING_DEFAULTS['epochs_per_group']==1
    assert apply_training_arguments({},parser.parse_args(['--early-stopping','false']))['early_stopping'] is False
    assert apply_training_arguments({},parser.parse_args(['--no-early-stopping']))['early_stopping'] is False
    assert apply_training_arguments({},parser.parse_args(['--steps','7','--epochs-per-group','2']))['steps']==7
    args=run_mosp.parser().parse_args(['--dry-run'])
    output,config,_=run_mosp.resolve(args,vars(args))
    _,training,evaluation=run_mosp.configurations(output,config)
    for _,c in training:
        assert c['steps']==30 and c['epochs_per_group']==1 and c['early_stopping'] is True
        assert c['validation_dataset']==evaluation['validation_dataset']
        assert c['validation_ks']==evaluation['ks']


def test_plateau_tracks_actual_best_separately_from_significant_improvement():
    monitor=Plateau(2,.1)
    assert monitor.observe(.4,0)
    assert monitor.observe(.45,1)  # Export a better adapter without resetting patience.
    assert monitor.stale==1 and monitor.best_step==1
    assert not monitor.observe(.45,2)  # Earliest checkpoint wins ties.
    assert monitor.stale==2 and monitor.best==.45
    assert monitor.observe(.6,3) and monitor.stale==0


@pytest.mark.parametrize('config',[dict(steps=0),dict(epochs_per_group=0),dict(early_stopping='true'),
    dict(early_stopping_patience=0),dict(early_stopping_min_delta=float('nan'))])
def test_invalid_monitor_controls_fail(config):
    with pytest.raises(ValueError):apply_training_arguments(config)


def test_validation_restores_rng_and_model_modes_even_on_failure():
    policy=ConstrainedPolicy(tiny_model(),ByteTokenizer())
    policy.model.train()
    random.seed(11);np.random.seed(12);torch.manual_seed(13)
    expected=(random.random(),np.random.random(),torch.rand(1))
    random.seed(11);np.random.seed(12);torch.manual_seed(13)
    with pytest.raises(RuntimeError,match='test interruption'):
        with validation_mode(policy):
            assert not policy.model.training and not torch.is_grad_enabled()
            random.seed(99);np.random.seed(99);torch.manual_seed(99)
            raise RuntimeError('test interruption')
    assert policy.model.training
    assert (random.random(),np.random.random())==expected[:2]
    assert torch.equal(torch.rand(1),expected[2])


def test_validation_rejects_training_graph_and_relabeled_copy(tmp_path):
    from ..eval_baselines import relabel
    for item in (fixture(),relabel(fixture(),123)):
        path=tmp_path/'validation.jsonl';path.write_text(json.dumps(item.record())+'\n')
        with pytest.raises(ValueError,match='overlaps training'):
            validation_instances(dict(validation_dataset=str(path),curriculum=[1],early_stopping_graphs=8),[fixture()])
    with pytest.raises(ValueError,match='requires validation_dataset'):
        validation_instances({},[fixture()])


def test_actual_validation_uses_fixed_tasks_and_preserves_training_samples():
    policy=ConstrainedPolicy(tiny_model(),ByteTokenizer(),context_limit=50000)
    policy.model.train()
    config=apply_training_arguments(dict(curriculum=[1],early_stopping_repetitions=2))
    random.seed(8);torch.manual_seed(9)
    state=(random.getstate(),torch.get_rng_state())
    first=validate(policy,[fixture()],config)
    assert random.getstate()==state[0] and torch.equal(torch.get_rng_state(),state[1])
    second=validate(policy,[fixture()],config)
    assert first['mean_S']==second['mean_S']
    assert [t['seed'] for t in first['tasks']]==[t['seed'] for t in second['tasks']]
    assert len(first['tasks'])==2 and policy.model.training
    assert all(t['metrics']['edge_traversals']<=8*8 for t in first['tasks'])


@pytest.mark.parametrize('curriculum,steps,expected_steps,best_step', [([1],10,3,1),([1,1,1],9,7,1),([1],3,3,1),([1],10,2,0)])
def test_training_stops_and_exports_best_adapter_not_last(tmp_path,monkeypatch,curriculum,steps,expected_steps,best_step):
    from .. import train
    model,tokenizer=tiny_model(),ByteTokenizer()
    exported=[]
    def save_model(path,**kwargs):
        Path(path).mkdir(parents=True,exist_ok=True)
        exported.append(model.weight.detach().clone())
        torch.save(model.state_dict(),Path(path)/'tiny.pt')
    model.save_pretrained=save_model
    tokenizer.save_pretrained=lambda path:(Path(path)/'tokenizer.json').write_text('{}')
    monkeypatch.setattr(train,'load_policy',lambda _:ConstrainedPolicy(model,tokenizer,context_limit=50000))
    for name in ('reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'CPU test')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch,'use_deterministic_algorithms',lambda *a,**kw:None)
    dataset=tmp_path/'train.jsonl';dataset.write_text(json.dumps(fixture().record())+'\n')
    validation_path=tmp_path/'validation.jsonl';validation_path.write_text('validation hash fixture')
    monkeypatch.setattr(train,'validation_instances',lambda *_:[fixture()])
    values=iter(([.2,.6] if best_step else [.6])+[.5]*steps)
    def measured(*_):
        return dict(mean_S=next(values),tasks=[dict(graph_id='fixed-validation',metrics=dict(generated_tokens=4,edge_traversals=2))])
    monkeypatch.setattr(train,'validate',measured)
    config=dict(model='Synthetic/EarlyStop',revision='0'*40,seed=17,context_limit=50000,
        temperature=1.,dataset=str(dataset),output=str(tmp_path/'result'),curriculum=curriculum,
        steps=steps,group_size=4,traversal_budget=1,objective='pareto',learning_rate=.001,
        beta=0.,clip=.2,epochs_per_group=1,max_grad_norm=1.,validation_dataset=str(validation_path),
        early_stopping=True,early_stopping_interval=1,early_stopping_patience=2)
    train.train(config)
    output=Path(config['output'])
    stopping=json.loads((output/'early_stopping.json').read_text())
    summary=json.loads((output/'training_summary.json').read_text())
    compute=json.loads((output/'training_compute.json').read_text())
    assert stopping['stopped_early']==(expected_steps<steps) and stopping['selected_step']==best_step
    assert stopping['completed_steps']==summary['iterations']==compute['steps']==expected_steps
    assert summary['tasks']==compute['complete_tasks']==expected_steps*4
    assert len(list(output.glob('rollout-*.json')))==expected_steps
    saved=torch.load(output/'adapter'/'tiny.pt',weights_only=True)['weight']
    assert len(exported)==best_step+1 and torch.equal(saved,exported[-1])
    assert not torch.equal(saved,model.weight.detach())
    assert compute['validation_tasks']==expected_steps+1
    assert (output/'training_state.pt').exists()
