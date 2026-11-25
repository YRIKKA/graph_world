"""Observable bonuses must be capped, task-deduplicated and failure-safe."""
import json
import pytest
from ..rollout import collect
from ..shaping import for_condition, options
from .helpers import fixture


class Scripted:
    def __init__(self, actions): self.actions = iter(actions)
    def action(self, messages, env): return json.dumps(next(self.actions)), []


def task(condition='both', submission=None):
    moves = [dict(move_to=v) for v in ('a', 't', 'a', 't', 'b', 't')]
    moves.append(submission or dict(paths=[['s', 'a', 't']]))
    return collect(Scripted(moves), fixture(), 1, 3,
                   shaping=for_condition(condition, edge_cap=3))


@pytest.mark.parametrize('condition,bonus', [('none', 0), ('edge', .01), ('path', .05), ('both', .06)])
def test_caps_deduplicate_across_attempts_and_keep_env_reward_unshaped(condition, bonus):
    result = task(condition)
    assert result.reward == pytest.approx(1 + bonus)
    assert result.env.reward == 1 and result.shaping['terminal_reward'] == 1
    assert result.shaping['unique_edges'] == 4 and result.shaping['unique_paths'] == 2
    assert len(result.shaping['events']) == 4
    assert 'bonus' not in result.env.observation()


def test_dominated_paths_receive_nonoracle_completion_bonus():
    actions = [dict(move_to='c'), dict(move_to='t'), dict(paths=[['s', 'c', 't']])]
    result = collect(Scripted(actions), fixture(), 1, 1, shaping=for_condition('path'))
    assert result.env.reward == 0 and result.reward == pytest.approx(.05)


def test_distinct_paths_with_equal_costs_both_count():
    actions = [dict(move_to=v) for v in ('a', 't', 'b', 't')]
    actions.append(dict(paths=[['s', 'a', 't'], ['s', 'b', 't']]))
    result = collect(Scripted(actions), fixture(), 2, 2, shaping=for_condition('path'))
    assert result.reward == pytest.approx(1.05)
    assert sum(e['path_bonus_earned'] > 0 for e in result.shaping['events']) == 2


@pytest.mark.parametrize('submission', [dict(paths=[[]]), dict(paths=[]), dict(paths=[['s', 'a', 't'], []])])
def test_incomplete_final_selection_cannot_cash_bonuses(submission):
    result = task(submission=submission)
    assert not result.shaping['eligible'] and result.reward == result.env.reward == 0
    assert result.shaping['edge_bonus'] == result.shaping['path_bonus'] == 0


def test_context_failure_revokes_previously_earned_bonuses():
    from ..policy import ContextOverflow
    class Overflow(Scripted):
        def action(self, messages, env):
            if env.phase == 'submission': raise ContextOverflow(500, 100)
            return super().action(messages, env)
    result = collect(Overflow([dict(move_to='a'), dict(move_to='t')]), fixture(), 1, 1,
                     shaping=for_condition('both'))
    assert result.status == 'context_limit' and result.reward == 0
    assert result.shaping['events'] and not result.shaping['eligible']
    assert result.shaping['edge_bonus'] == result.shaping['path_bonus'] == 0


@pytest.mark.parametrize('config', [{'edge_weight': -1}, {'path_weight': float('nan')}, {'edge_cap': -1}, {'path': 'true'}])
def test_invalid_shaping_options(config):
    with pytest.raises(ValueError): options(config)


def test_real_cpu_grpo_training_logs_separate_components(tmp_path, monkeypatch):
    import torch
    from pathlib import Path
    from .. import train
    from ..policy import ConstrainedPolicy
    from .helpers import tiny_model, ByteTokenizer
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
    config = dict(model='Synthetic/ShapingCPU', revision='0'*40, seed=17,
        context_limit=50000, temperature=1., dataset=str(dataset), output=str(output),
        curriculum=[1], steps=2, group_size=4, traversal_budget=1, objective='pareto',
        learning_rate=.001, beta=0., clip=.2, epochs_per_group=1, max_grad_norm=1.,
        early_stopping=False, shaping=for_condition('edge+path+hypervolume+cost_diversity+path_disjointness'))
    path = tmp_path/'config.json';path.write_text(json.dumps(config))
    train.main(path)
    tasks = [r for p in output.glob('rollout-*.json') for r in json.loads(p.read_text())]
    summary = json.loads((output/'training_summary.json').read_text())
    assert summary['tasks'] == len(tasks) == 8
    assert summary['total_reward'] == pytest.approx(sum(r['reward'] for r in tasks))
    for key in ('terminal_reward', 'edge_bonus', 'path_bonus'):
        assert summary['totals'][key] == pytest.approx(sum(r['shaping'][key] for r in tasks))
    assert summary['totals']['edge_bonus'] > 0 and summary['totals']['path_bonus'] > 0
    assert summary['totals']['hypervolume_bonus']>0
    assert 'cost_diversity_bonus' in summary['totals'] and 'path_disjointness_bonus' in summary['totals']
    audit=json.loads((output/'graph_sampling.json').read_text())
    assert audit['graphs'][0]['history']['groups']==2
    state=torch.load(output/'training_state.pt',weights_only=False)
    assert 'graph_sampler' in state
    assert not torch.equal(before, model.weight.detach())
    text = (output/'log.txt').read_text()
    assert 'Edge-information bonus total' in text and 'Terminal reward total (unshaped)' in text


def test_new_bonuses_add_without_discarding_equal_cost_paths():
    from ..shaping import SIGNALS
    actions=[dict(move_to=v) for v in ('a','t','b','t')]
    actions.append(dict(paths=[['s','a','t'],['s','b','t']]))
    result=collect(Scripted(actions),fixture(),2,2,shaping=for_condition('+'.join(SIGNALS)))
    shaped=result.shaping
    assert result.env.reward==1 and len(result.env.archive)==2
    assert shaped['cost_diversity_bonus']==0  # Equal costs have no extra spread.
    assert shaped['path_disjointness_bonus']==pytest.approx(.05)
    reference=(len(fixture().graph)-1)*fixture().max_cost+1
    assert shaped['hypervolume_bonus']==pytest.approx(.05*(1-2/reference)*(1-5/reference))
    bonuses=sum(shaped[name+'_bonus'] for name in SIGNALS)
    assert result.reward==pytest.approx(1+bonuses)
    assert shaped['events'][-1]['hypervolume_bonus_earned']==0
    assert shaped['events'][-1]['path_bonus_earned']>0


@pytest.mark.parametrize('signal',['hypervolume','cost_diversity','path_disjointness'])
def test_each_new_signal_is_independent_capped_and_failure_safe(signal):
    actions=[dict(move_to=v) for v in ('a','t','c','t','b','t','a','t')]
    actions.append(dict(paths=[['s','a','t']]))
    config=for_condition(signal,**{signal+'_weight':.03})
    result=collect(Scripted(actions),fixture(),1,4,shaping=config)
    assert 0<result.shaping[signal+'_bonus']<=.03
    assert result.reward==pytest.approx(result.env.reward+result.shaping[signal+'_bonus'])
    failed=collect(Scripted(actions[:-1]+[dict(paths=[[]])]),fixture(),1,4,shaping=config)
    assert failed.reward==failed.env.reward and failed.shaping[signal+'_bonus']==0


def test_all_32_factorial_combinations_and_cli_individual_toggles():
    from ..shaping import SIGNALS,experiment_conditions
    from .. import run_mosp_shaping as runner
    from .. import eval_shaping
    conditions=experiment_conditions('factorial')
    assert len(conditions)==len(set(conditions))==32
    for name in SIGNALS:
        assert sum(for_condition(c)[name] for c in conditions)==16
    args=runner.parser().parse_args(['--dry-run','--shaping-hypervolume','true',
        '--shaping-path','true','--shaping-cost-diversity','false'])
    output,settings,_=runner.resolve(args)
    _,training,evaluation=runner.configurations(output,settings)
    assert len(training)==1 and training[0][1]['shaping']['hypervolume']
    assert training[0][1]['shaping']['path'] and not training[0][1]['shaping']['cost_diversity']
    eval_shaping.validate_config(evaluation)
    args=runner.parser().parse_args(['--dry-run','--shaping','factorial',
        '--shaping-signals','edge','hypervolume','path_disjointness'])
    output,settings,_=runner.resolve(args)
    _,training,evaluation=runner.configurations(output,settings)
    assert len(training)==8 and len(evaluation['conditions'])==8
    eval_shaping.validate_config(evaluation)


def test_saved_individual_toggle_overrides_remain_stable(tmp_path):
    from .. import run_mosp_shaping as runner
    from ..run_artifacts import model_slug
    root=tmp_path/'full_runs/run_20260101120000'
    args=runner.parser().parse_args(['--dry-run','--output',str(root),
        '--shaping-signals','path','hypervolume','--shaping-path','false'])
    _,settings,_=runner.resolve(args)
    saved=root/f'{model_slug(settings["model"])}_trained/shaping/launch_config.json'
    saved.parent.mkdir(parents=True);saved.write_text(json.dumps(settings))
    later=runner.parser().parse_args(['--mode','eval','--dry-run','--output',str(root)])
    _,reloaded,existing=runner.resolve(later)
    assert existing and reloaded==settings and reloaded['shaping']=='hypervolume'
