"""Meaningful CPU checks for before/after isolation, pairing and uncertainty."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from .. import eval_baselines as harness
from ..dataset import Instance
from .helpers import fixture


def config(tmp_path):
    template = harness.BASELINE_PRESET_DIR / 'eval_baselines_pilot.json'
    c = json.loads(template.read_text())
    c.update(graphs=10, repetitions=2, relabelings=2, ks=[1], training_run_count=2,
        bootstrap_samples=8, context_limit=50000, max_task_tokens=10000,
        max_action_tokens=1024, device='cpu', suite_seed=12345,
        max_candidates_per_domain=10000, output=str(tmp_path/'experiment'))
    # Synthetic local checkpoint artifacts are used ONLY by an injected tiny
    # policy loader; no fake pretrained model or claimed learning result is made.
    model = tmp_path/'base'
    model.mkdir()
    (model/'weights.bin').write_bytes(b'synthetic test initialization')
    initial = dict(model=str(model), revision='0'*40)
    c['initial_checkpoint'] = initial
    training_data=tmp_path/'train.jsonl'
    harness.jsonl(training_data, [fixture().record()])
    validation_data=tmp_path/'validation.jsonl'
    harness.jsonl(validation_data, [harness.relabel(fixture(), 999).record()])
    c.update(excluded_datasets=[str(training_data)], validation_dataset=str(validation_data))
    runs=[]
    for i in range(2):
        adapter=tmp_path/f'adapter{i}'
        adapter.mkdir()
        (adapter/'weights.bin').write_bytes(f'synthetic adapter {i}'.encode())
        spec=dict(initial, adapter=str(adapter))
        tc=tmp_path/f'training{i}.json'
        harness.write_json(tc,dict(initial,seed=i,dataset=str(training_data)))
        evidence=tmp_path/f'selection{i}.json'
        harness.write_json(evidence,dict(selected_checkpoint_id=harness.checkpoint_id(spec)['checkpoint_id'],
            dataset_sha256=harness.file_hash(validation_data)))
        runs.append(dict(id=f'run{i}',post_checkpoint=spec,training_config=str(tc),
            training_compute=None,selection=dict(split='validation',dataset=str(validation_data),
                criterion='test-only synthetic selection record',evidence=str(evidence))))
    c['training_runs']=runs
    return c


def test_weighted_relabel_exclusion_and_checkpoint_identity(tmp_path):
    item=fixture()
    index=harness.IsomorphismIndex()
    assert index.add_if_new(item)
    assert not index.add_if_new(harness.relabel(item,45))
    changed=deepcopy(item)
    changed.graph['s']['a']['water'] += 1
    assert index.add_if_new(changed)
    c=config(tmp_path)
    copy=tmp_path/'copied'
    shutil.copytree(c['initial_checkpoint']['model'],copy)
    assert harness.checkpoint_id(dict(model=str(copy)))['checkpoint_id']==harness.checkpoint_id(c['initial_checkpoint'])['checkpoint_id']


def test_seeded_fixed_suite_has_all_shifts_and_no_training_copies(tmp_path):
    c=config(tmp_path)
    first=tmp_path/'first'
    second=tmp_path/'second'
    c['shared_suite'] = str(tmp_path/'shared-suite')
    a=harness.prepare(c,first)
    other = deepcopy(c)
    other['initial_checkpoint'] = dict(model='Synthetic/OtherQwen',revision='1'*40)
    other['output'] = str(second)
    b=harness.prepare(other,second)
    assert a['config_id'] != b['config_id']
    assert (first/'suite.jsonl').samefile(second/'suite.jsonl')
    assert (first/'transcripts.jsonl').samefile(second/'transcripts.jsonl')
    different = deepcopy(other)
    different['repetitions'] += 1
    with pytest.raises(ValueError,match='same held-out data design'):
        harness.prepare(different,tmp_path/'incompatible')
    assert a['suite_id']==b['suite_id'] and a['transcripts_id']==b['transcripts_id']
    rows=harness.read_jsonl(first/'suite.jsonl')
    assert len(rows)==20 and len({r['family_id'] for r in rows})==10
    assert len(a['graph_images']) == len(rows)
    assert all((first/r['image']).read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for r in rows)
    assert {(r['domain'],r['difficulty']) for r in rows}=={(d,s) for d in harness.DOMAINS for s in ('easy','hard')}
    assert len({r['graph_id'] for r in rows})==20
    index=harness.IsomorphismIndex()
    index.add_if_new(fixture())
    for row in rows[::2]: assert index.add_if_new(Instance.restore(row['instance']))
    assert all(max(a.values())<=11 for r in rows if r['domain']=='costs' for _,_,a in r['instance']['edges'])
    with pytest.raises(FileExistsError): harness.prepare(c,first)


def test_pairing_crossed_bootstrap_and_relabel_repetition_averaging():
    means={}
    metadata={}
    for i in range(8):
        family=str(i)
        metadata[family]=dict(domain='in_distribution',difficulty='easy' if i<4 else 'hard')
        baseline=np.full(len(harness.METRICS),i/10)
        means['initial',family]=baseline
        means['run0',family]=baseline+.2
        means['run1',family]=baseline+.4
    weights={('in_distribution','easy'):.5,('in_distribution','hard'):.5}
    point,low,high=harness.bootstrap_pair(means,metadata,['run0'],'initial',weights,100,42)
    assert np.allclose(point[2],.2) and np.allclose(low[2],.2) and np.allclose(high[2],.2)
    point,low,high=harness.bootstrap_pair(means,metadata,['run0','run1'],'initial',weights,100,42,True)
    assert np.allclose(point[2],.3) and np.all(low[2]<.3) and np.all(high[2]>.3)
    rows=[]
    for variant in range(2):
        for repetition in range(5):
            rows.append(dict(model_id='initial',family_id='f',k=1,variant=variant,
                             metrics={m:variant for m in harness.METRICS}))
    assert np.allclose(harness.family_means(rows,[1])['initial','f'],.5)


def test_limits_and_shared_archive_isolation(tmp_path):
    torch=pytest.importorskip('torch')
    from ..policy import ConstrainedPolicy
    from ..environment import Environment
    from ..memory import RULES
    from .helpers import tiny_model,ByteTokenizer
    c=config(tmp_path)
    policy=ConstrainedPolicy(tiny_model(),ByteTokenizer(),50000)
    env=Environment(fixture(),1,1)
    messages=[dict(role='system',content=RULES)]
    moves=[{'move_to':'a'},{'move_to':'t'}]
    for action in moves:
        messages.append(dict(role='user',content=env.observation()))
        messages.append(dict(role='assistant',content=json.dumps(action,separators=(',',':'))))
        env.step(action)
    messages.append(dict(role='user',content=env.observation()))
    transcript=dict(moves=moves,history_sha256=harness.digest(messages),
        final_observation_sha256=harness.digest(env.observation()),
        archive=[dict(path=list(p),cost=v) for p,v in sorted(env.archive.items())])
    before=deepcopy(transcript)
    a=harness.evaluate_task(policy,fixture(),1,1,10,c,transcript)
    b=harness.evaluate_task(policy,fixture(),1,1,12,c,transcript)
    assert transcript==before and a['archive']==b['archive']
    assert a['metrics']['A']==1 and a['metrics']['edge_traversals']==0
    assert a['metrics']['external_edge_traversals']==2
    assert all(not p or tuple(p) in env.archive for p in a['submission']['paths'])
    limited=dict(c,max_task_tokens=1)
    result=harness.evaluate_task(policy,fixture(),1,1,3,limited)
    assert result['status']=='token_limit' and result['metrics']['S']==0
    assert result['metrics']['generated_tokens']==1 and result['metrics']['limit_failure']==1
    assert result['metrics']['protocol_failure']==0
    short=ConstrainedPolicy(tiny_model(),ByteTokenizer(),1)
    result=harness.evaluate_task(short,fixture(),1,1,3,c)
    assert result['status']=='context_limit' and result['metrics']['S']==0


def test_validation_provenance_cannot_use_test_or_wrong_initialization(tmp_path):
    c=config(tmp_path)
    assert len(harness.validate_runs(c))==2
    c['training_runs'][0]['selection']['split']='test'
    with pytest.raises(ValueError,match='validation'): harness.validate_runs(c)
    c['training_runs'][0]['selection']['split']='validation'
    train=Path(c['training_runs'][0]['training_config'])
    record=json.loads(train.read_text())
    record['model']='wrong/model'
    record['revision']='1'*40
    harness.write_json(train,record)
    with pytest.raises(ValueError,match='initialization'): harness.validate_runs(c)


def test_real_evaluation_loop_and_deliverables(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from ..policy import ConstrainedPolicy
    from .helpers import tiny_model,ByteTokenizer
    c=config(tmp_path)
    c.update(repetitions=1,relabelings=1,bootstrap_samples=3,max_task_tokens=64)
    output=tmp_path/'evaluation'
    harness.prepare(c,output)
    policies=[]
    def loader(spec,config):
        model=tiny_model()
        policy=ConstrainedPolicy(model,ByteTokenizer(),config['context_limit'])
        policies.append(policy)
        return policy
    harness.run(c,output,loader)
    rows,suite=harness.checked_results(output,c)
    timings=json.loads((output/'inference_loading.json').read_text())
    assert all(set(r['evaluation_mode_wall_seconds'])=={'independent_exploration','shared_transcript_selection'} for r in timings)
    assert all(value>=0 for r in timings for value in r['evaluation_mode_wall_seconds'].values())
    assert len(rows)==10*3*1*5*1*2
    assert all(not p.model.training and not any(v.requires_grad for v in p.model.parameters()) for p in policies)
    assert all(r['metrics']['generated_tokens']<=64 for r in rows)
    table=harness.summarize(c,output,plots=True)
    assert (output/'comparison.csv').exists() and (output/'comparison.md').exists()
    from ..figures import figure_root
    assert len(list(figure_root(c,output).rglob('performance_vs_budget_initial.svg')))==1
    assert any(r['bootstrap']=='crossed_runs_graphs' for r in table)
    assert any(r['scope']=='distribution_shift_macro' for r in table)
    assert any(r['mode']=='shared_transcript_selection' for r in table)
    graph_sets = []
    for model_id in harness.checkpoint_specs(c):
        directory = harness.model_result_directory(c,output,model_id)
        model_rows = harness.read_jsonl(directory/'tasks.jsonl')
        assert model_rows and all(r['model_id'] == model_id for r in model_rows)
        assert (directory/'metrics.csv').exists() and (directory/'completed.json').exists()
        assert '_not_trained' in str(directory) if model_id == 'initial' else '_trained' in str(directory)
        graph_sets.append({r['graph_id'] for r in model_rows})
    assert all(ids == graph_sets[0] for ids in graph_sets)
    assert harness.pilot_plan(c,output)['plans']
    with pytest.raises(ValueError,match='separate pilot'):
        harness.pilot_plan(dict(c,phase='confirmatory'),output)
    # Partial reports must never silently compare different graph/repetition sets.
    with pytest.raises(ValueError,match='Incomplete'):
        harness.validate_results(rows[:-1],c,suite,harness.checkpoint_specs(c))
    with pytest.raises(FileExistsError): harness.run(c,output,loader)


def test_validation_selection_is_recorded_without_loading_test_suite(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from types import SimpleNamespace
    c = config(tmp_path)
    # A genuinely different validation graph, not a relabeled training copy.
    item = fixture()
    item.graph['s']['z']['water'] = 4
    harness.jsonl(c['validation_dataset'], [item.record()])
    monkeypatch.setattr(harness, 'load_frozen', lambda spec, _: SimpleNamespace(spec=spec))
    def validation_score(policy, instance, k, multiplier, seed, config):
        assert multiplier == 8
        assert instance.graph['s']['z']['water'] == 4
        return dict(metrics=dict(S=0.75 if policy.spec['adapter'].endswith('0') else 0.25))
    monkeypatch.setattr(harness, 'evaluate_task', validation_score)
    selected_path = tmp_path / 'selected.json'
    candidates = {'run0': [c['training_runs'][1]['post_checkpoint'], c['training_runs'][0]['post_checkpoint']]}
    selected = harness.select_validation(c, candidates, selected_path)
    assert selected['training_runs'][0]['post_checkpoint'] == c['training_runs'][0]['post_checkpoint']
    assert selected_path.exists()
    assert len(harness.validate_runs(selected)) == 2
    assert not Path(c['output']).exists()  # No held-out suite was loaded or scored.


def test_sft_requires_the_actual_immediate_pre_grpo_checkpoint(tmp_path):
    c = config(tmp_path)
    c['pre_grpo_checkpoint'] = c['training_runs'][0]['post_checkpoint']
    with pytest.raises(ValueError, match='initialization'):
        harness.validate_runs(c)
    for run in c['training_runs']:
        path = Path(run['training_config'])
        metadata = json.loads(path.read_text())
        metadata['initial_checkpoint'] = c['pre_grpo_checkpoint']
        harness.write_json(path, metadata)
    assert len(harness.validate_runs(c)) == 2
    assert 'pre_grpo' in harness.checkpoint_specs(c)
