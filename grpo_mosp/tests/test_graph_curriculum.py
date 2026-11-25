"""Graph curriculum schema, precedence, data isolation and stage scheduling."""
from copy import deepcopy
import json
from pathlib import Path
import random

import pytest

from .. import graph_curriculum as graphs
from .. import run_mosp_shaping as runner


def parameters():
    return {'subgraphs':[2,2,2],'nodes-per-subgraph':[[5,5],[6,6],[7,7]],'outdegree':[2,2,2]}


def config_file(tmp_path,values=None):
    path=tmp_path/'curriculum.json'
    path.write_text(json.dumps(parameters() if values is None else values))
    return path


@pytest.mark.parametrize('change',[
    lambda c:c.pop('outdegree'),
    lambda c:c.update(subgraphs=[2]),
    lambda c:c.update(outdegree=[]),
    lambda c:c.update(outdegree=None),
    lambda c:c.update(outdegree=[2,2,7]),
    lambda c:c.update(subgraphs=[2,True,3]),
    lambda c:c.update({'nodes-per-subgraph':[[5],[6,6],[7,7]]}),
    lambda c:c.update({'nodes-per-subgraph':[[5,5],[4,4],[7,7]]}),
])
def test_malformed_curriculum_raises_value_error(tmp_path,change):
    value=parameters();change(value)
    with pytest.raises(ValueError):graphs.read_config(config_file(tmp_path,value))


def test_tuple_notation_is_not_json(tmp_path):
    path=tmp_path/'invalid.json';path.write_text('{"nodes-per-subgraph": [(5,7)]}')
    with pytest.raises(ValueError,match='Cannot read graph curriculum'):graphs.read_config(path)


def test_cli_omission_default_and_parameter_precedence(tmp_path):
    p=runner.parser()
    args=p.parse_args(['--dry-run'])
    _,settings,_=runner.resolve(args)
    assert settings['graph_curriculum'] is None
    default=p.parse_args(['--dry-run','--curriculum-graph'])
    _,settings,_=runner.resolve(default)
    assert Path(settings['graph_curriculum']['source_file']).name=='graph_curriculum.json'
    path=config_file(tmp_path)
    args=p.parse_args(['--dry-run','--curriculum-graph',str(path),'--curriculum-k','1,2',
        '--subgraphs','1','--nodes-per-subgraph','0','1','--outdegree','100',
        '--objectives','0','--large-nodes-per-subgraph','-3','-1','--density-outdegree','100'])
    output,settings,_=runner.resolve(args)
    assert settings['subgraphs']==[2,2] and settings['nodes_per_subgraph']==[7,7]
    assert settings['outdegree']==[2,2] and settings['objectives']==3
    assert settings['large_nodes_per_subgraph']==[8,11] and settings['density_outdegree']==[3,3]
    assert settings['curriculum']==[1,2] and not output.exists()
    with pytest.raises(SystemExit):p.parse_args(['--curriculum','1'])


def test_every_stage_requires_a_graph_and_training_group(tmp_path):
    path=config_file(tmp_path)
    for flags in (['--train-graphs','2'],['--steps','2']):
        args=runner.parser().parse_args(['--dry-run','--curriculum-graph',str(path),*flags])
        with pytest.raises(ValueError):runner.resolve(args)


def test_real_stage_generation_shared_pools_rehearsal_and_saved_config(tmp_path):
    from ..dataset import load
    from ..eval_baselines import IsomorphismIndex
    source=config_file(tmp_path)
    output=tmp_path/'full_runs/run_20260101120000'
    args=runner.parser().parse_args(['--mode','prepare','--output',str(output),
        '--curriculum-graph',str(source),'--curriculum-k','1','--eval-ks','1',
        '--train-graphs','3','--validation-graphs','1','--eval-graphs','10',
        '--steps','6','--max-paths','128','--difficulty-cutoff','4'])
    runner.launch(args)
    root,settings,_=runner.resolve(args)
    _,training,evaluation=runner.configurations(root,settings)
    instances=load(output/'data/train.jsonl')
    assert len(instances)==3 and sorted(len(g.graph) for g in instances)==[10,12,14]
    validation=load(output/'data/validation.jsonl')
    assert len(validation)==1 and len(validation[0].graph)==14
    index=IsomorphismIndex()
    assert all(index.add_if_new(g) for g in instances+validation)
    assert len(list((output/'data').rglob('*.png')))==4
    memberships=[p['dataset'] for p in training[0][1]['graph_curriculum_datasets']]
    assert set(memberships)<=set(evaluation['excluded_datasets'])
    assert all(c['graph_curriculum_datasets']==training[0][1]['graph_curriculum_datasets'] for _,c in training)
    pools=graphs.TrainingPools(training[0][1],instances)
    sequence=[(item.seed,info) for step in range(6) for item,info in [pools.choose(step)]]
    other=graphs.TrainingPools(training[0][1],instances)
    random.seed(999)
    repeated=[]
    for step in range(6):
        random.random()  # Model/global sampling cannot perturb graph scheduling.
        item,info=other.choose(step);repeated.append((item.seed,info))
    assert sequence==repeated
    assert [info['graph_stage'] for _,info in sequence]==[0,0,1,1,2,2]
    assert all(sequence[step][1]['graph_pool_stage']==step//2 for step in (0,2,4))
    class AlwaysReplay:
        def random(self):return 0.
        def randrange(self,n):return 0
    pools.rng=AlwaysReplay()
    assert [pools.choose(step)[1]['graph_pool_stage'] for step in range(6)]==[0,0,1,0,2,0]
    # Later stages load their frozen settings/snapshot, not a moving external file.
    source.unlink()
    later=runner.parser().parse_args(['--mode','prepare','--output',str(output)])
    runner.launch(later)
    snapshot=output/f'{runner.model_slug(settings["model"])}_trained/shaping/configs/graph_curriculum.json'
    assert json.loads(snapshot.read_text())==parameters()
    # The held-out suite uses the final-stage distribution and excludes every
    # earlier training stage as well as its isomorphic relabelings.
    from .. import eval_shaping as light
    from ..dataset import Instance
    light.prepare(evaluation,Path(evaluation['output']))
    suite=[json.loads(line) for line in (Path(evaluation['output'])/'suite.jsonl').read_text().splitlines()]
    assert all(index.add_if_new(Instance.restore(row['instance'])) for row in suite)
    assert all(len(row['instance']['nodes'])==14 for row in suite if row['domain']=='in_distribution')
    # Preflight catches stage membership corruption before a model is loaded.
    Path(memberships[1]).write_bytes(Path(memberships[0]).read_bytes())
    with pytest.raises(ValueError,match='membership'):
        graphs.TrainingPools(training[0][1],instances)
    with pytest.raises(ValueError,match='training/validation data'):
        light.verify_suite(evaluation,Path(evaluation['output']))


def test_cpu_training_reaches_final_graph_stage_before_stopping(tmp_path,monkeypatch):
    import torch
    from .. import train
    from ..policy import ConstrainedPolicy
    from .helpers import fixture,tiny_model,ByteTokenizer
    model,tokenizer=tiny_model(),ByteTokenizer()
    def save(path,**kwargs):
        Path(path).mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),Path(path)/'tiny.pt')
    model.save_pretrained=save
    tokenizer.save_pretrained=lambda path:(Path(path)/'tokenizer.json').write_text('{}')
    monkeypatch.setattr(train,'load_policy',lambda _:ConstrainedPolicy(model,tokenizer,context_limit=50000))
    for name in ('reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'CPU test')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch,'use_deterministic_algorithms',lambda *a,**kw:None)
    # Tiny synthetic pools isolate scheduling; real-size generation is tested above.
    first=fixture();second=deepcopy(first)
    second.seed=1;second.graph['s']['z']['water']=4
    files=[]
    for index,item in enumerate((first,second)):
        path=tmp_path/f'stage-{index}.jsonl';path.write_text(json.dumps(item.record())+'\n')
        files.append(dict(stage=index,dataset=str(path)))
    dataset=tmp_path/'train.jsonl'
    dataset.write_text(''.join(json.dumps(item.record())+'\n' for item in (first,second)))
    value=parameters()
    for key in value:value[key]=value[key][:2]
    payload=graphs.read_config(config_file(tmp_path,value))
    validation=tmp_path/'validation.jsonl';validation.write_text('mocked validation')
    monkeypatch.setattr(train,'validation_instances',lambda *_:[first])
    monkeypatch.setattr(train,'validate',lambda *_:dict(mean_S=.5,tasks=[]))
    output=tmp_path/'result'
    config=dict(model='Synthetic/GraphCurriculumCPU',revision='0'*40,seed=17,
        context_limit=50000,temperature=1.,dataset=str(dataset),output=str(output),
        curriculum=[1],steps=6,group_size=2,traversal_budget=1,objective='pareto',
        learning_rate=.001,beta=0.,clip=.2,epochs_per_group=1,max_grad_norm=1.,
        early_stopping=True,validation_dataset=str(validation),
        early_stopping_interval=1,early_stopping_patience=1,
        graph_curriculum=payload,graph_curriculum_datasets=files)
    train.train(config)
    metrics=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    assert len(metrics)==4
    assert [r['graph_stage'] for r in metrics]==[0,0,0,1]
    assert metrics[-1]['graph_pool_stage']==1 and not metrics[-1]['rehearsal']
    stopping=json.loads((output/'early_stopping.json').read_text())
    assert stopping['stopped_early'] and stopping['selected_step']==0
    assert 'graph_rng' in torch.load(output/'training_state.pt',weights_only=True)
