import json
import pytest
from ..dataset import generate, build, load
from ..environment import Environment
from ..oracle import dominates, frontier, enumerate_paths, hypervolume
from ..scoring import score
from .helpers import fixture


def test_strict_dominance_and_equal_cost_paths():
    item = fixture()
    assert dominates((1,2),(1,3))
    assert not dominates((1,2),(1,2))
    assert not dominates((1,3),(2,2))
    assert frontier(item,enumerate_paths(item)) == item.pareto
    assert len(item.pareto)==2


def test_global_frontier_not_local_submission():
    item = fixture()
    dominated = ('s','c','t')
    assert frontier(item,[dominated]) == {dominated}
    assert score(item,{dominated:(5,6)},dict(paths=[list(dominated)]),1)['reward']==0


@pytest.mark.parametrize('submission,expected',[
    ({'paths':[['s','a','t'],['s','b','t']]},1),
    ({'paths':[['s','a','t'],['s','a','t']]},.5),
    ({'paths':[['s','a','t'],[]]},.5),
    ({'paths':[[],[]]},0), ({'paths':[]},0), ('garbage',0),
    ({'paths':[None,[]]},0), ({'paths':[['s','z','t'],[]]},0),
    ({'paths':[['s','a','t'],[]],'extra':1},0)])
def test_submission_validation(submission,expected):
    item=fixture()
    assert score(item,{p:None for p in item.pareto},submission,2)['reward']==expected


def test_untraversed_submission():
    assert score(fixture(),{},dict(paths=[['s','a','t']]),1)['reward']==0


def test_dead_end_reset_memory_and_no_leakage():
    env=Environment(fixture(),budget=2)
    initial=json.loads(env.observation())
    assert initial['known_costs']==[] and initial['archive']==[]
    assert set(initial['neighbors'])=={'s'}
    assert 'pareto' not in initial and 'graph' not in initial
    assert env.step({'move_to':'z'})==0
    assert env.current=='s' and env.attempts==1 and env.dead_nodes=={'z'}
    assert env.known[('s','z')]==[3,3]
    assert env.step({'move_to':'a'})==0
    assert env.cost==[1,4]
    assert env.step({'move_to':'t'})==0
    assert env.cost==[2,5] and env.phase=='submission'
    assert env.archive[('s','a','t')]==[2,5]
    assert len(env.known)==3 and ('s','b') not in env.known
    assert Environment(fixture()).known=={}
    assert env.step({'paths':[['s','a','t']]})==1
    assert env.phase=='terminal' and env.moves <= env.budget*(env.n-1)
    with pytest.raises(RuntimeError): env.step({})


@pytest.mark.parametrize('action',[{},'bad',{'move_to':'t'},{'move_to':'a','extra':1}])
def test_illegal_action_terminates(action):
    env=Environment(fixture())
    assert env.step(action)==0 and env.failure and env.phase=='terminal'


def test_hypervolume():
    assert hypervolume([], (5,5))==0
    assert hypervolume([(1,3),(3,1)], (5,5))==12
    assert hypervolume([(1,3),(1,3)], (5,5))==8
    assert hypervolume([(1,1,1)], (3,3,3))==8


def test_seeded_generation():
    # Rejection is itself reproducible. Locate a small acceptable benchmark.
    for seed in range(200):
        try: item=generate(seed,k=3)
        except ValueError: continue
        assert item.record()==generate(seed,k=3).record()
        assert len(item.names)==3
        assert set(item.names)=={'water','time','energy'}
        assert all(0<=data[name]<=5 for _,_,data in item.graph.edges(data=True) for name in item.names)
        assert all('transition_prob' not in data for _,_,data in item.graph.edges(data=True))
        break
    else: pytest.fail('No accepted graph')


def test_dataset_splits(tmp_path):
    build(tmp_path,count=1)
    groups=[load(tmp_path/f'{name}.jsonl') for name in ['train','validation','test','test_large','test_4d']]
    assert len({g[0].seed for g in groups})==5
    assert len(groups[-1][0].names)==4
    assert len(groups[-2][0].graph)>24


def test_pre_enumeration_limit():
    from ..generate_graph_world import GraphGenerator
    with pytest.raises(ValueError, match='path_limit'):
        GraphGenerator(n_subgraphs=1,n_nodes_per_subgraph=12,seed=0,max_paths=1)


def test_hidden_weights_do_not_change_observation():
    from copy import deepcopy
    first=fixture()
    second=deepcopy(first)
    second.graph['b']['t']['water']=999
    a,b=Environment(first),Environment(second)
    assert a.observation()==b.observation()
    for action in [{'move_to':'a'},{'move_to':'t'}]:
        a.step(action)
        b.step(action)
        assert a.observation()==b.observation()


def test_hypervolume_objective_is_explicit():
    item=fixture()
    archive={('s','c','t'):(5,6)}
    submission={'paths':[['s','c','t']]}
    assert score(item,archive,submission,1)['reward']==0
    result=score(item,archive,submission,1,'hypervolume')
    assert result['reward']==result['normalized_hypervolume']>0
