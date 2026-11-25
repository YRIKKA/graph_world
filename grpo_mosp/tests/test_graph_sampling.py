"""Coverage and adaptive sampling use independent RNGs and no held-out scores."""
import random
import networkx as nx
import pytest
from ..dataset import Instance
from ..graph_sampling import GraphSampler, features, sampling_options


def bank():
    result=[]
    for length in (2,5):
        for tradeoff in (False,True):
            graph=nx.DiGraph();paths=[]
            for route in range(2):
                path=['s']+[f'r{route}-{i}' for i in range(length-1)]+['t']
                for u,v in zip(path,path[1:]):
                    graph.add_edge(u,v,water=1+int(tradeoff and route==0)*3,
                                   time=1+int(tradeoff and route==1)*3)
                paths.append(tuple(path))
            graph.add_edge('s','dead',water=1,time=1)
            result.append(Instance(graph,'s','t',('water','time'),set(paths),len(result)))
    return result


def test_shuffled_coverage_and_independent_rng():
    items=bank();config=dict(shuffle_graphs=True)
    first=GraphSampler(items,config,17);second=GraphSampler(items,config,17)
    a=[];b=[]
    for _ in range(12):
        a.append(first.choose()[0].seed)
        random.seed(999);random.random()
        b.append(second.choose()[0].seed)
    assert a==b
    assert all(sorted(a[i:i+4])==list(range(4)) for i in range(0,12,4))
    assert first.state()['epoch']==2


def test_balanced_prefix_interleaves_measured_strata():
    items=bank()
    # Enlarge one stratum to check balance is a scheduling constraint, without
    # falsely claiming unequal bins become equal over a full coverage epoch.
    items += [bank()[0] for _ in range(5)]
    sampler=GraphSampler(items,dict(balance_graphs=True),101)
    assert len(set(sampler.bins))>=2
    chosen=[sampler.choose()[1]['sampling_stratum'] for _ in range(len(set(sampler.bins)))]
    assert len({tuple(b) for b in chosen})==len(set(sampler.bins))
    assert features(items[0])['frontier_vectors']==1
    assert features(items[1])['frontier_vectors']==2
    assert features(items[1])['tradeoff']>features(items[0])['tradeoff']


def test_variance_adaptation_downweights_but_never_excludes():
    items=bank()
    sampler=GraphSampler(items,dict(shuffle_graphs=False,adaptive_graph_sampling=True),17)
    for item in items[:3]:sampler.observe(item,[0,0,0,0],[0,0,0,0])
    sampler.observe(items[3],[0,1,0,1],[0,1,0,1])
    assert sampler.weight(0)==.25 and sampler.weight(3)==1
    selected=[sampler.choose()[0].seed for _ in range(3000)]
    assert set(selected)=={0,1,2,3} and selected.count(3)>2*selected.count(0)
    # EMA recovers after a graph begins producing useful group differences.
    feedback=sampler.observe(items[0],[0,1],[0,1])
    assert feedback['sampling_zero_variance_ema']==pytest.approx(.7)
    assert sampler.weight(0)>.25
    coverage=GraphSampler(items,dict(adaptive_graph_sampling=True),17)
    for item in items[:3]:coverage.observe(item,[0,0],[0,0])
    assert {coverage.choose()[0].seed for _ in range(4)}=={0,1,2,3}


def test_invalid_sampling_config():
    with pytest.raises(ValueError):sampling_options(dict(balance_graphs='true'))
