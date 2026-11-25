"""Seeded graph coverage, stratified ordering and reward-variance adaptation.

Sampling metadata may use the training oracle, but is never put in observations.
Coverage epochs visit every graph exactly once when shuffle_graphs is enabled.
Balancing interleaves strata while available; it cannot change full-epoch counts
without repeating graphs, so unequal strata retain their original epoch totals.
"""
import math
import random
import statistics
from collections import Counter

SAMPLING_DEFAULTS=dict(shuffle_graphs=True,balance_graphs=False,adaptive_graph_sampling=False)


def add_sampling_arguments(parser):
    import argparse
    from .model_options import boolean
    for key,value in SAMPLING_DEFAULTS.items():
        parser.add_argument('--'+key.replace('_','-'),type=boolean,nargs='?',const=True,
                            default=argparse.SUPPRESS,help=f'Training graph sampling option [default: {value}]')


def sampling_options(config):
    result={k:config.get(k,v) for k,v in SAMPLING_DEFAULTS.items()}
    if any(type(v) is not bool for v in result.values()):
        raise ValueError('Graph sampling options must be booleans')
    return result


def features(instance):
    """Cheap structural difficulty and frontier spread, not an oracle success rate."""
    import networkx as nx
    from .oracle import path_cost
    graph=instance.graph
    hops=nx.shortest_path_length(graph,instance.source,instance.target)
    reachable=nx.descendants(graph,instance.source)|{instance.source}
    dead=sum(graph.out_degree(v)==0 and v!=instance.target for v in reachable)/len(reachable)
    branching=sum(max(0,graph.out_degree(v)-1) for v in reachable)/len(reachable)
    vectors={path_cost(instance,p) for p in instance.pareto}
    bound=max(1,(len(graph)-1)*instance.max_cost)
    spread=sum(max(c[j] for c in vectors)-min(c[j] for c in vectors)
               for j in range(len(instance.names)))/(bound*len(instance.names))
    return dict(difficulty=hops*(1+branching)*(1+dead),
                tradeoff=math.log1p(len(vectors))*(1+spread),shortest_hops=hops,
                dead_end_fraction=dead,branching=branching,frontier_paths=len(instance.pareto),
                frontier_vectors=len(vectors),normalized_frontier_spread=spread)


class GraphSampler:
    def __init__(self,instances,config,seed):
        self.instances=instances;self.config=sampling_options(config)
        self.rng=random.Random(seed);self.remaining=[];self.epoch=-1
        self.features=[features(item) for item in instances] if self.config['balance_graphs'] else []
        if self.features:
            medians={key:statistics.median(f[key] for f in self.features) for key in ('difficulty','tradeoff')}
            self.bins=[tuple(int(f[key]>medians[key]) for key in ('difficulty','tradeoff')) for f in self.features]
        else:self.bins=[(0,0)]*len(instances)
        self.counts=Counter();self.history={};self.last=None
        self.indices={id(item):i for i,item in enumerate(instances)}

    def choose(self):
        if not self.remaining or not self.config['shuffle_graphs']:
            self.remaining=list(range(len(self.instances)))
            if self.config['shuffle_graphs'] or self.epoch<0:
                self.epoch+=1;self.counts.clear()
        candidates=self.remaining
        if self.config['balance_graphs']:
            available={self.bins[i] for i in candidates}
            minimum=min(self.counts[b] for b in available)
            candidates=[i for i in candidates if self.counts[self.bins[i]]==minimum]
        weights=[self.weight(i) for i in candidates]
        index=self.rng.choices(candidates,weights=weights,k=1)[0]
        self.last=index;self.counts[self.bins[index]]+=1
        if self.config['shuffle_graphs']:self.remaining.remove(index)
        return self.instances[index],dict(sampling_epoch=self.epoch,sampling_stratum=list(self.bins[index]),
            sampling_weight=self.weight(index),sampling_observations=self.history.get(index,{}).get('groups',0))

    def weight(self,index):
        # A floor prevents permanent exclusion. With coverage enabled adaptation
        # changes ordering only; with replacement it changes graph frequencies.
        if not self.config['adaptive_graph_sampling'] or index not in self.history:return 1.
        return .25+.75*(1-self.history[index]['zero_ema'])

    def observe(self,instance,rewards,terminal_scores):
        index=self.indices[id(instance)]
        zero=float(max(rewards)-min(rewards)<=1e-12)
        previous=self.history.get(index)
        self.history[index]=dict(groups=1+(previous['groups'] if previous else 0),
            zero_ema=.7*previous['zero_ema']+.3*zero if previous else zero,
            last_zero_variance=bool(zero),last_terminal_mean=sum(terminal_scores)/len(terminal_scores),
            last_terminal_zero_variance=max(terminal_scores)-min(terminal_scores)<=1e-12)
        return dict(sampling_zero_variance_ema=self.history[index]['zero_ema'],
                    sampling_next_weight=self.weight(index),**self.history[index])

    def state(self):
        return dict(rng=self.rng.getstate(),remaining=self.remaining,epoch=self.epoch,
                    counts=dict(self.counts),history=self.history)

    def audit(self):
        return dict(options=self.config,features=self.features,strata=self.bins,
            graphs=[dict(index=i,seed=item.seed,history=self.history.get(i,{})) for i,item in enumerate(self.instances)])
