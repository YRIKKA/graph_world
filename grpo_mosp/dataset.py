"""Seeded GraphGenerator adapter and evaluator-only JSONL datasets."""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import networkx as nx
from .generate_graph_world import GraphGenerator
from .oracle import enumerate_paths, frontier, path_cost


@dataclass
class Instance:
    graph: nx.DiGraph
    source: str
    target: str
    names: tuple
    pareto: set
    seed: int
    max_cost: int = 5

    def record(self):
        return dict(seed=self.seed, source=self.source, target=self.target, names=self.names,
                    nodes=list(self.graph), edges=[(u,v,dict(a)) for u,v,a in self.graph.edges(data=True)],
                    pareto=sorted(self.pareto), max_cost=self.max_cost)

    @classmethod
    def restore(cls, row):
        graph = nx.DiGraph()
        graph.add_nodes_from(row['nodes'])
        graph.add_edges_from(row['edges'])
        return cls(graph, row['source'], row['target'], tuple(row['names']),
                   {tuple(p) for p in row['pareto']}, row['seed'], row['max_cost'])


def generate(seed, k=1, larger=False, extra_dimension=False, limit=4096, outdegree=2, graph_config=None, pareto_solver="exhaustive"):
    # Keep the generator's endpoints, default named cost distribution and oracle.
    # Enabling costs is intentional; its edge_costs=False default omits weights.
    kwargs = dict(n_subgraphs=2, n_nodes_per_subgraph=(13,16) if larger else (6,12),
                  max_subgraph_outdegree=outdegree, edge_costs=True, mdp=False,
                  seed=seed, max_paths=limit, pareto_solver=pareto_solver)
    if graph_config is not None:
        # Sample the subgraph count independently from the generator's internal
        # RNG, so a whole graph specification remains a pure function of seed.
        spec = graph_config
        # Carry the same backend through fixed and curriculum graph specifications.
        kwargs['pareto_solver'] = spec.get('pareto_solver', pareto_solver)
        limits = spec.get('subgraphs', [2, 2])
        kwargs['n_subgraphs'] = random.Random(seed ^ 0x5A17).randint(*limits)
        kwargs['n_nodes_per_subgraph'] = tuple(spec.get(
            'large_nodes_per_subgraph' if larger else 'nodes_per_subgraph',
            [13, 16] if larger else [6, 12]))
        kwargs['max_subgraph_outdegree'] = tuple(spec.get('outdegree', [2, 2]))
        kwargs['max_paths'] = limit = spec.get('max_paths', limit)
        dimensions = spec.get('objectives', 3) + int(extra_dimension)
        names = ['water', 'time', 'energy', 'risk']
        names += [f'objective_{i}' for i in range(5, dimensions + 1)]
        kwargs.update(edge_cost_names=names[:dimensions], edge_cost_types=[int]*dimensions)
    elif extra_dimension:
        kwargs.update(edge_cost_names=['water','time','energy','risk'], edge_cost_types=[int]*4)
    world = GraphGenerator(**kwargs)
    if world.graph.out_degree(world.target):
        raise ValueError('target_not_sink')
    # Opaque IDs and independently shuffled insertion order conceal topology.
    rng = random.Random(seed)
    nodes = list(world.graph)
    labels = [f'n{i}' for i in range(len(nodes))]
    rng.shuffle(labels)
    mapping = dict(zip(nodes, labels))
    graph = nx.relabel_nodes(world.graph, mapping)
    oracle = {tuple(mapping[v] for v in path) for path in world.multiobjective_shortest_paths['paths']}
    instance = Instance(graph, mapping[world.source], mapping[world.target],
                        tuple(world.edge_cost_names), oracle, seed)
    paths = enumerate_paths(instance, limit)
    if frontier(instance, paths) != oracle:
        raise AssertionError('Generator oracle disagrees with independent enumeration')
    if len(oracle) < k:
        raise ValueError('insufficient_frontier')
    if len(oracle) == len(paths):
        raise ValueError('no_dominated_path')
    reachable = nx.descendants(graph, instance.source)
    if not any(v != instance.target and graph.out_degree(v) == 0 for v in reachable):
        raise ValueError('no_reachable_dead_end')
    assert not any('transition_prob' in a for _,_,a in graph.edges(data=True))
    return instance


def load(path):
    return [Instance.restore(json.loads(line)) for line in Path(path).read_text().splitlines() if line]


def build(output, count=32, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stats = {}
    from .graph_images import export_graph_image
    # Split seed residues are disjoint even if rejection sampling takes many draws.
    for offset, split in enumerate(['train','validation','test','test_large','test_4d']):
        rows, rejected, candidate = [], Counter(), seed * 5 + offset
        while len(rows) < count:
            if sum(rejected.values()) > 10000 * count:
                raise RuntimeError(f'Acceptance exhausted: {split}: {rejected}')
            try:
                item = generate(candidate, k=3, larger=split=='test_large', extra_dimension=split=='test_4d')
                export_graph_image(item, output/'images'/split/f'seed-{item.seed}.png')
                rows.append(item.record())
            except ValueError as error:
                rejected[str(error)] += 1
            candidate += 5
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        stats[split] = dict(rejections=dict(rejected), accepted=count,
                           graphs=[dict(seed=r['seed'], nodes=len(r['nodes']), edges=len(r['edges']),
                                        pareto_paths=len(r['pareto']),
                                        paths=len(enumerate_paths(Instance.restore(r)))) for r in rows])
    (output/'statistics.json').write_text(json.dumps(stats, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='jm_test/graph_world/grpo_mosp/data')
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build(args.output, args.count, args.seed)
