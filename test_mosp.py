"""Offline correctness checks for T-MDA and the exhaustive MOSP baseline."""

import unittest
from decimal import Decimal
from unittest.mock import patch

import networkx as nx
import numpy as np

if __package__:
    from .mosp import exhaustive_enumeration, targeted_multiobjective_dijkstra
else:
    from mosp import exhaustive_enumeration, targeted_multiobjective_dijkstra


class MospTests(unittest.TestCase):
    def assert_pareto_paths(self, graph, source, target, result, expected):
        # Check every route realizing a surviving vector, including equal-cost ties.
        self.assertEqual({cost for _, cost in result}, set(expected))
        routes = [[source]] if source == target else nx.all_simple_paths(graph, source, target)
        expected_paths = set()
        for path in routes if expected else []:
            cost = tuple(sum(graph[u][v]['cost'][i] for u, v in zip(path, path[1:]))
                         for i in range(len(next(iter(expected)))))
            if cost in expected:
                expected_paths.add(tuple(path))
        self.assertEqual({tuple(path) for path, _ in result}, expected_paths)
        self.assertEqual(len(result), len(expected_paths))
        self.assertEqual([cost for _, cost in result], sorted(cost for _, cost in result))
        for path, cost in result:
            # Returned paths must be simple, directed, and have their reported costs.
            self.assertEqual(path[0], source)
            self.assertEqual(path[-1], target)
            self.assertEqual(len(path), len(set(path)))
            actual = [0] * len(cost)
            for parent, child in zip(path, path[1:]):
                for index, value in enumerate(graph[parent][child]['cost']):
                    actual[index] += value
            self.assertEqual(tuple(actual), cost)

    def test_tradeoffs_equal_costs_and_dominated_paths(self):
        # Include three tradeoffs, one duplicate vector, and one dominated vector.
        graph = nx.DiGraph()
        for node, cost in enumerate(((1, 9), (4, 4), (9, 1), (4, 4), (8, 8))):
            graph.add_edge('s', node, cost=cost)
            graph.add_edge(node, 't', cost=(0, 0))
        for method in (targeted_multiobjective_dijkstra, exhaustive_enumeration):
            self.assert_pareto_paths(graph, 's', 't', method(graph, 's', 't'), ((1, 9), (4, 4), (9, 1)))

    def test_zero_cost_cycles_dead_ends_and_arbitrary_node_labels(self):
        # Mixed node types verify that priority queues never compare node names.
        source, a, b, target, dead = 's', (1, 0), 3, frozenset({1}), 'dead'
        graph = nx.DiGraph()
        for parent, child, cost in (
            (source, a, (0, 0)), (a, source, (0, 0)), (a, a, (0, 0)),
            (a, b, (0, 0)), (b, a, (0, 0)), (a, target, (2, 5)),
            (b, target, (5, 2)), (source, dead, (0, 0)),
        ):
            graph.add_edge(parent, child, cost=cost)
        original = graph.copy()
        for method in (targeted_multiobjective_dijkstra, exhaustive_enumeration):
            self.assert_pareto_paths(graph, source, target, method(graph, source, target), ((2, 5), (5, 2)))
        self.assertTrue(nx.utils.graphs_equal(graph, original))

    def test_named_attributes_scalar_costs_and_decimals(self):
        # Separate attributes use the same objective ordering as vector weights.
        graph = nx.DiGraph()
        graph.add_edge('s', 'a', time=1.5, distance=4.0)
        graph.add_edge('a', 't', time=1.5, distance=4.0)
        graph.add_edge('s', 't', time=4.0, distance=1.0)
        for method in (targeted_multiobjective_dijkstra, exhaustive_enumeration):
            results = method(graph, 's', 't', weight=('time', 'distance'))
            self.assertEqual({cost for _, cost in results}, {(3.0, 8.0), (4.0, 1.0)})
            self.assertEqual(method(graph, 's', 't', weight='time'), [(['s', 'a', 't'], (3.0,))])

    def test_unreachable_trivial_and_edgeless_graphs(self):
        # Unreachable queries yield no paths; a node reaches itself at zero cost.
        graph = nx.DiGraph()
        graph.add_nodes_from(['s', 't'])
        for method in (targeted_multiobjective_dijkstra, exhaustive_enumeration):
            self.assertEqual(method(graph, 's', 't'), [])
            self.assertEqual(method(graph, 's', 's'), [(['s'], (0,))])
            self.assertEqual(method(graph, 's', 's', weight=('time', 'distance')), [(['s'], (0, 0))])
        graph.add_edge('s', 't', cost=(0, 2, 0, 3))
        for method in (targeted_multiobjective_dijkstra, exhaustive_enumeration):
            self.assertEqual(method(graph, 's', 's'), [(['s'], (0, 0, 0, 0))])

    def test_invalid_graphs_and_costs(self):
        # Both methods reject unsupported graph types and malformed cost data.
        for method in (targeted_multiobjective_dijkstra, exhaustive_enumeration):
            for graph in (nx.Graph([(0, 1)]), nx.MultiDiGraph([(0, 1)])):
                with self.assertRaises(TypeError):
                    method(graph, 0, 1)
            with self.assertRaises(nx.NodeNotFound):
                method(nx.DiGraph(), 0, 1)
            for invalid in ((-1, 2), (np.nan, 2), (np.inf, 2), (), ((1, 2),), ('1', '2'), (True, False)):
                graph = nx.DiGraph()
                graph.add_edge(0, 1, cost=invalid)
                with self.subTest(method=method.__name__, invalid=invalid), self.assertRaises(ValueError):
                    method(graph, 0, 1)
            graph = nx.DiGraph()
            graph.add_edge(0, 1)
            with self.assertRaises(ValueError):
                method(graph, 0, 1)
            graph[0][1]['cost'] = (1, 2)
            graph.add_edge(1, 2, cost=(1, 2, 3))
            with self.assertRaises(ValueError):
                method(graph, 0, 2)
            with self.assertRaises(ValueError):
                method(graph, 0, 1, weight=())

    def test_reuses_supplied_paths_and_rejects_invalid_routes(self):
        # Reused paths must retain ties, avoid a new traversal, and stay unmodified.
        graph = nx.DiGraph()
        for node in ('a', 'b'):
            graph.add_edge('s', node, cost=(1, 2))
            graph.add_edge(node, 't', cost=(2, 1))
        paths = list(nx.all_simple_paths(graph, 's', 't'))
        original = [path[:] for path in paths]
        with patch.object(nx, 'all_simple_paths', side_effect=AssertionError('Unexpected traversal')):
            result = exhaustive_enumeration(graph, 's', 't', paths=iter(paths + paths))
            self.assertEqual({tuple(path) for path, _ in result}, {tuple(path) for path in paths})
            self.assertEqual(len(result), 2)
            self.assertEqual(exhaustive_enumeration(graph, 's', 't', paths=[]), [])
        self.assertEqual(paths, original)
        self.assertTrue(all(returned is not supplied for returned, _ in result for supplied in paths))
        # Reject wrong endpoints, repeated nodes, and nonexistent edges in cached paths.
        for invalid in ([], ['s', 'a'], ['a', 't'], ['s', 't'], ['s', 'a', 's', 'a', 't']):
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                exhaustive_enumeration(graph, 's', 't', paths=[invalid])

    def test_decimal_precision_retains_equivalent_routes(self):
        # Binary float addition must not discard a mathematically equivalent route.
        graph = nx.DiGraph()
        for node, first, second in (('a', 0.1, 0.2), ('b', 0.3, 0.0),
                                    ('c', 0.1 + 0.2, 0.0), ('d', 0.30001, 0.0)):
            graph.add_edge('s', node, cost=first)
            graph.add_edge(node, 't', cost=second)
        original = graph.copy()
        for method in (targeted_multiobjective_dijkstra, exhaustive_enumeration):
            result = method(graph, 's', 't')
            self.assertEqual({tuple(path) for path, _ in result},
                             {('s', node, 't') for node in ('a', 'b', 'c')})
            self.assertEqual({cost for _, cost in result}, {(0.3,)})
            # A coarser requested grid intentionally makes the fourth route tie too.
            result = method(graph, 's', 't', cost_precision=2)
            self.assertEqual(len(result), 4)
            self.assertEqual({cost for _, cost in result}, {(0.3,)})
        self.assertTrue(nx.utils.graphs_equal(graph, original))

    def test_rounding_boundaries_large_integers_and_invalid_precision(self):
        # Decimal half-way values round to even; mixed vectors preserve large integers.
        graph = nx.DiGraph()
        large = 2 ** 60 + 1
        graph.add_edge('s', 't', cost=(large, 1.005, 1.015))
        for method in (targeted_multiobjective_dijkstra, exhaustive_enumeration):
            self.assertEqual(method(graph, 's', 't', cost_precision=2),
                             [(['s', 't'], (large, 1, 1.02))])
            for invalid in (-1, True, 1.5, '2', None):
                with self.subTest(method=method.__name__, precision=invalid), self.assertRaises(ValueError):
                    method(graph, 's', 't', cost_precision=invalid)

    def test_float_graphs_against_independent_decimal_oracle(self):
        # Use Decimal quantization independently of the solvers' integer conversion.
        rng = np.random.default_rng(438)
        quantum = Decimal('0.01')
        for replicate in range(60):
            graph = nx.DiGraph()
            graph.add_nodes_from(range(6))
            for parent in graph:
                for child in graph:
                    if parent != child and rng.random() < 0.35:
                        graph.add_edge(parent, child, cost=tuple(rng.integers(0, 100, 2) / 1000))
            route_costs = []
            for path in nx.all_simple_paths(graph, 0, 5):
                cost = tuple(sum((Decimal(str(graph[u][v]['cost'][i])).quantize(quantum)
                                  for u, v in zip(path, path[1:])), Decimal(0)) for i in range(2))
                route_costs.append((tuple(path), cost))
            expected = {(path, tuple(float(value) for value in cost)) for path, cost in route_costs
                        if not any(other != cost and all(a <= b for a, b in zip(other, cost))
                                   for _, other in route_costs)}
            for method in (targeted_multiobjective_dijkstra, exhaustive_enumeration):
                with self.subTest(replicate=replicate, method=method.__name__):
                    result = method(graph, 0, 5, cost_precision=2)
                    self.assertEqual({(tuple(path), cost) for path, cost in result}, expected)
                    self.assertEqual(len(result), len(expected))

    def test_random_graphs_against_independent_exhaustive_oracle(self):
        # Enumerate independently to catch errors shared by the two public methods.
        rng = np.random.default_rng(12345)
        for dimension in (1, 2, 3, 4):
            for replicate in range(100):
                graph = nx.DiGraph()
                graph.add_nodes_from(range(6))
                for parent in graph:
                    for child in graph:
                        if parent != child and rng.random() < 0.3:
                            graph.add_edge(parent, child, cost=tuple(rng.integers(0, 6, dimension).tolist()))
                vectors = set()
                for path in nx.all_simple_paths(graph, 0, 5):
                    # Compute costs without relying on mosp.py's helpers.
                    vectors.add(tuple(sum(graph[u][v]['cost'][i] for u, v in zip(path, path[1:]))
                                      for i in range(dimension)))
                expected = {cost for cost in vectors if not any(
                    other != cost and all(a <= b for a, b in zip(other, cost)) for other in vectors
                )}
                with self.subTest(dimension=dimension, replicate=replicate):
                    for method in (targeted_multiobjective_dijkstra, exhaustive_enumeration):
                        self.assert_pareto_paths(graph, 0, 5, method(graph, 0, 5), expected)


if __name__ == '__main__':
    # Run these checks directly or through unittest discovery; no API calls occur.
    unittest.main()
