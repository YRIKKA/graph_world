"""Package-backed selection preserves feasible routes, ties and exact costs."""
import ast
from decimal import Decimal
from functools import wraps
from pathlib import Path
from importlib import import_module

import networkx as nx
import numpy as np
import pytest
import paretoset as package

from .. import mosp as demo_solver


@pytest.fixture(params=['jm_test.graph_world', 'jm_test.graph_world.grpo_mosp'])
def modules(request):
    # Check both public module locations used by the standalone world and GRPO.
    return (import_module(request.param + '.mosp'),
            import_module(request.param + '.generate_graph_world'))


def test_package_filter_retains_ties_and_avoids_legacy_solvers(modules, monkeypatch):
    solver, _ = modules

    def unused(*args, **kwargs):
        pytest.fail('The active solver must not invoke a legacy solver')

    monkeypatch.setattr(solver, 'exhaustive_enumeration', unused)
    monkeypatch.setattr(solver, 'targeted_multiobjective_dijkstra', unused)
    graph = nx.DiGraph()
    for node, cost in [('a', (2, 5)), ('b', (2, 5)), ('c', (5, 2)), ('d', (6, 6))]:
        graph.add_edge('s', node, cost=cost)
        graph.add_edge(node, 't', cost=(0, 0))
    paths = list(nx.all_simple_paths(graph, 's', 't'))
    original = graph.copy()
    actual = package.paretoset
    calls = []

    def track(costs, **kwargs):
        calls.append((costs.copy(), kwargs))
        return actual(costs, **kwargs)

    monkeypatch.setattr(solver, 'paretoset', track)
    # Cached paths must be reused, and duplicate copies of one route collapse.
    monkeypatch.setattr(nx, 'all_simple_paths', unused)
    results = solver.paretoset_paths(graph, 's', 't', paths=paths + paths)
    assert results == [(['s', 'a', 't'], (2, 5)), (['s', 'b', 't'], (2, 5)),
                       (['s', 'c', 't'], (5, 2))]
    assert len(calls) == 1
    assert calls[0][1] == dict(sense=['min', 'min'], distinct=False, use_numba=False)
    assert nx.utils.graphs_equal(graph, original)


def test_large_integer_precision_and_named_costs(modules):
    solver, _ = modules
    large = 10 ** 100
    graph = nx.DiGraph()
    for node, cost in [('a', (large, 1)), ('b', (large + 1, 1)), ('c', (large + 2, 0))]:
        graph.add_edge('s', node, time=cost[0], risk=cost[1])
        graph.add_edge(node, 't', time=0, risk=0)
    results = solver.paretoset_paths(graph, 's', 't', weight=('time', 'risk'))
    assert results == [(['s', 'a', 't'], (large, 1)), (['s', 'c', 't'], (large + 2, 0))]


def test_zero_length_unreachable_and_invalid_inputs(modules):
    solver, _ = modules
    graph = nx.DiGraph()
    graph.add_nodes_from(['s', 't'])
    assert solver.paretoset_paths(graph, 's', 't') == []
    assert solver.paretoset_paths(graph, 's', 's') == [(['s'], (0,))]
    graph.add_edge('s', 't', cost=(1, 2))
    assert solver.paretoset_paths(graph, 's', 't', paths=[]) == []
    for path in [[], ['s'], ['t'], ['s', 't', 's', 't']]:
        with pytest.raises(ValueError, match='simple directed'):
            solver.paretoset_paths(graph, 's', 't', paths=[path])
    graph['s']['t']['cost'] = (-1, 2)
    with pytest.raises(ValueError):
        solver.paretoset_paths(graph, 's', 't')


def test_randomized_decimal_costs_against_independent_oracle(modules):
    solver, _ = modules
    rng = np.random.default_rng(437)
    for dimension in (1, 2, 3, 4):
        for _ in range(25):
            graph = nx.DiGraph()
            graph.add_nodes_from(range(6))
            for u in graph:
                for v in graph:
                    if u != v and rng.random() < .3:
                        graph.add_edge(u, v, cost=tuple(rng.integers(0, 100, dimension) / 1000))
            paths = list(nx.all_simple_paths(graph, 0, 5))
            costs = [(tuple(path), tuple(sum((Decimal(str(graph[u][v]['cost'][i])).quantize(Decimal('.01'))
                      for u, v in zip(path, path[1:])), Decimal(0)) for i in range(dimension)))
                     for path in paths]
            # Independent strict dominance catches regressions shared by local helpers.
            expected = {(path, tuple(float(v) for v in cost)) for path, cost in costs
                        if not any(other != cost and all(a <= b for a, b in zip(other, cost))
                                   for _, other in costs)}
            result = solver.paretoset_paths(graph, 0, 5, cost_precision=2)
            assert {(tuple(path), cost) for path, cost in result} == expected


def test_generator_uses_package_without_legacy_methods(modules, monkeypatch):
    solver, generator_module = modules

    def unused(*args, **kwargs):
        pytest.fail('GraphGenerator must not use either legacy Pareto solver')

    monkeypatch.setattr(solver, 'exhaustive_enumeration', unused)
    monkeypatch.setattr(solver, 'targeted_multiobjective_dijkstra', unused)
    graph = nx.DiGraph()
    for node, costs in [('a', (1, 3)), ('b', (1, 3)), ('c', (3, 1)), ('d', (4, 4))]:
        graph.add_edge('s', node, time=costs[0], risk=costs[1])
        graph.add_edge(node, 't', time=0, risk=0)
    monkeypatch.setattr(generator_module.GraphGenerator, 'make_graph', lambda *args, **kwargs: graph)
    generated = generator_module.GraphGenerator(1, 5, 2, restriction=(0., 0., 0.),
        mdp=False, edge_cost_names=['time', 'risk'], edge_cost_types=[int, int],
        pareto_solver='paretoset')
    assert generated.multiobjective_shortest_paths == dict(
        paths=[['s', 'a', 't'], ['s', 'b', 't'], ['s', 'c', 't']],
        costs=[dict(time=1, risk=3, scalar_sum=4), dict(time=1, risk=3, scalar_sum=4),
               dict(time=3, risk=1, scalar_sum=4)])


def test_demo_compares_same_graph_and_prints_labeled_path_costs(monkeypatch, capsys):
    solver = demo_solver

    graph = nx.DiGraph()
    for node, cost in [('a', (1, 2, 3)), ('b', (1, 2, 3)), ('c', (4, 4, 4))]:
        graph.add_edge('s', node, cost=cost)
        graph.add_edge(node, 't', cost=(0, 0, 0))
    monkeypatch.setattr(solver, '_random_test_graph', lambda rng: graph)
    calls = []

    def track(method):
        @wraps(method)
        def wrapped(candidate, source, target):
            assert candidate is graph and (source, target) == ('s', 't')
            calls.append(method.__name__)
            return method(candidate, source, target)
        return wrapped

    names = ('targeted_multiobjective_dijkstra', 'exhaustive_enumeration', 'paretoset_paths')
    for name in names:
        monkeypatch.setattr(solver, name, track(getattr(solver, name)))
    solver.main(n_replicates=2)
    output = capsys.readouterr().out
    assert calls == list(names) * 2
    for name in names:
        assert '[' + name + ']' in output
    assert output.count("Path: ['s', 'a', 't'] | Cost: (1, 2, 3)") == 3
    assert output.count("Path: ['s', 'b', 't'] | Cost: (1, 2, 3)") == 3
    assert 'All three methods agree: True' in output
    assert 'Agreement across random graphs: 2/2' in output


def test_solver_imports_are_all_at_module_scope(modules):
    solver, _ = modules
    tree = ast.parse(Path(solver.__file__).read_text())
    top_level_imports = {id(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))}
    all_imports = {id(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))}
    assert all_imports == top_level_imports


@pytest.mark.parametrize('seed', [0, 1, 42])
def test_random_demo_graph_has_twenty_nodes_and_outdegree_three(modules, seed):
    solver, _ = modules
    graph = solver._random_test_graph(np.random.default_rng(seed))
    assert len(graph) == 20 and graph.number_of_edges() == 60
    assert all(degree == 3 for _, degree in graph.out_degree())
    assert nx.number_of_selfloops(graph) == 0
    assert nx.has_path(graph, 's', 't')
    for _, _, data in graph.edges(data=True):
        assert len(data['cost']) == 3
        assert all(isinstance(value, int) and 1 <= value <= 10 for value in data['cost'])


@pytest.mark.parametrize('selection', [None, 'dijkstra', 'exhaustive', 'paretoset'])
def test_generator_dispatches_only_to_selected_solver(modules, monkeypatch, selection):
    solver, generator_module = modules
    selected = selection or 'exhaustive'
    names = dict(dijkstra='targeted_multiobjective_dijkstra', exhaustive='exhaustive_enumeration',
                 paretoset='paretoset_paths')
    original = getattr(solver, names[selected])
    calls = []

    def track(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    def unused(*args, **kwargs):
        pytest.fail('A non-selected Pareto solver was invoked')

    for key, name in names.items():
        monkeypatch.setattr(solver, name, track if key == selected else unused)
    graph = nx.DiGraph()
    for node, costs in [('a', (1, 3)), ('b', (1, 3)), ('c', (3, 1)), ('d', (4, 4))]:
        graph.add_edge('s', node, time=costs[0], risk=costs[1])
        graph.add_edge(node, 't', time=0, risk=0)
    monkeypatch.setattr(generator_module.GraphGenerator, 'make_graph', lambda *a, **kw: graph)
    kwargs = {} if selection is None else dict(pareto_solver=selection)
    generated = generator_module.GraphGenerator(1, 5, 2, restriction=(0., 0., 0.), mdp=False,
        edge_cost_names=['time', 'risk'], edge_cost_types=[int, int], **kwargs)
    assert generated.pareto_solver == selected
    # Equal-cost routes may be returned in different orders by different solvers.
    assert {tuple(path) for path in generated.multiobjective_shortest_paths['paths']} == {
        ('s', 'a', 't'), ('s', 'b', 't'), ('s', 'c', 't')}
    assert len(generated.multiobjective_shortest_paths['paths']) == 3
    assert len(calls) == 1
    assert ('paths' in calls[0]) is (selected != 'dijkstra')


def test_generator_rejects_unknown_solver_before_graph_construction(modules, monkeypatch):
    _, generator_module = modules

    def unused(*args, **kwargs):
        pytest.fail('Invalid selection must fail before graph construction')

    monkeypatch.setattr(generator_module.GraphGenerator, 'make_graph', unused)
    with pytest.raises(ValueError, match='pareto_solver'):
        generator_module.GraphGenerator(pareto_solver='unknown')
