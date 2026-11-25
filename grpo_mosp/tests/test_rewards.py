"""Validity rewards include missing and duplicate slots in their denominator."""

import networkx as nx
import numpy as np
import pytest

from ..rewards import Reward


@pytest.fixture
def graph():
    graph = nx.DiGraph([(0, 1), (1, 3), (0, 2), (2, 3), (0, 3)])
    nx.set_edge_attributes(graph, (1, 1), 'cost')
    return graph


@pytest.mark.parametrize('paths,expected', [
    ([], 0),
    ([[0, 1, 3]], 1 / 3),
    ([[0, 1, 3], [0, 1, 3], [0, 2, 3]], 2 / 3),
    ([[0, 1, 3], [0, 2, 3], [0, 3]], 1),
    ([[0, 1, 3], [0, 1, 2, 3], []], 1 / 3),
    ({(0, 1, 3), (0, 2, 3)}, 2 / 3),
])
def test_valid_fraction(graph, paths, expected):
    assert Reward(graph, 0, 3, 3, paths, {}).reward_valid() == pytest.approx(expected)


@pytest.mark.parametrize('path', [[], [0], [1, 3], [0, 2], [0, 9, 3],
                                  [0, 3, 1, 3], [0, 1, 2, 3], None, [[0], 3]])
def test_invalid_paths(graph, path):
    assert Reward(graph, 0, 3, 3, [], {}).is_valid_path(path) is False


def test_single_node_path():
    graph = nx.DiGraph()
    graph.add_node('s')
    reward = Reward(graph, 's', 's', 1, [['s']], {})
    assert reward.is_valid_path(['s']) is True
    assert reward.reward_valid() == 1
    assert not reward.is_valid_path(['missing'])


def test_input_constraints(graph):
    with pytest.raises(ValueError, match='cannot be longer than length k'):
        Reward(graph, 0, 3, 3, [[0, 3]] * 4, {})
    for source, target in [(9, 3), (0, 9)]:
        with pytest.raises(ValueError, match='source and target'):
            Reward(graph, source, target, 3, [], {})
    for k in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match='positive integer'):
            Reward(graph, 0, 3, k, [], {})
    assert Reward(graph, 0, 3, 2, [[0, 3]], {}).reward_valid() == .5


@pytest.mark.parametrize('graph', [nx.DiGraph([(0, 1), (1, 0)]),
                                   nx.DiGraph([(0, 0)])])
def test_cycles_rejected(graph):
    with pytest.raises(ValueError, match='acyclic'):
        Reward(graph, 0, 0, 3, [], {})


@pytest.mark.parametrize('graph', [nx.Graph([(0, 1)]), np.zeros((2, 2))])
def test_non_digraph_rejected(graph):
    with pytest.raises(TypeError, match='directed graph'):
        Reward(graph, 0, 1, 3, [], {})


def test_stored_attributes_and_adjacency_matrix():
    graph = nx.DiGraph([('s', 'a', {'weight': 2, 'cost': (2, 1)}), ('a', 't', {'weight': 3, 'cost': (3, .5)})])
    paths = [('s', 'a', 't')]
    frontier = {('s', 'a', 't'): (5, 1.5), ('s', 't'): [2, 3.5]}
    reward = Reward(graph=graph, source='s', target='t', k=3,
                    paths=paths, pareto_frontier=frontier)
    assert reward.graph is graph
    assert reward.source == 's' and reward.target == 't' and reward.k == 3
    assert reward.paths is paths
    assert reward.pareto_frontier is frontier
    assert isinstance(reward.adjacency_matrix, np.ndarray)
    np.testing.assert_array_equal(reward.adjacency_matrix,
                                  [[0, 2, 0], [0, 0, 3], [0, 0, 0]])
    assert reward.reward_valid() == pytest.approx(1 / 3)


def test_terminal_reward_empty_selection(graph):
    assert Reward(graph, 0, 3, 3, [], {}).terminal_reward_additive([]) == 0.0


@pytest.mark.parametrize('path', [[], [0, 1, 2, 3], None, [[0], 3]])
def test_paths_classified_once_with_duplicates_preserved(graph, monkeypatch, path):
    paths = [[0, 1, 3], path, [0, 1, 3], [0, 2, 3]]
    original = Reward.is_valid_path
    calls = []

    def track_validation(self, candidate):
        calls.append(candidate)
        return original(self, candidate)

    monkeypatch.setattr(Reward, 'is_valid_path', track_validation)
    reward = Reward(graph, 0, 3, 5, paths, {})
    assert calls == paths
    assert reward.valid_paths == [[0, 1, 3], [0, 1, 3], [0, 2, 3]]
    assert reward.invalid_paths == [path]

    def unexpected_validation(self, candidate):
        pytest.fail('reward_valid must use the constructor classification')

    monkeypatch.setattr(Reward, 'is_valid_path', unexpected_validation)
    assert reward.reward_valid() == pytest.approx(2 / 5)
    assert reward.reward_valid() == pytest.approx(2 / 5)


@pytest.fixture
def hypervolume_reward():
    graph = nx.DiGraph()
    graph.add_edge('s', 'a', cost=(1, 2))
    graph.add_edge('a', 't', cost=[1, 3])
    graph.add_edge('s', 'b', cost=(3, 1))
    graph.add_edge('b', 't', cost=[2, 1])
    graph.add_edge('s', 't', cost=(6, 6))
    frontier = {('s', 'a', 't'): [2, 5], ('s', 'b', 't'): (5, 2)}
    # One Pareto path, its duplicate, a dominated path, and an invalid path.
    return Reward(graph, 's', 't', 4,
                  [['s', 'a', 't'], ['s', 'a', 't'], ['s', 't'], ['s', 'missing', 't']],
                  frontier)


@pytest.fixture(params=['pymoo', 'moocore'])
def backend(request):
    return request.param


def test_hypervolume_raw_and_normalized(hypervolume_reward, backend):
    # A covers 8*5=40; the full front covers 40+5*8-5*5=55.
    assert hypervolume_reward.reward_hypervolume([10, 10], normalize=False, backend=backend) == 40
    assert hypervolume_reward.reward_hypervolume([10, 10], backend=backend) == pytest.approx(40 / 55)


def test_full_front_hypervolume(hypervolume_reward, backend):
    reward = hypervolume_reward
    reward = Reward(reward.graph, 's', 't', 3,
                    list(reward.pareto_frontier), reward.pareto_frontier)
    assert reward.reward_hypervolume([10, 10], backend=backend) == 1


def test_empty_valid_hypervolume(hypervolume_reward, backend):
    reward = hypervolume_reward
    reward = Reward(reward.graph, 's', 't', 3, [], reward.pareto_frontier)
    assert reward.reward_hypervolume([10, 10], normalize=False, backend=backend) == 0
    assert reward.reward_hypervolume([10, 10], backend=backend) == 0


def test_hypervolume_zero_length_path(backend):
    graph = nx.DiGraph()
    graph.add_node('s')
    reward = Reward(graph, 's', 's', 1, [['s']], {('s',): (0, 0)})
    assert reward.reward_hypervolume([2, 3], normalize=False, backend=backend) == 6
    assert reward.reward_hypervolume([2, 3], backend=backend) == 1


def test_hypervolume_requires_nonzero_front(hypervolume_reward, backend):
    reward = hypervolume_reward
    with pytest.raises(ValueError, match='zero Pareto-front'):
        reward.reward_hypervolume([1, 1], backend=backend)
    reward.pareto_frontier = {}
    assert reward.reward_hypervolume([10, 10], normalize=False, backend=backend) == 40
    with pytest.raises(ValueError, match='pareto_frontier'):
        reward.reward_hypervolume([10, 10], backend=backend)


@pytest.mark.parametrize('reference', [[], [[10, 10]], [float('inf'), 10]])
def test_invalid_hypervolume_reference(hypervolume_reward, reference, backend):
    with pytest.raises(ValueError, match='reference_point'):
        hypervolume_reward.reward_hypervolume(reference, backend=backend)


@pytest.mark.parametrize('cost', [(1,), (1, float('nan'))])
def test_invalid_edge_cost_vector(hypervolume_reward, cost):
    hypervolume_reward.graph['s']['a']['cost'] = cost
    with pytest.raises(ValueError, match='Edge cost vectors'):
        Reward(hypervolume_reward.graph, 's', 't', 4, hypervolume_reward.paths,
               hypervolume_reward.pareto_frontier)


def test_missing_edge_cost(hypervolume_reward):
    del hypervolume_reward.graph['s']['a']['cost']
    with pytest.raises(ValueError, match='missing its cost vector'):
        Reward(hypervolume_reward.graph, 's', 't', 4, hypervolume_reward.paths,
               hypervolume_reward.pareto_frontier)


@pytest.mark.parametrize('cost', [(1,), (1, float('nan'))])
def test_invalid_frontier_cost_vector(hypervolume_reward, cost, backend):
    hypervolume_reward.pareto_frontier = {('s', 'a', 't'): cost}
    with pytest.raises(ValueError, match='pareto_frontier'):
        hypervolume_reward.reward_hypervolume([10, 10], backend=backend)


@pytest.mark.parametrize('backend', ['unknown', '', None])
def test_unknown_hypervolume_backend(hypervolume_reward, backend):
    with pytest.raises(ValueError, match='backend must be'):
        hypervolume_reward.reward_hypervolume([10, 10], backend=backend)


def test_default_hypervolume_backend_is_pymoo(hypervolume_reward, monkeypatch):
    from pymoo.indicators.hv import HV

    calls = []
    original = HV._do

    def track(self, points):
        calls.append(points)
        return original(self, points)

    monkeypatch.setattr(HV, '_do', track)
    assert hypervolume_reward.reward_hypervolume([10, 10]) == pytest.approx(40 / 55)
    assert len(calls) == 2


@pytest.mark.parametrize('score', [0.0, 0.5, 1.0])
def test_doctrine_receives_valid_paths(graph, score):
    received = []

    def doctrine(paths):
        received.append(paths)
        return score

    reward = Reward(graph, 0, 3, 3, [[0, 1, 3], [0, 1, 3], [0, 9, 3]], {},
                    doctrine=doctrine)
    assert reward.doctrine is doctrine
    assert reward.reward_doctrine() == score
    assert received[0] is reward.valid_paths
    assert received[0] == [[0, 1, 3], [0, 1, 3]]


def test_no_doctrine_returns_zero(graph):
    reward = Reward(graph, 0, 3, 3, [], {})
    assert reward.doctrine is None
    assert reward.reward_doctrine() == 0.0


def test_doctrine_returns_zero_for_empty_valid_paths(graph):
    reward = Reward(graph, 0, 3, 3, [[0, 9, 3]], {},
                    doctrine=lambda paths: 1.0 if paths == [] else 0.0)
    assert reward.reward_doctrine() == 0.0


@pytest.mark.parametrize('score', [-0.1, 1.1, float('nan'), float('inf'), None, '0.5'])
def test_invalid_doctrine_score(graph, score):
    reward = Reward(graph, 0, 3, 3, [[0, 3]], {}, doctrine=lambda paths: score)
    with pytest.raises(ValueError, match='between 0 and 1'):
        reward.reward_doctrine()


def test_noncallable_doctrine_rejected(graph):
    with pytest.raises(TypeError, match='callable or None'):
        Reward(graph, 0, 3, 3, [], {}, doctrine=0.5)


def test_cached_valid_costs(hypervolume_reward, backend):
    reward = hypervolume_reward
    np.testing.assert_array_equal(reward.valid_costs, [[2, 5], [2, 5], [6, 6]])
    assert len(reward.valid_costs) == len(reward.valid_paths)
    # Hypervolume must use the cached vectors without consulting edge costs again.
    for _, _, data in reward.graph.edges(data=True):
        del data['cost']
    assert reward.reward_hypervolume([10, 10], normalize=False, backend=backend) == 40
    assert reward.reward_hypervolume([10, 10], backend=backend) == pytest.approx(40 / 55)


def test_only_valid_path_costs_are_computed():
    graph = nx.DiGraph()
    graph.add_edge('s', 't', cost=(2, 3))
    graph.add_edge('s', 'dead')  # An invalid submitted path uses this uncosted edge.
    reward = Reward(graph, 's', 't', 3, [['s', 't'], ['s', 'dead', 't']], {})
    np.testing.assert_array_equal(reward.valid_costs, [[2, 3]])
    assert reward.invalid_paths == [['s', 'dead', 't']]


def test_cached_cost_dimension_matches_reference(hypervolume_reward, backend):
    with pytest.raises(ValueError, match='match reference_point'):
        hypervolume_reward.reward_hypervolume([10, 10, 10], backend=backend)


def test_empty_valid_costs(graph):
    reward = Reward(graph, 0, 3, 3, [[0, 9, 3]], {})
    assert reward.valid_costs == []


@pytest.mark.parametrize('normalize_igd', [False, True])
@pytest.mark.parametrize('normalize_costs', [False, True])
def test_igd_plus_normalization_flags(normalize_igd, normalize_costs):
    graph = nx.DiGraph()
    graph.add_edge('s', 't', cost=(200, 20, 5))
    front = {('a',): (100, 10, 5), ('b',): (300, 0, 5)}
    reward = Reward(graph, 's', 't', 1, [['s', 't']], front)
    original_costs = np.array(reward.valid_costs, copy=True)
    if normalize_costs:
        expected_igd = (np.sqrt(.5) + 1) / 2
    else:
        expected_igd = (np.sqrt(10100) + 20) / 2
    expected = 1 - expected_igd / np.sqrt(3) if normalize_igd else expected_igd
    assert reward.reward_igd(normalize_igd=normalize_igd,
                             normalize_costs=normalize_costs) == pytest.approx(expected)
    np.testing.assert_array_equal(reward.valid_costs, original_costs)
    assert list(front.values()) == [(100, 10, 5), (300, 0, 5)]
    if normalize_costs:
        np.testing.assert_allclose(reward.valid_costs_normalized, [[.5, 1, 0]])
        np.testing.assert_allclose(reward.pareto_front_normalized, [[0, .5, 0], [1, 0, 0]])
        assert np.all((reward.valid_costs_normalized >= 0) & (reward.valid_costs_normalized <= 1))
        assert np.all((reward.pareto_front_normalized >= 0) & (reward.pareto_front_normalized <= 1))


def test_igd_plus_custom_front_and_direction(hypervolume_reward):
    reward = hypervolume_reward
    # Candidate (2, 5) dominates (3, 6): IGD+ is zero, while ordinary IGD is positive.
    assert reward.reward_igd(front=[[3, 6]], normalize_igd=False,
                             normalize_costs=False) == 0
    assert reward.reward_igd(front=[[3, 6]]) == 1
    assert reward.reward_igd(normalize_igd=False, normalize_costs=False) == pytest.approx(1.5)


def test_igd_plus_perfect_front(hypervolume_reward):
    reward = hypervolume_reward
    reward = Reward(reward.graph, 's', 't', 3,
                    list(reward.pareto_frontier), reward.pareto_frontier)
    assert reward.reward_igd() == 1
    assert reward.reward_igd(normalize_igd=False) == 0


@pytest.mark.parametrize('front', [[], [[1]], [[1, float('nan')]], [[1, 2, 3]]])
def test_igd_plus_invalid_front(hypervolume_reward, front):
    with pytest.raises(ValueError):
        hypervolume_reward.reward_igd(front=front)


def test_igd_plus_empty_valid_costs(graph):
    reward = Reward(graph, 0, 3, 3, [], {('a',): (1, 2)})
    assert reward.reward_igd() == 0.0


@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('weight_by_valid', [True, False])
def test_additive_all_rewards(hypervolume_reward, backend, normalize, weight_by_valid):
    reward = hypervolume_reward
    reward.doctrine = lambda paths: .25
    selected = [reward.reward_valid_per_path, reward.reward_valid,
                reward.reward_hypervolume, reward.reward_doctrine, reward.reward_igd]
    weighted_terms = (reward.reward_hypervolume([10, 10], normalize=normalize, backend=backend)
                      + .25 + reward.reward_igd(normalize_igd=False, normalize_costs=False))
    expected = 2 + .5 + weighted_terms * (.5 if weight_by_valid else 1)
    assert reward.terminal_reward_additive(
        selected, normalize=normalize, weight_by_valid=weight_by_valid,
        reference_point=[10, 10], backend=backend,
        normalize_igd=False, normalize_costs=False,
    ) == pytest.approx(expected)


def test_additive_ablation(hypervolume_reward):
    reward = hypervolume_reward
    # Omitted rewards must not be evaluated or require their arguments.
    reward.pareto_frontier = {}
    assert reward.terminal_reward_additive([reward.reward_valid_per_path]) == 2
    assert reward.terminal_reward_additive([reward.reward_valid]) == .5


def test_additive_rejects_unsupported_function(hypervolume_reward):
    reward = hypervolume_reward
    for function in [lambda: 1, reward.is_valid_path, Reward.reward_valid]:
        with pytest.raises(ValueError, match='supported reward methods'):
            reward.terminal_reward_additive([function])


def test_additive_rejects_unknown_options(hypervolume_reward):
    with pytest.raises(TypeError, match='Unsupported arguments'):
        hypervolume_reward.terminal_reward_additive(
            [hypervolume_reward.reward_valid], unknown=True)


def test_additive_hypervolume_requires_reference(hypervolume_reward):
    with pytest.raises(TypeError, match='reference_point'):
        hypervolume_reward.terminal_reward_additive([hypervolume_reward.reward_hypervolume])


@pytest.mark.parametrize('has_valid_path', [False, True])
def test_additive_validity_weight_extremes(graph, has_valid_path):
    paths = [[0, 1, 3]] if has_valid_path else [[0, 9, 3]]
    reward = Reward(graph, 0, 3, 1, paths, {}, doctrine=lambda paths: .75)
    assert reward.terminal_reward_additive([reward.reward_doctrine]) == (.75 if has_valid_path else 0)
    assert reward.terminal_reward_additive([reward.reward_doctrine], weight_by_valid=False) == (.75 if has_valid_path else 0)


@pytest.mark.parametrize('custom_front', [False, True])
def test_igd_uses_shared_per_objective_normalization(custom_front):
    graph = nx.DiGraph()
    graph.add_edge('s', 'a', cost=(20, 200, 7))
    graph.add_edge('a', 't', cost=(0, 0, 0))
    graph.add_edge('s', 'b', cost=(40, 400, 7))
    graph.add_edge('b', 't', cost=(0, 0, 0))
    frontier = {('s', 'c', 't'): (10, 300, 7), ('s', 'a', 't'): (20, 200, 7)}
    reward = Reward(graph, 's', 't', 2, [['s', 'a', 't'], ['s', 'b', 't']], frontier)
    if custom_front:
        front = [[0, 500, 7], [20, 200, 7]]
        expected_costs = [[.5, 0, 0], [1, 2 / 3, 0]]
        expected_front = [[0, 1, 0], [.5, 0, 0]]
    else:
        front = None
        expected_costs = [[1 / 3, 0, 0], [1, 1, 0]]
        expected_front = [[0, .5, 0], [1 / 3, 0, 0]]
    reward.reward_igd(front=front)
    np.testing.assert_allclose(reward.valid_costs_normalized, expected_costs)
    np.testing.assert_allclose(reward.pareto_front_normalized, expected_front)
    # The same raw vector in both sets must map to exactly the same vector.
    np.testing.assert_array_equal(reward.valid_costs_normalized[0],
                                  reward.pareto_front_normalized[1])


@pytest.mark.parametrize('paths', [[], [[0, 9, 3]]])
@pytest.mark.parametrize('normalize', [False, True])
def test_empty_valid_rewards_short_circuit(graph, backend, paths, normalize):
    def doctrine(_):
        pytest.fail('Doctrine must not be called without valid paths')

    reward = Reward(graph, 0, 3, 3, paths, {}, doctrine=doctrine)
    assert reward.valid_paths == []
    # Empty reference/front data must not prevent automatic zero rewards.
    assert reward.reward_hypervolume(None, normalize=normalize, backend=backend) == 0.0
    assert reward.reward_doctrine() == 0.0
    assert reward.reward_igd(front=[], normalize_igd=normalize, normalize_costs=normalize) == 0.0
    assert reward.terminal_reward_additive(
        [reward.reward_valid, reward.reward_hypervolume, reward.reward_doctrine, reward.reward_igd],
        reference_point=None, normalize=normalize, backend=backend,
        weight_by_valid=False,
    ) == 0.0


def test_additive_records_raw_and_weighted_contributions(hypervolume_reward):
    reward = hypervolume_reward
    selected = [reward.reward_valid_per_path, reward.reward_valid, reward.reward_hypervolume]
    total = reward.terminal_reward_additive(selected, reference_point=[10, 10])
    assert reward.reward_components == pytest.approx(dict(
        reward_valid_per_path=2, reward_valid=.5, reward_hypervolume=40 / 55))
    assert reward.reward_contributions == pytest.approx(dict(
        reward_valid_per_path=2, reward_valid=.5, reward_hypervolume=.5 * 40 / 55))
    assert total == pytest.approx(sum(reward.reward_contributions.values()))
    reward.terminal_reward_additive([reward.reward_valid])
    assert reward.reward_components == reward.reward_contributions == {'reward_valid': .5}
    assert reward.terminal_reward_additive([]) == 0
    assert reward.reward_components == reward.reward_contributions == {}
