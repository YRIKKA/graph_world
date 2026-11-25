"""Post-hoc scores preserve saved ground truth and rank ties without agent-order bias."""
from copy import deepcopy
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from . import combined_metrics_post_hoc_eval as analysis


def synthetic_suite():
    graph = nx.DiGraph()
    graph.add_edge('s', 'a', time=1, risk=2)
    graph.add_edge('a', 't', time=1, risk=3)
    graph.add_edge('s', 'b', time=3, risk=1)
    graph.add_edge('b', 't', time=2, risk=1)
    graph.add_edge('s', 't', time=6, risk=6)
    instance = dict(nodes=list(graph), edges=list(graph.edges(data=True)), source='s', target='t',
                    names=['time', 'risk'], max_cost=6, pareto=[['s', 'a', 't'], ['s', 'b', 't']])
    saved = dict(graph_id='graph', family_id='graph', domain='in_distribution',
                 difficulty='easy', variant=0, instance=instance)
    return {'graph': saved}


def task(paths, raw_hv=280, agent='initial', mode='independent_exploration'):
    return dict(model_id=agent, graph_id='graph', family_id='graph', domain='in_distribution',
        difficulty='easy', variant=0, checkpoint_id='checkpoint-'+agent, k=2, B=16, repetition=0,
        mode=mode, status='ok', submission=dict(paths=paths),
        archive=[dict(path=['s', 'a', 't'], cost=[2, 5]), dict(path=['s', 'b', 't'], cost=[5, 2]),
                 dict(path=['s', 't'], cost=[6, 6])], metrics=dict(hypervolume=raw_hv, S=1))


def test_perfect_front_and_disjoint_routes_score_four():
    suite = synthetic_suite()
    row = task([['s', 'a', 't'], ['s', 'b', 't']])
    graph = analysis.prepare_graphs(suite, [row])['graph']
    result = analysis.score_task(row, graph)
    assert result['validity_fraction'] == 1
    assert result['igd_plus_score'] == 1
    assert result['hypervolume_ratio'] == 1
    assert result['min_jaccard_distance'] == 1
    assert result['combined_score'] == 4


def test_duplicates_reduce_validity_and_destroy_minimum_diversity():
    suite = synthetic_suite()
    duplicate = task([['s', 'a', 't'], ['s', 'a', 't']], raw_hv=238)
    dominated = task([['s', 't'], ['s', 't']], raw_hv=169)
    prepared = analysis.prepare_graphs(suite, [duplicate, dominated])['graph']
    result = analysis.score_task(duplicate, prepared)
    assert result['validity_fraction'] == .5
    assert result['valid_slot_fraction'] == 1
    assert result['min_jaccard_distance'] == 0
    assert result['hypervolume_ratio'] == pytest.approx(238 / 280)
    assert result['igd_plus_distance'] == pytest.approx(.375)
    assert result['igd_plus_score'] == pytest.approx(1 - .375 / np.sqrt(2))
    assert result['combined_score'] == pytest.approx(.5*(1 + result['igd_plus_score'] + 238 / 280))


def test_shared_elementwise_normalization_and_saved_truth_unchanged():
    suite = synthetic_suite()
    original = deepcopy(suite)
    best = task([['s', 'a', 't'], ['s', 'b', 't']])
    worst = task([['s', 't'], ['s', 't']], raw_hv=169)
    prepared = analysis.prepare_graphs(suite, [best, worst])['graph']
    np.testing.assert_array_equal(prepared['minimum'], [2, 2])
    np.testing.assert_array_equal(prepared['scale'], [4, 4])
    np.testing.assert_array_equal(prepared['front_normalized'], [[0, .75], [.75, 0]])
    assert suite == original


def test_empty_paths_receive_zero_metrics():
    row = task([[], []], raw_hv=0)
    prepared = analysis.prepare_graphs(synthetic_suite(), [row])['graph']
    result = analysis.score_task(row, prepared)
    assert all(result[key] == 0 for key, _ in analysis.METRICS)
    assert result['igd_plus_distance'] is None


def test_directed_edge_jaccard_and_insufficient_paths():
    assert analysis.minimum_jaccard([['s', 'a', 't'], ['s', 'a', 'b', 't']]) == pytest.approx(.75)
    assert analysis.minimum_jaccard([['s', 'a', 't']]) == 0
    assert analysis.minimum_jaccard([['s', 'a', 't'], ['s', 'a', 't']]) == 0


def test_fractional_ties_conserve_rank_positions_and_top_three():
    values = dict(a=1, b=.5, c=.5, d=.5, e=0)
    ranks = analysis.fractional_rank_credits(values)
    assert ranks['a']['first_fraction'] == 1
    for agent in ('b', 'c', 'd'):
        assert ranks[agent]['second_fraction'] == pytest.approx(1 / 3)
        assert ranks[agent]['third_fraction'] == pytest.approx(1 / 3)
        assert ranks[agent]['top3_fraction'] == pytest.approx(2 / 3)
    for key in analysis.RANK_COLUMNS[:3]:
        assert sum(row[key] for row in ranks.values()) == pytest.approx(1)
    assert sum(row['top3_fraction'] for row in ranks.values()) == pytest.approx(3)
    tied = analysis.fractional_rank_credits({str(i): 0 for i in range(9)})
    assert all(row['top3_fraction'] == pytest.approx(1 / 3) for row in tied.values())


def test_saved_artifact_checks_detect_tampering_and_missing_agents(tmp_path):
    directory = tmp_path/'comparisons/model/moo_rewards'
    directory.mkdir(parents=True)
    suite = list(synthetic_suite().values())
    row = task([['s', 'a', 't'], ['s', 'b', 't']])
    config = dict(training_runs=[], ks=[2], repetitions=1, modes=['independent_exploration'])
    config_id = analysis.hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (directory/'suite.jsonl').write_text(json.dumps(suite[0])+'\n')
    (directory/'tasks.jsonl').write_text(json.dumps(row)+'\n')
    suite_id = analysis.file_hash(directory/'suite.jsonl')
    analysis.write_json(directory/'suite_manifest.json', dict(config=config, config_id=config_id, suite_id=suite_id))
    analysis.write_json(directory/'completed.json', dict(task_count=1,
                        tasks_sha256=analysis.file_hash(directory/'tasks.jsonl')))
    analysis.write_json(directory/'evaluation_manifest.json', dict(config_id=config_id, suite_id=suite_id,
                        checkpoints={'initial': dict(checkpoint_id='checkpoint-initial')}))
    assert len(analysis.load_evaluation(tmp_path)[4]) == 1
    (directory/'tasks.jsonl').write_text('')
    with pytest.raises(ValueError, match='checksum'):
        analysis.load_evaluation(tmp_path)
    analysis.write_json(directory/'completed.json', dict(task_count=0,
                        tasks_sha256=analysis.file_hash(directory/'tasks.jsonl')))
    with pytest.raises(ValueError, match='Missing evaluation cells'):
        analysis.load_evaluation(tmp_path)


def test_pearson_known_values_constant_and_missing_pairs():
    assert analysis.pearson_coefficient([0, 1, 2, 3], [0, 1, 3, 2]) == pytest.approx((.8, 4))
    assert analysis.pearson_coefficient([0, 1, 2], [2, 1, 0]) == pytest.approx((-1, 3))
    assert analysis.pearson_coefficient([1, 1], [1, 2]) == (None, 2)
    assert analysis.pearson_coefficient([0, 1, np.nan], [1, 2, 9]) == pytest.approx((1, 2))
    assert analysis.pearson_coefficient([1], [1]) == (None, 1)
    with pytest.raises(ValueError, match='paired'):
        analysis.pearson_coefficient([1, 2], [1])


def test_correlation_stratification_averages_before_correlating():
    rows = []
    # Within-group variation makes task correlation .8, but group means correlate 1.
    for i, (x, y) in enumerate(zip([0, 1, 2, 3], [0, 1, 3, 2])):
        row = dict(agent_id='initial', condition='frozen', mode='independent_exploration',
                   domain='in_distribution', difficulty='easy' if i < 2 else 'hard',
                   family_id=str(i), k=2)
        row.update({metric: y for metric, _ in analysis.METRICS})
        row['reward_values'] = dict.fromkeys(analysis.CORRELATION_REWARDS, x)
        rows.append(row)
    correlations, strata = analysis.compute_reward_correlations(rows, ['independent_exploration'])
    assert len(strata) == 2
    overall = {row['level']: row for row in correlations if row['mode'] == 'all_modes'
               and row['domain'] == 'overall' and row['metric'] == 'combined_score'
               and row['reward_function'] == 'valid'}
    assert overall['pooled_tasks']['observations'] == 4
    assert overall['pooled_tasks']['pearson_r'] == pytest.approx(.8)
    assert overall['model_stratum_means']['observations'] == 2
    assert overall['model_stratum_means']['pearson_r'] == pytest.approx(1)


def test_correlation_reward_values_use_training_rules_and_all_combinations():
    row = task([['s', 'a', 't'], ['s', 'b', 't']])
    prepared = analysis.prepare_graphs(synthetic_suite(), [row])['graph']
    values = analysis.score_reward_functions(row, prepared)
    assert set(values) == set(analysis.CORRELATION_REWARDS)
    assert values['valid_per_path'] == 2
    assert values['valid'] == 1
    assert values['hypervolume'] == 1
    assert values['igd'] == 1
    assert values['valid_per_path+hypervolume+igd__weighted'] == 4
    assert values['valid+hypervolume+igd__weighted'] == 3
    duplicate = task([['s', 'a', 't'], ['s', 'a', 't']], raw_hv=238)
    values = analysis.score_reward_functions(duplicate, prepared)
    # Training HV has its own graph-derived reference, also for the weighted term.
    assert values['valid+hypervolume__weighted'] == pytest.approx(.5 + .5*values['hypervolume'])
    assert values['valid_per_path+igd__weighted'] == pytest.approx(1 + .5*values['igd'])
    empty = dict(row, status='error')
    assert all(value == 0 for value in analysis.score_reward_functions(empty, prepared).values())


@pytest.mark.parametrize('paths,raw_hv,expected', [
    ([['s', 'a', 't'], ['s', 'b', 't']], 280, 1),
    ([['s', 'a', 't'], ['s', 'a', 't']], 238, .5),
    ([['s', 'a', 't']], 238, .5),
    ([['s', 't']], 169, 0),
    ([['s', 'a', 't'], ['s', 'missing', 't']], 238, .5),
    ([], 0, 0),
])
def test_exact_pareto_membership_counts_unique_paths_over_requested_k(paths, raw_hv, expected):
    row = task(paths, raw_hv=raw_hv)
    prepared = analysis.prepare_graphs(synthetic_suite(), [row])['graph']
    result = analysis.score_task(row, prepared)
    assert result['pareto_membership_fraction'] == expected
    assert result['pareto_unique_count'] == expected*row['k']
    assert result['pareto_membership_fraction'] <= result['validity_fraction']


def test_exact_pareto_membership_requires_saved_path_not_equal_cost():
    suite = synthetic_suite()
    # Give an unstored route the same costs as a stored Pareto path.
    for u, v, cost in suite['graph']['instance']['edges']:
        if (u, v) == ('s', 't'):
            cost.update(time=2, risk=5)
    row = task([['s', 't']], raw_hv=238)
    prepared = analysis.prepare_graphs(suite, [row])['graph']
    result = analysis.score_task(row, prepared)
    assert result['validity_fraction'] == .5
    assert result['hypervolume_ratio'] == pytest.approx(238/280)
    assert result['pareto_membership_fraction'] == 0
    # Reuse saved membership verbatim; do not infer additional Pareto paths from costs.
    assert prepared['pareto_paths'] == {('s', 'a', 't'), ('s', 'b', 't')}
