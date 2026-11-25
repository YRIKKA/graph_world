"""Rescore a completed MOO experiment using saved predictions; no model inference.

All primary scores are higher-is-better. IGD+ is transformed to 1-d/sqrt(m)
using shared per-objective bounds for all agents on each graph. Hypervolume is
normalized by the saved true front's hypervolume. Raw indicators are also saved.
The combined reward is V + V*(IGD_score + HV_ratio + minimum_Jaccard_distance).
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
from importlib.metadata import version
from itertools import combinations, product
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from jm_test.graph_world.grpo_mosp.experiment_moo_rewards.moo_rewards import COMBINATIONS, functions_for
    from jm_test.graph_world.grpo_mosp.rewards import Reward
    from jm_test.graph_world.grpo_mosp.reporting import table
    from jm_test.graph_world.grpo_mosp.run_artifacts import console_log
else:
    from .moo_rewards import COMBINATIONS, functions_for
    from ..rewards import Reward
    from ..reporting import table
    from ..run_artifacts import console_log

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE.parent/'full_runs/moo_rewards_run_20261113140643'
DOMAINS = ('in_distribution', 'size', 'density', 'costs', 'objectives')
METRICS = (
    ('validity_fraction', 'Validity fraction'),
    ('igd_plus_score', 'IGD+ score'),
    ('hypervolume_ratio', 'Hypervolume ratio'),
    ('min_jaccard_distance', 'Minimum Jaccard distance'),
    ('pareto_membership_fraction', 'Exact Pareto-path fraction'),
    ('combined_score', 'Combined score'),
)
RANK_COLUMNS = ('first_fraction', 'second_fraction', 'third_fraction', 'top3_fraction')
TIE_TOLERANCE = 1e-10
# Include all eight training objectives and the two remaining standalone signals.
CORRELATION_REWARDS = (*COMBINATIONS, 'hypervolume', 'igd')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setting_key(task):
    return (task['graph_id'], task['k'], task['B'], task['repetition'], task['mode'])


def load_evaluation(run):
    # Locate local artifacts instead of relying on historical absolute filenames.
    candidates = sorted((run/'comparisons').glob('*/moo_rewards/completed.json'))
    if len(candidates) != 1:
        raise ValueError('Expected exactly one completed MOO comparison under this run')
    directory = candidates[0].parent
    manifest = json.loads((directory/'suite_manifest.json').read_text())
    completed = json.loads(candidates[0].read_text())
    evaluation = json.loads((directory/'evaluation_manifest.json').read_text())
    config = manifest['config']
    sources = {name: directory/name for name in
               ('suite.jsonl', 'tasks.jsonl', 'suite_manifest.json', 'completed.json', 'evaluation_manifest.json')}
    hashes = {name: file_hash(path) for name, path in sources.items()}
    if hashes['suite.jsonl'] != manifest['suite_id'] or hashes['tasks.jsonl'] != completed['tasks_sha256']:
        raise ValueError('Saved graph suite or evaluation outputs failed their original checksum')
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if config_id != manifest['config_id'] or evaluation['config_id'] != config_id:
        raise ValueError('Saved evaluation configuration identities disagree')
    if evaluation['suite_id'] != manifest['suite_id']:
        raise ValueError('Evaluation does not refer to this saved graph suite')
    suite = read_jsonl(sources['suite.jsonl'])
    tasks = read_jsonl(sources['tasks.jsonl'])
    graphs = {row['graph_id']: row for row in suite}
    agents = ['initial'] + [row['id'] for row in config['training_runs']]
    if len(graphs) != len(suite) or len(set(agents)) != len(agents):
        raise ValueError('Duplicate graph or agent identity')
    expected = {(agent, graph, k, 8*k, repetition, mode)
                for agent, graph, k, repetition, mode in product(
                    agents, graphs, config['ks'], range(config['repetitions']), config['modes'])}
    seen = set()
    for task in tasks:
        cell = (task['model_id'], *setting_key(task))
        if cell in seen or cell not in expected:
            raise ValueError('Duplicate or unexpected agent/evaluation setting')
        seen.add(cell)
        graph = graphs[task['graph_id']]
        if any(task[key] != graph[key] for key in ('family_id', 'domain', 'difficulty', 'variant')):
            raise ValueError('Task graph metadata differs from the saved suite')
        if task['checkpoint_id'] != evaluation['checkpoints'][task['model_id']]['checkpoint_id']:
            raise ValueError('Task checkpoint identity differs from its evaluation manifest')
    if seen != expected or len(tasks) != completed['task_count']:
        raise ValueError('Missing evaluation cells; partial models must not be silently omitted')
    condition = {'initial': 'frozen'}
    condition.update({row['id']: row['condition'] for row in config['training_runs']})
    return config, agents, condition, graphs, tasks, sources, hashes


def path_cost(graph, names, path):
    # Reuse the stored Pareto path membership; this only sums their saved edge costs.
    return tuple(sum(graph[u][v].get(name, 1) for u, v in zip(path, path[1:])) for name in names)


def valid_path(graph, source, target, path):
    if not isinstance(path, (list, tuple)) or not path:
        return False
    try:
        return (path[0] == source and path[-1] == target and len(set(path)) == len(path)
                and all(node in graph for node in path)
                and all(graph.has_edge(u, v) for u, v in zip(path, path[1:])))
    except TypeError:
        return False


def returned_valid_paths(task, graph, instance):
    submission = task.get('submission')
    paths = submission.get('paths') if isinstance(submission, dict) and set(submission) == {'paths'} else None
    if task['status'] != 'ok' or not isinstance(paths, list) or len(paths) > task['k']:
        return []
    # Retain duplicate slots for pairwise diversity; validity itself counts unique paths.
    valid = [path for path in paths if valid_path(graph, instance['source'], instance['target'], path)]
    archive = {tuple(item['path']) for item in task.get('archive', [])}
    if any(tuple(path) not in archive for path in valid):
        raise ValueError('A returned valid path was not in the saved exploration archive')
    return valid


def minimum_jaccard(paths):
    # Zero means there is no demonstrable pairwise diversity when fewer than two paths exist.
    if len(paths) < 2:
        return 0.0
    edge_sets = [set(zip(path, path[1:])) for path in paths]
    return min(1 - len(left & right)/len(left | right) if left | right else 0.0
               for left, right in combinations(edge_sets, 2))


def prepare_graphs(graphs, tasks):
    prepared = {}
    for graph_id, saved in graphs.items():
        instance = saved['instance']
        graph = nx.DiGraph()
        graph.add_nodes_from(instance['nodes'])
        graph.add_edges_from(instance['edges'])
        names = instance['names']
        frontier = []
        for path in instance['pareto']:
            if not valid_path(graph, instance['source'], instance['target'], path):
                raise ValueError('Stored Pareto path is invalid for its saved graph')
            frontier.append(path_cost(graph, names, path))
        if not frontier:
            raise ValueError('The saved Pareto front is empty')
        # One reference point per distinct cost vector avoids multiplicity bias from tied routes.
        front = np.unique(np.asarray(frontier, dtype=float), axis=0)
        reference = np.full(len(names), (len(graph)-1)*instance['max_cost']+1, dtype=float)
        indicator = HV(ref_point=reference)
        front_hv = float(indicator(front))
        if front_hv <= 0:
            raise ValueError('The saved true front has zero hypervolume')
        prepared[graph_id] = dict(graph=graph, instance=instance, front=front, reference=reference,
                                  indicator=indicator, front_hv=front_hv, pooled_costs=[front],
                                  pareto_paths={tuple(path) for path in instance['pareto']})
    for task in tasks:
        item = prepared[task['graph_id']]
        paths = returned_valid_paths(task, item['graph'], item['instance'])
        if paths:
            item['pooled_costs'].append(np.asarray([path_cost(item['graph'], item['instance']['names'], path)
                                                  for path in paths], dtype=float))
    for item in prepared.values():
        # Every agent and mode uses identical per-objective bounds on a given graph.
        pooled = np.vstack(item.pop('pooled_costs'))
        item['minimum'] = pooled.min(axis=0)
        scale = pooled.max(axis=0) - item['minimum']
        item['scale'] = np.where(scale == 0, 1.0, scale)
        item['front_normalized'] = (item['front']-item['minimum'])/item['scale']
    return prepared


def score_task(task, item):
    paths = returned_valid_paths(task, item['graph'], item['instance'])
    unique = list(dict.fromkeys(tuple(path) for path in paths))
    validity = len(unique)/task['k']
    # Compare ordered node sequences against saved truth, not merely equal cost vectors.
    # Duplicate and missing submissions earn no additional credit, as with validity.
    pareto_count = len(set(unique) & item['pareto_paths'])
    result = dict(validity_fraction=validity, igd_plus_score=0.0, hypervolume_ratio=0.0,
                  min_jaccard_distance=minimum_jaccard(paths), combined_score=0.0,
                  pareto_membership_fraction=pareto_count/task['k'], pareto_unique_count=pareto_count,
                  igd_plus_distance=None, igd_plus_raw=None, hypervolume_raw=0.0,
                  valid_slot_fraction=len(paths)/task['k'], valid_unique_count=len(unique))
    if paths:
        costs = np.asarray([path_cost(item['graph'], item['instance']['names'], path) for path in unique], dtype=float)
        normalized = (costs-item['minimum'])/item['scale']
        distance = float(IGDPlus(item['front_normalized'])(normalized))
        result.update(igd_plus_distance=distance, igd_plus_raw=float(IGDPlus(item['front'])(costs)),
                      igd_plus_score=float(1-distance/np.sqrt(costs.shape[1])),
                      hypervolume_raw=float(item['indicator'](costs)))
        result['hypervolume_ratio'] = result['hypervolume_raw']/item['front_hv']
    # Check the reconstructed indicator against the original evaluation, not just against itself.
    if not np.isclose(result['hypervolume_raw'], task['metrics']['hypervolume'], rtol=1e-9, atol=1e-8):
        raise ValueError('Reconstructed hypervolume differs from the saved original evaluation')
    for name, _ in METRICS[:-1]:
        if not -1e-10 <= result[name] <= 1+1e-10:
            raise ValueError(f'Normalized metric out of bounds: {name}')
        result[name] = float(np.clip(result[name], 0, 1))
    result['combined_score'] = validity * (1 + result['igd_plus_score']
        + result['hypervolume_ratio'] + result['min_jaccard_distance'])
    return result



def score_reward_functions(task, item):
    """Evaluate every training reward on every output, using the actual Reward methods."""
    graph = item['graph'].copy()
    names = item['instance']['names']
    for _, _, edge in graph.edges(data=True):
        edge['cost'] = tuple(edge.get(name, 1) for name in names)
    # Keep saved Pareto path multiplicity, matching the training adapter exactly.
    frontier = {tuple(path): path_cost(graph, names, path) for path in item['instance']['pareto']}
    submission = task.get('submission')
    paths = submission.get('paths') if isinstance(submission, dict) and set(submission) == {'paths'} else None
    wellformed = (isinstance(paths, list) and len(paths) <= task['k']
                  and all(isinstance(path, list) and all(isinstance(node, str) for node in path)
                          for path in paths))
    if task['status'] != 'ok' or not wellformed:
        return {name: 0.0 for name in CORRELATION_REWARDS}
    reward = Reward(graph, item['instance']['source'], item['instance']['target'], task['k'], paths, frontier)
    # The training adapter bounds each objective by the graph's actual maximum edge cost.
    reference = [max((edge['cost'][i] for _, _, edge in graph.edges(data=True)), default=0)
                 * (len(graph)-1) + 1 for i in range(len(names))]
    values = {}
    for name in CORRELATION_REWARDS:
        functions = functions_for(name) if name in COMBINATIONS else ['reward_'+name]
        kwargs = dict(normalize=True, weight_by_valid=True)
        if 'reward_hypervolume' in functions:
            kwargs.update(reference_point=reference, backend='pymoo')
        if 'reward_igd' in functions:
            kwargs.update(normalize_igd=True, normalize_costs=True)
        # Standalone signals must remain standalone, without the validity multiplier.
        if name not in COMBINATIONS:
            kwargs['weight_by_valid'] = False
        values[name] = reward.terminal_reward_additive([getattr(reward, method) for method in functions], **kwargs)
    return values


def pearson_coefficient(x, y):
    """Return Pearson r and sample count; constant or insufficient data give None."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError('Pearson inputs must be paired one-dimensional arrays')
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = len(x)
    if n < 2:
        return None, n
    # Center before the dot products to avoid cancellation in the covariance formula.
    x, y = x-x.mean(), y-y.mean()
    nx_value, ny_value = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx_value == 0 or ny_value == 0:
        return None, n
    return float(np.clip(np.dot(x/nx_value, y/ny_value), -1, 1)), n


def correlation_observations(rows):
    """Average within model/mode/domain/difficulty, using the original family weighting."""
    cells = defaultdict(list)
    for row in rows:
        cells[row['agent_id'], row['mode'], row['domain'], row['difficulty']].append(row)
    averaged = []
    for (agent, mode, domain, difficulty), items in cells.items():
        row = dict(agent_id=agent, condition=items[0]['condition'], mode=mode,
                   domain=domain, difficulty=difficulty, underlying_tasks=len(items))
        row.update({metric: stratified_mean(items, metric) for metric, _ in METRICS})
        row['reward_values'] = {name: stratified_mean(
            [dict(item, correlation_reward=item['reward_values'][name]) for item in items], 'correlation_reward')
            for name in CORRELATION_REWARDS}
        averaged.append(row)
    return averaged


def compute_reward_correlations(rows, modes):
    """Correlate reward and metric pairs across tasks and across model-stratum means."""
    strata = correlation_observations(rows)
    results = []
    analysis_scopes = scopes(modes) + [('all_modes', domain, 'all') for domain in DOMAINS]
    for level, observations in (('pooled_tasks', rows), ('model_stratum_means', strata)):
        for scope in analysis_scopes:
            selected = [row for row in observations if in_scope(row, scope)]
            if not selected:
                continue
            for reward_name in CORRELATION_REWARDS:
                for metric, _ in METRICS:
                    r, n = pearson_coefficient([row['reward_values'][reward_name] for row in selected],
                                               [row[metric] for row in selected])
                    results.append(dict(level=level, mode=scope[0], domain=scope[1], difficulty=scope[2],
                        reward_function=reward_name, metric=metric, pearson_r=r, observations=n,
                        status='ok' if r is not None else 'undefined_constant_or_insufficient'))
    return results, strata


def report_reward_correlations(correlations, modes, output):
    print('\nReward/metric correlations: Pearson r; all models including frozen are pooled.')
    print('Each reward function is evaluated on every saved output, regardless of its training condition.')
    print('Pooled tasks: one observation per model/graph/mode/k/repetition.')
    print('Model-stratum means: average within each model/mode/domain/difficulty, then correlate those means.')
    print('This correlates group means; it is not an average of within-group correlations or of ranks.')
    print('Evaluation metrics use the existing shared graph bounds and higher-is-better IGD+ score/HV ratio.')
    print('Reward values reproduce Reward and the training adapter: IGD bounds use each output plus its saved front;')
    print('reward HV uses the actual maximum edge cost per objective. These can differ from evaluation normalization.')
    print('Raw IGD+ distance is lower-is-better; the IGD+ score reported here reverses that direction.')
    print('Constant variables have undefined correlation (NA). Correlations are descriptive, with repeated graphs/models;')
    print('they do not establish training benefit. Shared terms create mathematical dependence, including some exact self-correlations.')
    print('Count and validity fraction are proportional when k is fixed, so their correlations are identical.')
    for level in ('pooled_tasks', 'model_stratum_means'):
        # Print overall, each mode, and each domain; CSV also includes difficulty and shift scopes.
        selected_scopes = [('all_modes', 'overall', 'all')]
        selected_scopes += [(mode, 'overall', 'all') for mode in modes]
        selected_scopes += [('all_modes', domain, 'all') for domain in DOMAINS]
        for scope in selected_scopes:
            selected = [row for row in correlations if row['level'] == level and
                        (row['mode'], row['domain'], row['difficulty']) == scope]
            if not selected:
                continue
            indexed = {(row['reward_function'], row['metric']): row for row in selected}
            table(f'Pearson reward/metric correlation | {level} | {scope[0]} | {scope[1]}',
                  ['Reward', 'N', 'IGD+ score', 'HV ratio', 'Min Jaccard', 'Validity', 'Pareto fraction', 'Combined'],
                  [[name, indexed[name, 'combined_score']['observations'],
                    *['NA' if indexed[name, metric]['pearson_r'] is None else
                      f"{indexed[name, metric]['pearson_r']:+.5f}"
                      for metric in ('igd_plus_score', 'hypervolume_ratio', 'min_jaccard_distance',
                                     'validity_fraction', 'pareto_membership_fraction', 'combined_score')]] for name in CORRELATION_REWARDS])
            if scope[1] != 'overall':
                continue
            fig, axes = plt.subplots(3, 2, figsize=(17, 17))
            labels = [agent_label(name) for name in CORRELATION_REWARDS]
            for ax, (metric, title) in zip(axes.flat, METRICS):
                values = [indexed[name, metric]['pearson_r'] for name in CORRELATION_REWARDS]
                colors = ['#2878b5' if value is not None and value >= 0 else '#c45c48' for value in values]
                ax.barh(np.arange(len(labels)), [value if value is not None else 0 for value in values], color=colors)
                ax.set_yticks(np.arange(len(labels)), labels)
                ax.invert_yaxis()
                for i, value in enumerate(values):
                    ax.text((value if value is not None else 0)+(.025 if value is None or value >= 0 else -.025),
                            i, 'NA' if value is None else f'{value:+.3f}', va='center',
                            ha='left' if value is None or value >= 0 else 'right', fontsize=8)
                ax.axvline(0, color='black', linewidth=.7)
                ax.set(title=title, xlim=(-1.2, 1.2), xlabel='Pearson correlation coefficient')
                ax.grid(axis='x', alpha=.2)
            for unused in list(axes.flat)[len(METRICS):]:
                unused.axis('off')
            n = selected[0]['observations']
            fig.suptitle(f'Reward values vs. evaluation metrics | {level} | {scope[0]} | N={n}', fontsize=14)
            save_figure(fig, output/'plots'/scope[0]/'overall'/('reward_metric_correlations_'+level))


def fractional_rank_credits(values, tolerance=TIE_TOLERANCE):
    """Share positions occupied by tied agents; never resolve ties by agent order."""
    ordered = sorted(values, key=lambda agent: (-values[agent], agent))
    result = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and abs(values[ordered[end]]-values[ordered[start]]) <= tolerance:
            end += 1
        # A tie over positions 2..5 earns each member 1/4 credit for second and third.
        for agent in ordered[start:end]:
            credits = [float(start <= rank < end)/(end-start) for rank in range(3)]
            result[agent] = dict(first_fraction=credits[0], second_fraction=credits[1],
                third_fraction=credits[2], top3_fraction=sum(credits),
                competition_rank=start+1, tied_agents=end-start)
        start = end
    return result


def scopes(modes):
    result = [('all_modes', 'overall', 'all')]
    for mode in modes:
        for domain in ('overall', *DOMAINS, 'distribution_shift_macro'):
            for difficulty in ('all', 'easy', 'hard'):
                result.append((mode, domain, difficulty))
    return result


def in_scope(row, scope):
    mode, domain, difficulty = scope
    return ((mode == 'all_modes' or row['mode'] == mode)
            and (domain == 'overall' or row['domain'] == domain
                 or domain == 'distribution_shift_macro' and row['domain'] != 'in_distribution')
            and (difficulty == 'all' or row['difficulty'] == difficulty))


def stratified_mean(rows, metric):
    # Average repetitions/variants within graph families, then equally weight difficulty/domain strata.
    families = defaultdict(list)
    for row in rows:
        families[row['mode'], row['domain'], row['difficulty'], row['family_id'], row['k']].append(row[metric])
    strata = defaultdict(list)
    for (mode, domain, difficulty, _, _), values in families.items():
        strata[mode, domain, difficulty].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in strata.values()]))


def summarize(rows, rank_rows, agents, condition, modes):
    summary, fractions = [], []
    for scope in scopes(modes):
        selected = [row for row in rows if in_scope(row, scope)]
        if not selected:
            continue
        for agent in agents:
            items = [row for row in selected if row['agent_id'] == agent]
            row = dict(mode=scope[0], domain=scope[1], difficulty=scope[2], agent_id=agent,
                       condition=condition[agent], evaluation_settings=len(items))
            row.update({metric: stratified_mean(items, metric) for metric, _ in METRICS})
            summary.append(row)
        for metric, _ in METRICS:
            for agent in agents:
                items = [row for row in rank_rows if row['agent_id'] == agent and row['metric'] == metric
                         and in_scope(row, scope)]
                fractions.append(dict(mode=scope[0], domain=scope[1], difficulty=scope[2],
                    metric=metric, agent_id=agent, condition=condition[agent], evaluation_settings=len(items),
                    **{key: float(np.mean([row[key] for row in items])) for key in RANK_COLUMNS},
                    joint_first_fraction=float(np.mean([row['competition_rank'] == 1 for row in items])),
                    joint_second_fraction=float(np.mean([row['competition_rank'] == 2 for row in items])),
                    joint_third_fraction=float(np.mean([row['competition_rank'] == 3 for row in items])),
                    joint_top3_fraction=float(np.mean([row['competition_rank'] <= 3 for row in items]))))
    return summary, fractions



def rank_evaluation_strata(summary, agents, condition, modes):
    """Also rank the 20 mode/domain/difficulty combinations after graph averaging."""
    cells = defaultdict(dict)
    for row in summary:
        if row['mode'] in modes and row['domain'] in DOMAINS and row['difficulty'] in ('easy', 'hard'):
            cells[row['mode'], row['domain'], row['difficulty']][row['agent_id']] = row
    ranked = []
    for (mode, domain, difficulty), competitors in cells.items():
        if set(competitors) != set(agents):
            raise ValueError('Incomplete evaluation stratum')
        for metric, _ in METRICS:
            credits = fractional_rank_credits({agent: row[metric] for agent, row in competitors.items()})
            for agent in agents:
                ranked.append(dict(mode=mode, domain=domain, difficulty=difficulty,
                    agent_id=agent, condition=condition[agent], metric=metric,
                    value=competitors[agent][metric], **credits[agent]))
    # Reuse the rank aggregation without changing the graph-level metric estimates.
    fractions = []
    for scope in scopes(modes):
        for metric, _ in METRICS:
            selected = [row for row in ranked if row['metric'] == metric and in_scope(row, scope)]
            if not selected:
                continue
            for agent in agents:
                items = [row for row in selected if row['agent_id'] == agent]
                fractions.append(dict(mode=scope[0], domain=scope[1], difficulty=scope[2],
                    metric=metric, agent_id=agent, condition=condition[agent], evaluation_settings=len(items),
                    **{key: float(np.mean([row[key] for row in items])) for key in RANK_COLUMNS},
                    joint_first_fraction=float(np.mean([row['competition_rank'] == 1 for row in items])),
                    joint_second_fraction=float(np.mean([row['competition_rank'] == 2 for row in items])),
                    joint_third_fraction=float(np.mean([row['competition_rank'] == 3 for row in items])),
                    joint_top3_fraction=float(np.mean([row['competition_rank'] <= 3 for row in items]))))
    return ranked, fractions


def plot_stratum_ranks(fractions, agents, condition, modes, output):
    for mode in ('all_modes', *modes):
        fig, axes = plt.subplots(3, 2, figsize=(17, 16))
        y = np.arange(len(agents))
        for ax, (metric, title) in zip(axes.flat, METRICS):
            indexed = {row['agent_id']: row for row in fractions if row['metric'] == metric
                       and row['mode'] == mode and row['domain'] == 'overall' and row['difficulty'] == 'all'}
            for offset, (key, label, color) in enumerate(zip(RANK_COLUMNS,
                    ('First', 'Second', 'Third', 'Top 3'), ('#2a9d8f', '#e9c46a', '#f4a261', '#264653'))):
                ax.barh(y+(offset-1.5)*.19, [indexed[agent][key] for agent in agents], height=.18,
                        label=label, color=color)
            ax.set_yticks(y, [agent_label(condition[agent]) for agent in agents])
            ax.invert_yaxis()
            ax.set(title=title, xlim=(0, 1.05), xlabel='Fraction of mode/domain/difficulty strata')
            ax.legend(fontsize=8)
            ax.grid(axis='x', alpha=.2)
        for unused in list(axes.flat)[len(METRICS):]:
            unused.axis('off')
        fig.suptitle(f'Evaluation-stratum ranks, fractional tie credit | {mode}', fontsize=14)
        save_figure(fig, output/'plots'/mode/'overall'/'rank_fractions_by_evaluation_stratum')


def agent_label(condition):
    if condition == 'frozen':
        return 'Frozen base'
    return condition.replace('valid_per_path', 'Count').replace('valid', 'Fraction').replace(
        'hypervolume', 'HV').replace('igd', 'IGD+').replace('__weighted', '').replace('+', ' + ')


def save_figure(fig, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(destination.with_suffix('.'+suffix), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_comparisons(summary, fractions, agents, condition, modes, output):
    labels = [agent_label(condition[agent]) for agent in agents]
    colors = ['#777777'] + ['#2878b5']*(len(agents)-1)
    plot_scopes = [('all_modes', 'overall', 'all')]
    plot_scopes += [(mode, domain, 'all') for mode in modes
                   for domain in ('overall', *DOMAINS, 'distribution_shift_macro')]
    for scope in plot_scopes:
        selected = {row['agent_id']: row for row in summary
                    if (row['mode'], row['domain'], row['difficulty']) == scope}
        if not selected:
            continue
        fig, axes = plt.subplots(3, 2, figsize=(17, 16))
        for ax, (metric, title) in zip(axes.flat, METRICS):
            values = [selected[agent][metric] for agent in agents]
            ax.barh(labels[::-1], values[::-1], color=colors[::-1])
            for index, value in enumerate(values[::-1]):
                ax.text(value+.01, index, f'{value:.3f}', va='center', fontsize=8)
            ax.set(title=title+' (higher is better)', xlim=(0, 4.2 if metric == 'combined_score' else 1.1))
            ax.grid(axis='x', alpha=.2)
        for unused in list(axes.flat)[len(METRICS):]:
            unused.axis('off')
        fig.suptitle(f'Saved evaluation metrics | {scope[0]} | {scope[1]}', fontsize=14)
        save_figure(fig, output/'plots'/scope[0]/scope[1]/'agent_metrics')
        fig, axes = plt.subplots(3, 2, figsize=(17, 16))
        y = np.arange(len(agents))
        for ax, (metric, title) in zip(axes.flat, METRICS):
            indexed = {row['agent_id']: row for row in fractions if row['metric'] == metric
                       and (row['mode'], row['domain'], row['difficulty']) == scope}
            for offset, (key, label, color) in enumerate(zip(RANK_COLUMNS,
                    ('First', 'Second', 'Third', 'Top 3'), ('#2a9d8f', '#e9c46a', '#f4a261', '#264653'))):
                ax.barh(y+(offset-1.5)*.19, [indexed[agent][key] for agent in agents],
                        height=.18, label=label, color=color)
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
            ax.set(title=title, xlim=(0, 1.05), xlabel='Fraction of paired evaluation settings')
            ax.legend(fontsize=8)
            ax.grid(axis='x', alpha=.2)
        for unused in list(axes.flat)[len(METRICS):]:
            unused.axis('off')
        fig.suptitle(f'Rank frequencies, fractional tie credit | {scope[0]} | {scope[1]}', fontsize=14)
        save_figure(fig, output/'plots'/scope[0]/scope[1]/'rank_fractions')


def print_results(summary, fractions, agents, condition, modes):
    selected_scopes = [('all_modes', 'overall', 'all')]
    selected_scopes += [(mode, domain, 'all') for mode in modes
                        for domain in ('overall', *DOMAINS, 'distribution_shift_macro')]
    for scope in selected_scopes:
        indexed = {row['agent_id']: row for row in summary
                   if (row['mode'], row['domain'], row['difficulty']) == scope}
        if not indexed:
            continue
        table(f'Combined metrics | {scope[0]} | {scope[1]}',
              ['Agent', 'Validity', 'IGD+ score', 'HV ratio', 'Min Jaccard', 'Pareto fraction', 'Combined'],
              [[condition[agent], *[f'{indexed[agent][key]:.6f}' for key, _ in METRICS]] for agent in agents])
        for metric, title in METRICS:
            indexed = {row['agent_id']: row for row in fractions if row['metric'] == metric
                       and (row['mode'], row['domain'], row['difficulty']) == scope}
            table(f'Rank fractions | {title} | {scope[0]} | {scope[1]}',
                  ['Agent', 'Settings', 'First', 'Second', 'Third', 'Top 3'],
                  [[condition[agent], indexed[agent]['evaluation_settings'],
                    *[f'{100*indexed[agent][key]:.2f}%' for key in RANK_COLUMNS]] for agent in agents])


def main(run=DEFAULT_RUN, output=None):
    run = Path(run).expanduser().resolve()
    output = Path(output).expanduser().resolve() if output else run/'combined_metrics_post_hoc_eval'
    config, agents, condition, graphs, tasks, sources, hashes = load_evaluation(run)
    output.mkdir(parents=True, exist_ok=True)
    # Refresh only this generated analysis log; original experiment logs are untouched.
    log_path = output/'log_combined_metrics.txt'
    log_path.write_text('')
    with console_log(log_path):
        print('Post-hoc combined metric evaluation; saved predictions only, no inference.')
        print(f'Run: {run}\nOutput: {output}\nAgents: {len(agents)} | Tasks: {len(tasks)} | Graphs: {len(graphs)}')
        print('Validity = unique valid returned paths / requested k; missing and duplicate paths score zero.')
        print('IGD+ score = 1 - IGD+(normalized costs)/sqrt(m); all agents share per-graph objective bounds.')
        print('Reference IGD+ front uses distinct cost vectors of the saved Pareto paths; no front search is performed.')
        print('Hypervolume ratio uses the original evaluation reference point and stored true front.')
        print('Jaccard = minimum of 1-|E1 intersect E2|/|E1 union E2| over valid returned path pairs.')
        print('Jaccard = 0 with fewer than two valid paths; duplicate paths give distance 0.')
        print('Exact Pareto-path fraction = unique valid returned paths in saved Pareto path set / requested k.')
        print('Membership requires identical ordered nodes; duplicates and missing paths earn no credit.')
        print('This additional metric is not added to the existing combined score.')
        print('Combined = V + V*(IGD+ score + HV ratio + minimum Jaccard); range [0,4].')
        print('Graph-level ranks: graph/k/budget/repetition/mode is one paired setting; ties share occupied positions equally.')
        print('Stratum-level ranks are also exported for mode/domain/difficulty settings, after averaging their graphs.')
        print('Joint competition-rank fractions are additional CSV columns; they count tied ranks in full.')
        print('All six primary metrics are higher-is-better. Raw IGD+ distances (lower is better) and HV are in task exports.')
        prepared = prepare_graphs(graphs, tasks)
        rows = []
        for task in tasks:
            row = {key: task[key] for key in ('graph_id', 'family_id', 'domain', 'difficulty', 'variant',
                                             'k', 'B', 'repetition', 'mode', 'status', 'checkpoint_id')}
            row.update(agent_id=task['model_id'], condition=condition[task['model_id']],
                       original_pareto_S=task['metrics']['S'], **score_task(task, prepared[task['graph_id']]))
            row['reward_values'] = score_reward_functions(task, prepared[task['graph_id']])
            rows.append(row)
        paired = defaultdict(dict)
        for row in rows:
            paired[setting_key(row)][row['agent_id']] = row
        rank_rows = []
        for key, competitors in paired.items():
            if set(competitors) != set(agents):
                raise ValueError('Rank comparison has missing competitors')
            for metric, _ in METRICS:
                credits = fractional_rank_credits({agent: row[metric] for agent, row in competitors.items()})
                for agent in agents:
                    row = competitors[agent]
                    rank_rows.append(dict(graph_id=key[0], k=key[1], B=key[2], repetition=key[3], mode=key[4],
                        domain=row['domain'], difficulty=row['difficulty'], agent_id=agent, condition=condition[agent],
                        metric=metric, value=row[metric], **credits[agent]))
        summary, fractions = summarize(rows, rank_rows, agents, condition, config['modes'])
        print_results(summary, fractions, agents, condition, config['modes'])
        stratum_ranks, stratum_fractions = rank_evaluation_strata(summary, agents, condition, config['modes'])
        for mode in ('all_modes', *config['modes']):
            for metric, title in METRICS:
                indexed = {row['agent_id']: row for row in stratum_fractions if row['mode'] == mode
                           and row['domain'] == 'overall' and row['difficulty'] == 'all' and row['metric'] == metric}
                table(f'Evaluation-stratum rank fractions | {title} | {mode}',
                      ['Agent', 'Strata', 'First', 'Second', 'Third', 'Top 3'],
                      [[condition[agent], indexed[agent]['evaluation_settings'],
                        *[f'{100*indexed[agent][key]:.2f}%' for key in RANK_COLUMNS]] for agent in agents])
        with (output/'task_metrics.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        flat_rows = [{**{key: value for key, value in row.items() if key != 'reward_values'},
                      **{'reward_value__'+key: value for key, value in row['reward_values'].items()}} for row in rows]
        write_csv(output/'task_metrics.csv', flat_rows)
        correlations, correlation_strata = compute_reward_correlations(rows, config['modes'])
        report_reward_correlations(correlations, config['modes'], output)
        write_csv(output/'reward_metric_correlations.csv', correlations)
        write_json(output/'reward_metric_correlations.json', correlations)
        write_json(output/'correlation_stratum_observations.json', correlation_strata)
        write_csv(output/'metric_summary.csv', summary)
        write_csv(output/'rankings_per_setting.csv', rank_rows)
        write_csv(output/'rank_fractions.csv', fractions)
        write_csv(output/'rankings_by_evaluation_stratum.csv', stratum_ranks)
        write_csv(output/'rank_fractions_by_evaluation_stratum.csv', stratum_fractions)
        write_json(output/'results.json', dict(metric_summary=summary, rank_fractions=fractions, stratum_rank_fractions=stratum_fractions,
            reward_metric_correlations=correlations))
        normalization = {key: dict(minimum=item['minimum'].tolist(), scale=item['scale'].tolist(),
            reference_point=item['reference'].tolist(), true_hypervolume=item['front_hv'],
            pareto_path_count=len(item['instance']['pareto']), distinct_front_vectors=len(item['front']))
            for key, item in prepared.items()}
        write_json(output/'normalization.json', normalization)
        plot_comparisons(summary, fractions, agents, condition, config['modes'], output)
        plot_stratum_ranks(stratum_fractions, agents, condition, config['modes'], output)
        if any(file_hash(path) != hashes[name] for name, path in sources.items()):
            raise ValueError('An original evaluation artifact changed during analysis')
        write_json(output/'manifest.json', dict(
            created_at=datetime.now(ZoneInfo('America/New_York')).isoformat(),
            run=str(run), sources={name: dict(path=str(path), sha256=hashes[name]) for name, path in sources.items()},
            script_sha256=file_hash(__file__), agents=agents, tasks=len(rows),
            paired_settings=len(paired), evaluation_strata=len(stratum_ranks)//(len(agents)*len(METRICS)),
            correlation_method='Pearson; pooled tasks and pooled model/mode/domain/difficulty means',
            correlation_rewards=list(CORRELATION_REWARDS),
            correlation_reward_normalization='Exact Reward/training-adapter rules; evaluation metrics retain shared graph normalization',
            primary_metrics=dict(METRICS), packages={name: version(name) for name in ('numpy', 'networkx', 'pymoo', 'moocore', 'matplotlib')},
            combined_formula='V + V*(IGD_plus_score + hypervolume_ratio + min_jaccard_distance)',
            normalization='Shared per-objective min/max over all agents/modes and saved distinct Pareto front, per graph',
            pareto_membership_definition='Unique valid returned ordered paths in saved Pareto path set divided by requested k',
            jaccard_representation='directed edge sets', tie_policy='fractional credit across occupied rank positions',
            tie_tolerance=TIE_TOLERANCE, graph_truth='Original saved Pareto path membership; never recomputed',
            checkpoints=json.loads(sources['evaluation_manifest.json'].read_text())['checkpoints'],
            python=sys.executable))
        print(f'Exported {len(rows)} task scores, {len(rank_rows)} rank rows and '
              f'{len(list((output/"plots").rglob("*.png")))} bar-chart figures (PNG and SVG).')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.run, args.output)
