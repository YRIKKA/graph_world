"""Serializable terminal-reward conditions and their GRPO rollout adapter."""
from itertools import combinations

BASES = ('valid_per_path', 'valid')
SIGNALS = ('hypervolume', 'igd')
COMBINATIONS = BASES + tuple(
    '+'.join((base, *subset)) + '__weighted'
    for base in BASES
    for size in range(1, len(SIGNALS) + 1)
    for subset in combinations(SIGNALS, size)
)


def functions_for(condition):
    if condition not in COMBINATIONS:
        raise ValueError(f'Unknown MOO reward condition: {condition}')
    return ['reward_' + name for name in condition.split('__')[0].split('+')]


def options(condition, backend='pymoo'):
    if backend not in ('pymoo', 'moocore'):
        raise ValueError('Unknown hypervolume backend')
    return dict(functions=functions_for(condition), normalize=True,
                weight_by_valid=True, backend=backend)


def validate(config):
    for condition in COMBINATIONS:
        if config == options(condition, config.get('backend', 'pymoo')):
            return condition
    raise ValueError('MOO rewards require canonical functions, normalize=true and weight_by_valid=true')


def apply_rewards(tasks, instance, k, config):
    """Replace training returns only; retain environment S and its diagnostics."""
    from ..oracle import path_cost
    from ..rewards import Reward

    validate(config)
    graph = instance.graph.copy()
    for _, _, edge in graph.edges(data=True):
        edge['cost'] = tuple(edge.get(name, 1) for name in instance.names)
    frontier = {tuple(path): path_cost(instance, path) for path in instance.pareto}
    # Nonnegative additive DAG costs: no simple path can exceed n-1 edges.
    reference = [max((edge['cost'][i] for _, _, edge in graph.edges(data=True)), default=0)
                 * (len(graph) - 1) + 1 for i in range(len(instance.names))]
    for task in tasks:
        original = task.env.reward
        submission = task.env.submission
        paths = submission.get('paths') if isinstance(submission, dict) and set(submission) == {'paths'} else None
        wellformed = (isinstance(paths, list) and len(paths) <= k
                      and all(isinstance(path, list) and all(isinstance(v, str) for v in path) for path in paths))
        components = dict.fromkeys(config['functions'], 0.0)
        contributions = dict.fromkeys(config['functions'], 0.0)
        if task.status != 'ok' or not wellformed:
            total = 0.0
        else:
            reward = Reward(graph, instance.source, instance.target, k, paths, frontier)
            selected = [getattr(reward, name) for name in config['functions']]
            kwargs = dict(normalize=True, weight_by_valid=config['weight_by_valid'])
            if 'reward_hypervolume' in config['functions']:
                kwargs.update(reference_point=reference, backend=config['backend'])
            if 'reward_igd' in config['functions']:
                kwargs.update(normalize_igd=True, normalize_costs=True)
            total = reward.terminal_reward_additive(selected, **kwargs)
            components = reward.reward_components
            contributions = reward.reward_contributions
        task.reward = total
        task.shaping = dict(terminal_reward=original, moo_rewards=config, total_reward=total,
                            reward_components=components, reward_contributions=contributions)
    return tasks
