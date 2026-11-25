"""Observation-limited exploration and explicitly privileged upper bounds."""
import random
from .environment import Environment
from .oracle import dominates


def discovered_frontier(archive):
    return [p for p,c in sorted(archive.items()) if not any(dominates(other,c) for other in archive.values())]


def baseline(instance, k=1, method='random', seed=0, budget=None, objective='pareto'):
    env, rng, visits = Environment(instance,k,budget,objective), random.Random(seed), {}
    while env.phase=='traversal':
        # Exploration consumes only currently visible choices and local visit counts.
        options = env.choices()
        if method != 'random':
            counts = [visits.get((env.current,a['move_to']),0) for a in options]
            options = [a for a,c in zip(options,counts) if c==min(counts)]
        action = rng.choice(options)
        edge = env.current,action['move_to']
        visits[edge] = visits.get(edge,0)+1
        env.step(action)
    if method=='privileged_selection':
        selected = sorted(set(env.archive)&instance.pareto)
    else:
        selected = discovered_frontier(env.archive)
        rng.shuffle(selected)
    env.step(dict(paths=[list(p) for p in selected[:k]]+[[]]*max(0,k-len(selected))))
    return env
