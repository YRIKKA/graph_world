"""Small-DAG correctness oracle; all objectives are minimized independently."""
from itertools import islice
import networkx as nx


def dominates(x, y):
    return all(a <= b for a, b in zip(x, y)) and any(a < b for a, b in zip(x, y))


def path_cost(instance, path):
    return tuple(sum(instance.graph[u][v].get(name, 1) for u, v in zip(path, path[1:]))
                 for name in instance.names)


def frontier(instance, paths):
    # Filter paths, rather than unique vectors: equal-cost node sequences survive.
    costs = {tuple(p): path_cost(instance, p) for p in paths}
    return {p for p, c in costs.items() if not any(dominates(other, c) for other in costs.values())}


def enumerate_paths(instance, limit=4096):
    paths = list(islice(nx.all_simple_paths(instance.graph, instance.source, instance.target), limit + 1))
    if len(paths) > limit:
        raise ValueError("path_limit")
    return paths


def hypervolume(points, reference):
    """Exact box-union volume by recursive coordinate slicing (small sets only)."""
    points = sorted(set(tuple(p) for p in points if all(a < b for a, b in zip(p, reference))))
    if not points:
        return 0
    if len(reference) == 1:
        return reference[0] - min(p[0] for p in points)
    cuts = sorted({p[0] for p in points} | {reference[0]})
    return sum((hi - lo) * hypervolume([p[1:] for p in points if p[0] <= lo], reference[1:])
               for lo, hi in zip(cuts, cuts[1:]))
