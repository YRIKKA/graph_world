"""Selectable Dijkstra, exhaustive, and paretoset MOSP solvers.

Reference: Maristany de las Casas et al., Algorithm 1 and Section 2.6:
https://arxiv.org/abs/2110.10978
Reference implementation: https://github.com/maristanyPedro/targetedMosp

All objectives are minimized. Each method returns [(node_path, cost_tuple), ...]
with every simple Pareto-optimal path, including routes with identical costs,
sorted lexicographically by cost vector.
Use weight="cost" for a vector-valued edge attribute, or weight=("time", "distance")
for separate scalar attributes. A scalar attribute also supports one objective.
Costs must be finite and nonnegative. Round each edge objective to cost_precision
decimal places (default 12, ties to even), then use scaled Python integers for
all comparisons and additions. Integer costs remain exact. Returned costs are
ordinary numbers in the original units; the input graph is never rounded in place.
Graphs must be DiGraphs, without parallel arcs.

The implementation keeps T-MDA's one active queue path per node and sorted
per-arc deferred lists. It uses full dominance checks and parent pointers for
clarity, omitting the reference code's dimensionality and memory optimizations.
T-MDA first finds distinct Pareto vectors, then recovers all simple routes that
achieve them, pruning prefixes with its target potentials. Exhaustive enumeration
is intended only for small graphs; pass previously enumerated paths to reuse them.
Either method may return exponentially many paths, even for a single cost vector.

Example::

    paths = paretoset_paths(graph, source, target,
                                            weight=("distance", "time"))

Ground-truth generation selects a solver through solve_pareto_paths (default: exhaustive).
Run this file for a randomized solver demonstration.
"""

import argparse
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
import heapq
from itertools import count
from time import perf_counter

import networkx as nx
import numpy as np
from paretoset import paretoset


DEFAULT_COST_PRECISION = 12
PARETO_SOLVERS = ("dijkstra", "exhaustive", "paretoset")


def validate_cost_precision(cost_precision):
    """Return a nonnegative integer number of decimal places."""
    # Reject booleans and fractional settings before preparing any edge costs.
    if (isinstance(cost_precision, (bool, np.bool_))
            or not isinstance(cost_precision, (int, np.integer)) or cost_precision < 0):
        raise ValueError("cost_precision must be a nonnegative integer")
    return int(cost_precision)


def cost_to_units(value, cost_precision=DEFAULT_COST_PRECISION):
    """Round a nonnegative cost to integer units of 10 ** -cost_precision."""
    # Scale integers directly so even large integer costs retain their precision.
    scale = 10 ** validate_cost_precision(cost_precision)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Costs must be finite nonnegative numbers")
    if isinstance(value, (int, np.integer)):
        if value < 0:
            raise ValueError("Costs must be finite nonnegative numbers")
        return int(value) * scale
    if not isinstance(value, (float, np.floating)) or not np.isfinite(value) or value < 0:
        raise ValueError("Costs must be finite nonnegative numbers")

    # Interpret the float's decimal spelling and round exactly, with ties to even.
    numerator, denominator = Decimal(str(value)).as_integer_ratio()
    units, remainder = divmod(numerator * scale, denominator)
    return units + int(2 * remainder > denominator or (2 * remainder == denominator and units % 2))


def cost_from_units(units, cost_precision=DEFAULT_COST_PRECISION):
    """Convert an integer total back to its original units for reporting."""
    # Keep integral totals as Python integers; only fractional totals need floats.
    scale = 10 ** validate_cost_precision(cost_precision)
    if units % scale == 0:
        return units // scale
    try:
        result = units / scale
    except OverflowError:
        raise ValueError("Path cost is too large to report as a finite float") from None
    if not np.isfinite(result):
        raise ValueError("Path cost is too large to report as a finite float")
    return result


def _prepare_graph(graph, source, target, weight, cost_precision):
    """Validate input and copy edge costs to outgoing/incoming dictionaries."""
    # Keep one unambiguous integer cost vector for each directed arc.
    validate_cost_precision(cost_precision)
    if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
        raise TypeError("Expected a NetworkX DiGraph without parallel edges")
    if source not in graph or target not in graph:
        raise nx.NodeNotFound("Both source and target must be present in the graph")

    # A string selects a vector attribute; a sequence selects scalar attributes.
    attributes = None if isinstance(weight, str) else tuple(weight)
    if attributes is not None and (not attributes or any(not isinstance(name, str) for name in attributes)):
        raise ValueError("weight must be an attribute name or a nonempty sequence of names")
    dimension = None if attributes is None else len(attributes)
    outgoing = {node: {} for node in graph}
    incoming = {node: {} for node in graph}
    for parent, child, data in graph.edges(data=True):
        # Reject missing costs instead of silently substituting unit weights.
        try:
            raw = data[weight] if attributes is None else [data[name] for name in attributes]
        except KeyError as exc:
            raise ValueError(f"Missing cost attribute {exc.args[0]!r} on edge {(parent, child)!r}") from None

        # Normalize vectors once and preserve integer arithmetic where possible.
        vector = np.asarray(raw, dtype=object)
        if vector.ndim == 0:
            vector = vector.reshape(1)
        if vector.ndim != 1 or not vector.size:
            raise ValueError(f"Edge {(parent, child)!r} needs finite nonnegative scalar costs or a 1D cost vector")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError("Every edge must have the same number of objectives")
        cost = tuple(cost_to_units(value, cost_precision) for value in vector.tolist())
        outgoing[parent][child] = cost
        incoming[child][parent] = cost

    # An edgeless vector-attribute graph defaults to one zero-cost objective.
    return outgoing, incoming, dimension if dimension is not None else 1


def _add(left, right):
    # Sum each objective separately; path costs are additive vectors.
    return tuple(a + b for a, b in zip(left, right))


def _dominates(left, right):
    # The vector search includes equality to prune duplicate vectors and zero-cost cycles.
    return all(a <= b for a, b in zip(left, right))


def _preprocess(incoming, target, dimension):
    """Compute ideal-point potentials using reverse lexicographic Dijkstra."""
    # Rotate the objective order so each objective leads exactly one search.
    searches = []
    serial = count()
    zero = (0,) * dimension
    for first in range(dimension):
        order = tuple(range(first, dimension)) + tuple(range(first))
        distances = {target: zero}
        queue = [(zero, next(serial), target, zero)]
        while queue:
            _, _, node, cost = heapq.heappop(queue)
            # Ignore entries superseded by a lexicographically smaller vector.
            if distances[node] != cost:
                continue
            for parent, edge_cost in incoming[node].items():
                candidate = _add(cost, edge_cost)
                key = tuple(candidate[index] for index in order)
                previous = distances.get(parent)
                if previous is None or key < tuple(previous[index] for index in order):
                    distances[parent] = candidate
                    # Serial numbers prevent comparisons between arbitrary node names.
                    heapq.heappush(queue, (key, next(serial), parent, candidate))
        searches.append(distances)

    # Each component is a lower bound on the remaining cost to the target.
    potential = {
        node: tuple(searches[index][node][index] for index in range(dimension))
        for node in searches[0]
    }
    return potential, searches


@dataclass(eq=False)
class _Label:
    # Parent pointers retain paths without copying a full node list on every arc.
    node: object
    cost: tuple
    previous: "_Label | None" = None


def targeted_multiobjective_dijkstra(graph, source, target, weight="cost", *,
                                     cost_precision=DEFAULT_COST_PRECISION):
    """Find Pareto vectors with T-MDA, then return all simple routes realizing them.

    Supports any positive number of minimizing objectives, arbitrary hashable
    node names, nonnegative cycles, and zero-cost edges. Returns [] if unreachable.
    Quantize edge costs to cost_precision decimal places before searching.
    The input graph and its attributes are not modified.
    """
    # Build a dictionary representation and initialize target-directed potentials.
    outgoing, incoming, dimension = _prepare_graph(graph, source, target, weight, cost_precision)
    potential, searches = _preprocess(incoming, target, dimension)
    if source not in potential:
        return []

    # Use the preprocessing paths to obtain a safe initial dominance bound.
    # Strict comparison avoids pruning equal-cost solutions or choosing epsilon.
    bound = tuple(max(search[source][index] for search in searches) for index in range(dimension))
    permanent = {node: [] for node in potential}
    deferred = {(parent, child): deque() for parent in potential for child in outgoing[parent] if child in potential}
    active, queue, solutions = {}, [], []
    serial = count()

    def rejected(label):
        # Discard dominated/equivalent prefixes at the same node.
        if any(_dominates(cost, label.cost) for cost in permanent[label.node]):
            return True
        estimate = _add(label.cost, potential[label.node])
        # Discard prefixes whose optimistic completion is dominated at the target.
        return (any(_dominates(cost, estimate) for cost in permanent[target])
                or all(value > limit for value, limit in zip(estimate, bound)))

    def activate(label):
        # Only one label per node is active; stale heap entries are ignored later.
        active[label.node] = label
        estimate = _add(label.cost, potential[label.node])
        heapq.heappush(queue, (estimate, next(serial), label))

    # Start with the zero-cost source path, prioritized by its target potential.
    activate(_Label(source, (0,) * dimension))
    while queue:
        _, _, label = heapq.heappop(queue)
        node = label.node
        if active.get(node) is not label:
            continue
        del active[node]

        # Recheck pruning because another target path may have become permanent.
        keep = not rejected(label)
        if keep:
            permanent[node].append(label.cost)
            if node == target:
                solutions.append(label)

        # nextQueuePath: discard obsolete list heads and select the smallest
        # surviving candidate over all incoming arcs of this node.
        next_label, next_list = None, None
        for parent in incoming[node]:
            candidates = deferred.get((parent, node))
            if candidates is None:
                continue
            while candidates and rejected(candidates[0]):
                candidates.popleft()
            if candidates and (next_label is None or candidates[0].cost < next_label.cost):
                next_label, next_list = candidates[0], candidates
        if next_label is not None:
            next_list.popleft()
            activate(next_label)

        # Target paths are complete; dominated prefixes cannot yield new results.
        if not keep or node == target:
            continue
        for child, edge_cost in outgoing[node].items():
            if child not in potential:
                continue  # A node with no route to the target cannot help.
            candidate = _Label(child, _add(label.cost, edge_cost), label)
            if rejected(candidate):
                continue
            current = active.get(child)
            if current is None:
                activate(candidate)
            elif candidate.cost < current.cost:
                # propagate/decreaseKey: defer the displaced path at its arc head.
                deferred[(current.previous.node, child)].appendleft(current)
                activate(candidate)
            else:
                # Permanent predecessor labels arrive in lexicographic order,
                # so appending preserves the per-arc nextQueuePath ordering.
                deferred[(node, child)].append(candidate)

    # The vector search prunes equal prefixes; recover their distinct routes now.
    target_costs = {label.cost for label in solutions}
    results = []
    stack = [([source], (0,) * dimension)]
    while stack:
        path, cost = stack.pop()
        node = path[-1]
        if node == target:
            if cost in target_costs:
                results.append((path, cost))
            continue
        for child, edge_cost in outgoing[node].items():
            # Restrict recovery to simple paths, including when zero-cost cycles exist.
            if child in path or child not in potential:
                continue
            candidate = _add(cost, edge_cost)
            estimate = _add(candidate, potential[child])
            # A prefix can help only if its lower bound fits some Pareto vector.
            if any(_dominates(estimate, target_cost) for target_cost in target_costs):
                stack.append((path + [child], candidate))

    # Convert units only after all exact dominance and route checks have finished.
    return [(path, tuple(cost_from_units(value, cost_precision) for value in cost))
            for path, cost in sorted(results, key=lambda result: result[1])]


def exhaustive_enumeration(graph, source, target, weight="cost", *, paths=None,
                           cost_precision=DEFAULT_COST_PRECISION):
    """Return every simple Pareto-optimal path, including equal-cost alternatives.

    Supply an iterable of paths to avoid enumerating the graph again; include all
    simple source-target paths to obtain the complete frontier. An empty iterable
    returns no results. Quantize edge costs to cost_precision decimal places.
    """
    # Apply exactly the same input and cost conventions as the targeted method.
    outgoing, _, dimension = _prepare_graph(graph, source, target, weight, cost_precision)
    if paths is None:
        paths = [[source]] if source == target else nx.all_simple_paths(graph, source, target)
    frontier = {}
    for path in paths:
        # Validate reusable paths and copy them so callers retain their own lists.
        path = list(path)
        if (not path or path[0] != source or path[-1] != target
                or len(set(path)) != len(path)
                or any(child not in outgoing.get(parent, {}) for parent, child in zip(path, path[1:]))):
            raise ValueError("paths must contain simple directed source-target paths")
        # Compute each enumerated path's complete additive cost vector.
        cost = (0,) * dimension
        for parent, child in zip(path, path[1:]):
            cost = _add(cost, outgoing[parent][child])
        # Equal vectors retain every distinct route rather than dominating each other.
        if cost in frontier:
            frontier[cost][tuple(path)] = path
            continue
        if any(_dominates(known_cost, cost) for known_cost in frontier):
            continue
        # Remove every route associated with a newly dominated cost vector.
        frontier = {known_cost: routes for known_cost, routes in frontier.items()
                    if not _dominates(cost, known_cost)}
        frontier[cost] = {tuple(path): path}
    return [(path, tuple(cost_from_units(value, cost_precision) for value in cost))
            for cost, routes in sorted(frontier.items()) for path in routes.values()]



def paretoset_paths(graph, source, target, weight="cost", *, paths=None,
                    cost_precision=DEFAULT_COST_PRECISION):
    """Return all simple Pareto-optimal paths using the paretoset package.

    Reuse pre-enumerated paths when supplied. Preserve distinct routes with equal
    cost vectors, exact integer/quantized-decimal costs, and the legacy return
    format: [(node_list, cost_tuple), ...], ordered by increasing cost vector.
    """
    # Reuse cost validation and quantization, without invoking either legacy solver.
    outgoing, _, dimension = _prepare_graph(graph, source, target, weight, cost_precision)
    if paths is None:
        # The package filters objective vectors; NetworkX supplies feasible paths.
        paths = [[source]] if source == target else nx.all_simple_paths(graph, source, target)
    routes = {}
    for supplied in paths:
        # Copy inputs and reject malformed cached paths before computing any costs.
        path = list(supplied)
        if (not path or path[0] != source or path[-1] != target
                or len(set(path)) != len(path)
                or any(child not in outgoing.get(parent, {}) for parent, child in zip(path, path[1:]))):
            raise ValueError("paths must contain simple directed source-target paths")
        # Sum objectives separately in exact integer units, avoiding float roundoff.
        cost = (0,) * dimension
        for parent, child in zip(path, path[1:]):
            cost = _add(cost, outgoing[parent][child])
        # Repeated submissions of one route collapse; different equal-cost routes survive.
        routes[tuple(path)] = (path, cost)
    if not routes:
        return []

    candidates = list(routes.values())
    ranks = np.empty((len(candidates), dimension), dtype=np.int64)
    for objective in range(dimension):
        # paretoset rejects object arrays, while exact scaled costs may exceed int64.
        # Per-objective ordinal ranks preserve every ordering and equality exactly;
        # only these ranks enter filtering, and original cost units remain untouched.
        values = sorted({cost[objective] for _, cost in candidates})
        order = {value: index for index, value in enumerate(values)}
        ranks[:, objective] = [order[cost[objective]] for _, cost in candidates]

    # Minimize all objectives; distinct=False retains every path tied on optimal costs.
    # The NumPy backend avoids optional Numba compilation during graph generation.
    mask = paretoset(ranks, sense=["min"] * dimension, distinct=False, use_numba=False)
    selected = [candidate for candidate, keep in zip(candidates, mask) if keep]
    # Convert only for reporting, after exact Pareto selection has finished.
    return [(path, tuple(cost_from_units(value, cost_precision) for value in cost))
            for path, cost in sorted(selected, key=lambda item: item[1])]



def solve_pareto_paths(graph, source, target, weight="cost", *, pareto_solver="exhaustive",
                       paths=None, cost_precision=DEFAULT_COST_PRECISION):
    """Dispatch ground-truth computation to the selected exact Pareto solver."""
    # Keep a single selector for generators and transformed evaluation graphs.
    solvers = {"dijkstra": targeted_multiobjective_dijkstra,
               "exhaustive": exhaustive_enumeration, "paretoset": paretoset_paths}
    if pareto_solver not in PARETO_SOLVERS:
        raise ValueError("pareto_solver must be 'dijkstra', 'exhaustive', or 'paretoset'")
    kwargs = dict(weight=weight, cost_precision=cost_precision)
    if pareto_solver != "dijkstra":
        # Enumerating/filtering solvers can reuse routes; Dijkstra searches on its own.
        kwargs['paths'] = paths
    return solvers[pareto_solver](graph, source, target, **kwargs)


def _random_test_graph(rng):
    # Use 20 nodes with three distinct outgoing arcs per node (60 arcs total).
    nodes = ["s", *[f"n{i}" for i in range(1, 19)], "t"]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)

    # Reserve a randomized backbone so source-to-target reachability is guaranteed.
    route = ["s", *rng.permutation(nodes[1:-1]).tolist(), "t"]
    successor = dict(zip(route, route[1:]))
    for parent in nodes:
        children = [successor[parent]] if parent in successor else []
        # Exclude self-loops and the reserved arc before sampling without replacement.
        candidates = [node for node in nodes if node != parent and node not in children]
        children.extend(rng.choice(candidates, size=3 - len(children), replace=False).tolist())
        for child in children:
            # Positive three-objective costs are independent of the random topology.
            cost = tuple(rng.integers(1, 11, size=3).tolist())
            graph.add_edge(parent, child, cost=cost)
    return graph


def main(n_replicates=100):
    if type(n_replicates) is not int or n_replicates < 1:
        raise ValueError("n_replicates must be a positive integer")
    rng = np.random.default_rng()
    elapsed = 0.0
    for replicate in range(n_replicates):
        # Exercise only the active package-backed solver; legacy methods stay unused.
        graph = _random_test_graph(rng)
        started = perf_counter()
        results = paretoset_paths(graph, "s", "t")
        elapsed += perf_counter() - started
        if replicate == 0:
            print(f"Random graph: {len(graph)} nodes, {graph.number_of_edges()} edges, 3 objectives")
            for path, cost in results:
                print(f"  {path}: {cost}")
            print(f"Pareto set: {len(results)} paths, {len({cost for _, cost in results})} distinct cost vectors")
    print(f"paretoset_paths: {n_replicates} random instances in {elapsed:.6f} seconds")


if __name__ == "__main__":
    # Allow direct/module execution with an optional replicate-count override.
    parser = argparse.ArgumentParser(description="Demonstrate paretoset-based Pareto path filtering")
    parser.add_argument("--n-replicates", type=int, default=100,
                        help="Number of random graphs (default: 100)")
    args = parser.parse_args()
    if args.n_replicates < 1:
        parser.error("--n-replicates must be a positive integer")
    main(n_replicates=args.n_replicates)
