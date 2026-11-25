"""Composable terminal rewards for path-selection experiments."""

from collections.abc import Callable, Collection, Hashable, Sequence
from functools import partial
from inspect import signature
from numbers import Integral, Real

import networkx as nx
import numpy as np


class Reward:
    """Store a DAG, candidate paths, and the Pareto frontier for reward scoring."""

    def __init__(
        self,
        graph: nx.DiGraph,
        source: Hashable,
        target: Hashable,
        k: int,
        paths: Collection[Sequence[Hashable]],
        pareto_frontier: dict[tuple[Hashable, ...], tuple[int | float, ...] | list[int | float]],
        doctrine: Callable[[list[Sequence[Hashable]]], float] | None = None,
    ):
        self.graph = graph
        self.source = source
        self.target = target
        self.k = k
        self.paths = paths
        self.pareto_frontier = pareto_frontier
        self.doctrine = doctrine

        if self.doctrine is not None and not callable(self.doctrine):
            raise TypeError("doctrine must be callable or None.")

        if not isinstance(self.graph, nx.DiGraph):
            raise TypeError("graph must be a networkx directed graph.")
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("graph must be a directed acyclic graph.")
        if isinstance(self.k, bool) or not isinstance(self.k, Integral) or self.k <= 0:
            raise ValueError("k must be a positive integer.")
        if len(self.paths) > self.k:
            raise ValueError(f"The path list cannot be longer than length k (k={self.k}).")
        if self.source not in self.graph or self.target not in self.graph:
            raise ValueError("Both source and target must be in the graph.")

        # NetworkX: A[i, j] represents the edge i -> j, so rows are parents
        # and columns are children. A node's column lists its incoming parents;
        # its row lists its outgoing children. Both axes follow node iteration order.
        # https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.to_numpy_array.html
        self.adjacency_matrix = nx.to_numpy_array(self.graph)

        self.valid_paths = []
        self.invalid_paths = []
        for path in self.paths:
            if self.is_valid_path(path):
                self.valid_paths.append(path)
            else:
                self.invalid_paths.append(path)


        # Cache costs in the same order as valid_paths, retaining duplicates.
        self.valid_costs = []
        cost_dimension = None
        for path in self.valid_paths:
            total = None
            for u, v in zip(path, path[1:]):
                if 'cost' not in self.graph[u][v]:
                    raise ValueError(f"Edge {(u, v)!r} is missing its cost vector.")
                edge_cost = np.asarray(self.graph[u][v]['cost'], dtype=float)
                if (edge_cost.ndim != 1 or edge_cost.size == 0
                        or not np.all(np.isfinite(edge_cost))):
                    raise ValueError("Edge cost vectors must be nonempty and finite.")
                if cost_dimension is None:
                    cost_dimension = edge_cost.size
                if edge_cost.size != cost_dimension:
                    raise ValueError("Edge cost vectors must have the same dimension.")
                if total is None:
                    total = np.zeros(cost_dimension)
                total += edge_cost
            self.valid_costs.append(total)

        # A single-node path has zero cost. Infer its dimension from the front
        # when no traversed edge supplies one; default to one objective otherwise.
        if cost_dimension is None:
            cost_dimension = len(next(iter(self.pareto_frontier.values()), (0,)))
        self.valid_costs = [np.zeros(cost_dimension) if cost is None else cost
                            for cost in self.valid_costs]
        self.valid_costs_normalized = None
        self.pareto_front_normalized = None
        self.reward_components = {}
        self.reward_contributions = {}

    def terminal_reward_additive(
        self,
        reward_functions: Collection[Callable[..., float]],
        normalize: bool = True,
        weight_by_valid: bool = True,
        **reward_kwargs,
    ) -> float:
        """Sum selected reward methods bound to this instance.

        Pass a collection such as [self.reward_valid, self.reward_hypervolume].
        Extra keywords (e.g. reference_point, backend, front) are forwarded only
        to selected methods that accept them. The normalize flag is forwarded
        to methods with a parameter named normalize; IGD's normalize_igd and
        normalize_costs can be supplied separately through reward_kwargs.
        When weight_by_valid is True, multiply each term by reward_valid(),
        except reward_valid and reward_valid_per_path, which remain unweighted.
        An empty collection returns zero. Raw terms and their weighted additive
        contributions are saved in reward_components and reward_contributions.
        """
        allowed = (self.reward_valid_per_path, self.reward_valid,
                   self.reward_hypervolume, self.reward_doctrine, self.reward_igd)
        calls = []
        used_kwargs = set()
        for reward_function in reward_functions:
            if reward_function not in allowed:
                raise ValueError("Only supported reward methods bound to this Reward instance are allowed.")
            parameters = signature(reward_function).parameters
            kwargs = {key: value for key, value in reward_kwargs.items() if key in parameters}
            used_kwargs.update(kwargs)
            if 'normalize' in parameters:
                kwargs['normalize'] = normalize
            signature(reward_function).bind(**kwargs)
            calls.append((reward_function, kwargs))
        unknown = set(reward_kwargs) - used_kwargs
        if unknown:
            raise TypeError(f"Unsupported arguments for the selected rewards: {sorted(unknown)}")
        weight = self.reward_valid() if weight_by_valid else 1.0
        unweighted = (self.reward_valid, self.reward_valid_per_path)
        components, contributions = {}, {}
        total = 0.0
        for function, kwargs in calls:
            value = float(function(**kwargs))
            contribution = value * (1.0 if function in unweighted else weight)
            name = function.__name__
            components[name] = components.get(name, 0.0) + value
            contributions[name] = contributions.get(name, 0.0) + contribution
            total += contribution
        self.reward_components = components
        self.reward_contributions = contributions
        return float(total)

    def is_valid_path(self, path: Sequence[Hashable]) -> bool:
        """Check endpoints, node membership, uniqueness, and directed edges.

        A single-node path is valid when source and target are that same node.
        Invalid path contents return False.
        """
        try:
            nodes = tuple(path)
            return bool(
                nodes
                and nodes[0] == self.source
                and nodes[-1] == self.target
                and len(set(nodes)) == len(nodes)
                and all(node in self.graph for node in nodes)
                and all(self.graph.has_edge(u, v) for u, v in zip(nodes, nodes[1:]))
            )
        except (TypeError, ValueError):
            return False

    def reward_valid_per_path(self) -> int:
        """Return the number of unique valid paths."""
        return len({tuple(path) for path in self.valid_paths})

    def reward_valid(self) -> float:
        """Return the fraction of k slots with distinct valid source-target paths.

        Missing paths and repeated copies of a path contribute zero. Use a list
        of paths to preserve duplicate submissions; sets already remove duplicates.
        """
        unique_valid_count = len({tuple(path) for path in self.valid_paths})
        duplicate_count = len(self.valid_paths) - unique_valid_count
        missing_count = self.k - len(self.valid_paths) - len(self.invalid_paths)
        invalid_count = len(self.invalid_paths) + duplicate_count + missing_count
        return (self.k - invalid_count) / self.k

    def reward_hypervolume(
        self, reference_point: Sequence[int | float], normalize: bool = True,
        backend: str = "pymoo",
    ) -> float:
        """Compute valid-path hypervolume, optionally relative to the Pareto front.

        All objectives are minimized. Cached valid-path cost vectors must match
        the reference point; costs are summed elementwise during initialization. The same
        reference point is used for both hypervolumes. Empty valid-path sets have
        zero reward immediately. Otherwise, normalization requires a positive Pareto hypervolume.
        Select backend="pymoo" (default) or backend="moocore" for both calculations.
        """
        if not self.valid_paths:
            return 0.0
        if backend not in ("pymoo", "moocore"):
            raise ValueError('backend must be "pymoo" or "moocore".')
        reference = np.asarray(reference_point, dtype=float)
        if reference.ndim != 1 or reference.size == 0 or not np.all(np.isfinite(reference)):
            raise ValueError("reference_point must be a nonempty finite vector.")

        if backend == "pymoo":
            from pymoo.indicators.hv import HV

            hypervolume = HV(ref_point=reference)
        else:
            import moocore

            hypervolume = partial(moocore.hypervolume, ref=reference)

        costs = np.asarray(self.valid_costs, dtype=float)
        if self.valid_costs and costs.shape[1] != reference.size:
            raise ValueError("Cached path cost vectors must match reference_point.")
        volume = float(hypervolume(costs)) if self.valid_costs else 0.0
        if not normalize:
            return volume

        frontier_costs = np.asarray(list(self.pareto_frontier.values()), dtype=float)
        if (frontier_costs.ndim != 2 or frontier_costs.shape[1] != reference.size
                or not np.all(np.isfinite(frontier_costs))):
            raise ValueError("pareto_frontier must contain finite cost vectors matching reference_point.")
        frontier_volume = float(hypervolume(frontier_costs))
        if frontier_volume <= 0:
            raise ValueError("Cannot normalize by a zero Pareto-front hypervolume.")
        return volume / frontier_volume

    def reward_doctrine(self) -> float:
        """Return zero for no valid paths or no doctrine; otherwise evaluate doctrine.

        The callback must return a numeric score between 0 and 1, inclusive.
        Valid paths are passed as classified, including duplicate submissions.
        """
        if not self.valid_paths or self.doctrine is None:
            return 0.0
        score = self.doctrine(self.valid_paths)
        if not isinstance(score, Real) or not 0 <= score <= 1:
            raise ValueError("doctrine must return a number between 0 and 1, inclusive.")
        return float(score)

    def reward_igd(
        self,
        front: Collection[Sequence[int | float]] | None = None,
        normalize_igd: bool = True,
        normalize_costs: bool = True,
    ) -> float:
        """Compute IGD+, optionally returning 1 - IGD+ / sqrt(objective count).

        Use the stored Pareto frontier when front is None. Cost normalization
        uses shared per-objective minima and maxima over the candidates and
        chosen front, mapping constant objectives to zero. Store the resulting
        arrays without changing the original cost vectors. Both sets must be
        nonempty and contain finite vectors of the same dimension. Return zero
        immediately when there are no valid paths.
        """
        if not self.valid_paths:
            return 0.0

        from pymoo.indicators.igd_plus import IGDPlus

        chosen_front = self.pareto_frontier.values() if front is None else front
        front_costs = np.asarray(list(chosen_front), dtype=float)
        costs = np.asarray(self.valid_costs, dtype=float)
        if (front_costs.ndim != 2 or 0 in front_costs.shape
                or not np.all(np.isfinite(front_costs))):
            raise ValueError("front must contain nonempty finite cost vectors.")
        if (costs.ndim != 2 or 0 in costs.shape
                or costs.shape[1] != front_costs.shape[1]
                or not np.all(np.isfinite(costs))):
            raise ValueError("valid_costs must be nonempty, finite, and match the front dimension.")

        if normalize_costs:
            combined = np.vstack((costs, front_costs))
            minimum = combined.min(axis=0)
            scale = combined.max(axis=0) - minimum
            scale[scale == 0] = 1.0
            self.valid_costs_normalized = (costs - minimum) / scale
            self.pareto_front_normalized = (front_costs - minimum) / scale
            costs = self.valid_costs_normalized
            front_costs = self.pareto_front_normalized

        igd = float(IGDPlus(front_costs)(costs))
        return float(1 - igd / np.sqrt(costs.shape[1])) if normalize_igd else igd


def main(iterations: int = 5, seed: int | None = None, k: int = 5) -> None:
    """Print randomized small-DAG examples and check normalized reward bounds."""
    from pprint import pprint

    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer.")
    rng = np.random.default_rng(seed)
    print("Normalized reward demo (all objectives are minimized).")
    print("reward_valid_per_path is a count and may exceed 1; the additive sum may also exceed 1.")
    print("Cost normalization uses a shared min/max for each objective (each column).")
    print("Reward bounds checks allow 1e-12 tolerance for floating-point roundoff.")

    for iteration in range(1, iterations + 1):
        node_count = int(rng.integers(k + 3, 2 * k + 4))
        objective_count = int(rng.integers(2, 5))
        graph = nx.DiGraph()
        graph.add_nodes_from(range(node_count))
        scales = 10.0 ** np.arange(objective_count)
        source, target = 0, node_count - 1
        # Parallel two-edge routes guarantee more than k distinct valid paths
        # while keeping complete frontier enumeration small as k grows.
        for middle in range(1, target):
            for u, v in ((source, middle), (middle, target)):
                graph.add_edge(u, v, cost=np.round(rng.uniform(1, 10, objective_count) * scales, 2))

        all_costs = {
            tuple(path): np.sum([graph[u][v]['cost'] for u, v in zip(path, path[1:])], axis=0)
            for path in nx.all_simple_paths(graph, source, target)
        }
        frontier = {
            path: cost for path, cost in all_costs.items()
            if not any(np.all(other <= cost) and np.any(other < cost)
                       for other in all_costs.values())
        }
        all_paths = list(all_costs)
        # Exactly k distinct valid submissions contribute k cost vectors to
        # normalization, alongside the complete Pareto frontier.
        paths = [all_paths[int(index)] for index in rng.choice(len(all_paths), size=k, replace=False)]

        preferred_node = node_count // 2

        def doctrine(valid_paths):
            return sum(preferred_node in path for path in valid_paths) / len(valid_paths)

        reward = Reward(graph, source, target, k, paths, frontier, doctrine=doctrine)
        reference_point = np.max(list(all_costs.values()), axis=0) + 1
        scores = {
            'reward_valid': reward.reward_valid(),
            'reward_hypervolume': reward.reward_hypervolume(reference_point, normalize=True),
            'reward_doctrine': reward.reward_doctrine(),
            'reward_igd (IGD+)': reward.reward_igd(normalize_igd=True, normalize_costs=True),
        }
        total = reward.terminal_reward_additive(
            [reward.reward_valid, reward.reward_hypervolume, reward.reward_doctrine, reward.reward_igd],
            normalize=True, weight_by_valid=True, reference_point=reference_point,
            normalize_igd=True, normalize_costs=True,
        )
        print(f"\nIteration {iteration}/{iterations}: {node_count} nodes, {objective_count} objectives, k={k}")
        pprint({'submitted_paths': paths, 'valid_paths': reward.valid_paths,
                'invalid_paths': reward.invalid_paths, 'reference_point': reference_point.tolist()},
               sort_dicts=False)
        print("Valid path cost vectors: raw -> normalized")
        for path, raw, normalized in zip(reward.valid_paths, reward.valid_costs,
                                         reward.valid_costs_normalized):
            pprint({'path': path, 'raw': raw.tolist(), 'normalized': normalized.round(6).tolist()},
                   sort_dicts=False)
        print("Pareto front cost vectors: raw -> normalized")
        for (path, raw), normalized in zip(frontier.items(), reward.pareto_front_normalized):
            pprint({'path': path, 'raw': raw.tolist(), 'normalized': normalized.round(6).tolist()},
                   sort_dicts=False)
        for name, score in scores.items():
            assert np.isfinite(score) and -1e-12 <= score <= 1 + 1e-12, f"{name} out of bounds: {score}"
            print(f"  {name:<28} {score:.6f}  [0, 1] PASS")
        for costs in (reward.valid_costs_normalized, reward.pareto_front_normalized):
            assert np.all(np.isfinite(costs)) and np.all((0 <= costs) & (costs <= 1))
        print(f"  {'reward_valid_per_path (count)':<28} {reward.reward_valid_per_path()}")
        print(f"  {'terminal_reward_additive':<28} {total:.6f}")

    print(f"\nAll {iterations} iterations passed normalized cost and reward bounds checks.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Demonstrate rewards on random synthetic DAGs.")
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--k', type=int, default=5, help='Number of distinct valid paths per iteration.')
    args = parser.parse_args()
    main(iterations=args.iterations, seed=args.seed, k=args.k)
