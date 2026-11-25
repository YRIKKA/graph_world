"""Random directed graphs, named edge costs, plotting, and routing.

Requires numpy, networkx, and paretoset; plot_graph additionally requires matplotlib.
Run this file to save an example graph to graph.png.
"""

from collections import Counter
from pprint import pprint

import networkx as nx
import numpy as np

if __package__:
    from . import mosp
else:
    import mosp
    
'''
Notes on the pareto frontier.
-------------------------------------

Different `scalar_sum` values are expected: **Pareto-optimal paths represent tradeoffs between objectives; 
they do not necessarily minimize their sum.**

In [generate_graph_world.py:164](/home/ubuntu/IsaacSim/jm_test/graph_world/generate_graph_world.py:164), 
the computation works as follows:

1. Select the objectives from `self.edge_cost_names`, normally `water`, `time`, and `energy`.
2. Copy the graph and fill any missing objective attributes with `1`.
3. Call `mosp.solve_pareto_paths` with the selected pareto_solver (default: exhaustive).
4. Store each returned path alongside its objective totals. **`scalar_sum` is added afterward**, by summing those 
totals; it does not participate in selecting the paths.

The selected solver returns paths paired with nondominated objective vectors.
Exhaustive and paretoset reuse enumerated paths; Dijkstra searches the graph directly.

A path **Q dominates P** when:

- Q’s cost is less than or equal to P’s cost in **every** objective.
- Q’s cost is strictly smaller in **at least one** objective.

A path belongs to the Pareto frontier when no other feasible path dominates it.

For example, I verified this case against the current implementation:

| Path | Water | Time | Energy | `scalar_sum` | Retained? |
|---|---:|---:|---:|---:|---|
| A | 1 | 9 | 1 | 11 | Yes |
| B | 4 | 4 | 1 | 9 | Yes |
| C | 9 | 1 | 1 | 11 | Yes |
| D | 5 | 5 | 1 | 11 | No |

B has a smaller sum than A, but uses more water. B also has a smaller sum than C, but takes more time. 
Consequently, neither A nor C is dominated by B. D is dominated by B because B improves both water and 
time without increasing energy.

In contrast, `self.scalar_dijkstra` minimizes the sum with equal weights for all objectives. 
It selects B in this example. With consistent costs and exact arithmetic, its `scalar_sum` 
should equal the **minimum** `scalar_sum` among the Pareto results. The other Pareto paths may have larger sums.
'''


class GraphGenerator:
    """Generate, connect, visualize, and analyze directed graphs."""

    def __init__(self, 
                 n_subgraphs: int = 3, 
                 n_nodes_per_subgraph: int | tuple[int, int] = 5,
                 max_subgraph_outdegree: int | tuple[int, int] = 2,
                 themes: list[str] = ["physics", "spatial", "quantitative"],
                 edge_costs: bool = False,
                 edge_cost_names: list[str] = ["water", "time", "energy"],
                 edge_cost_types: list[type] = [int, int, int],
                 task_complexity: int | tuple[int, int] = (1, 10),
                 log_metadata: str | None = None,
                 cost_precision: int = mosp.DEFAULT_COST_PRECISION,
                 restriction: tuple[float, float, float] = (0.7, 0.2, 0.1),
                 mdp: bool = False,
                 seed: int | None = None,
                 max_paths: int | None = None,
                 pareto_solver: str = "exhaustive"):
        """Create subgraphs with balanced random node themes and combine them.

        Node counts and max_subgraph_outdegree accept integers or tuples (lower, upper).
        max_subgraph_outdegree must be at least 2, including a tuple's lower bound.
        Each node has at least two internal children when two later nodes exist;
        the penultimate node has one child and the final node is a sink.
        Inter-subgraph connectors may increase a node's out-degree by one.
        All generated subgraphs and the global graph are directed acyclic graphs.
        Tuple ranges are sampled independently for each subgraph, including
        both endpoints. Each sampled pair must satisfy make_subgraph's bounds.
        Assign balanced themes inside each subgraph before graph composition.
        Counts per distinct theme differ by at most one node in each subgraph
        and across the generated global graph. Duplicate strings count once.
        Inherit node attributes and collect them in node_themes. Then choose
        source and target uniformly among distinct reachable ordered pairs with
        the largest unweighted directed shortest-path distance (edge hops).
        At least one such pair must exist. Computing all distances takes
        O(V * (V + E)) time using breadth-first search from every node.
        seed controls every construction draw; max_paths bounds enumeration.
        mdp defaults to False: assign transition_prob to every global edge.
        Outgoing probabilities sum to 1, with values in [0.1, 0.9] for
        branching nodes and 1.0 for single-edge nodes. Sinks have no transitions.
        More than ten outgoing edges cannot satisfy the bounds and are rejected.
        restriction defaults to (0.7, 0.2, 0.1). Supply three probabilities for
        "none", "discouraged", and "forbidden", respectively. Each probability
        must be finite and in [0, 1], and their sum must be 1. The special tuple
        (0.0, 0.0, 0.0) skips the restriction attribute entirely. Otherwise,
        independently sample a categorical restriction for every global node.
        This annotation does not exclude nodes from path calculations.
        edge_costs defaults to False: leave generated edges without cost
        attributes and hide cost labels and legends when plotting, even if
        edge_cost is supplied. Path calculations retain unit-cost defaults.
        Set edge_costs=True to sample and display costs as configured below.
        edge_cost_names defaults to water, time, and energy. edge_cost_types
        supplies one int or float per name.
        make_subgraph samples internal edge costs; make_graph samples only new
        inter-subgraph edge costs. Each named cost is independent and uniform in
        [0, 5], with its configured Python type, before paths are computed.
        all_shortest_paths, multiobjective_shortest_paths, and scalar_dijkstra
        store parallel "paths" and "costs" lists, with one cost dictionary per
        path. all_shortest_paths reports hop counts under "distance";
        multiobjective_shortest_paths reports each objective's path total and
        "scalar_sum"; scalar_dijkstra reports the total under "scalar_sum".
        pareto_solver selects "dijkstra", "exhaustive" (default), or "paretoset".
        Retain every simple Pareto-optimal route, including equal-cost routes.
        Exhaustive and paretoset reuse all_paths; Dijkstra performs its own search. Reserve "scalar_sum" for the derived total.
        cost_precision (default 12) rounds each edge objective to that many
        decimal places, with ties to even, for both Pareto and scalar searches.
        Computation uses integer units; reported costs use the original units.
        Integer costs remain exact and original edge attributes are preserved.
        Missing costs default to 1; without named objectives, minimize edge hops.
        These initialization snapshots precede later add_edge_costs calls.
        task_complexity is a fixed integer for all subgraphs or a (lower, upper)
        tuple sampled uniformly and independently per subgraph, inclusive.
        The default (1, 10) preserves the original random difficulty range.
        log_metadata defaults to None, disabling file output. A filename string
        writes all initialized attributes to that file, replacing its contents.
        """
        # Validate the backend before constructing graphs or writing metadata.
        if pareto_solver not in mosp.PARETO_SOLVERS:
            raise ValueError("pareto_solver must be 'dijkstra', 'exhaustive', or 'paretoset'")
        self.pareto_solver = pareto_solver
        if not isinstance(mdp, bool):
            raise TypeError("mdp must be a boolean")
        self.mdp = mdp
        if not isinstance(edge_costs, bool):
            raise TypeError("edge_costs must be a boolean")
        self.edge_costs = edge_costs
        if not isinstance(restriction, tuple) or len(restriction) != 3:
            raise TypeError("restriction must be a tuple of three probabilities")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ) for value in restriction):
            raise TypeError("restriction must contain three numeric probabilities")
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in restriction):
            raise ValueError("restriction probabilities must be finite and between 0 and 1")
        self.restriction = tuple(float(value) for value in restriction)
        if any(self.restriction) and not np.isclose(sum(self.restriction), 1.0, rtol=0, atol=1e-12):
            raise ValueError("restriction probabilities must sum to 1, or all be zero")
        # Validate logging before generating the graph or creating output files.
        if log_metadata is not None and not isinstance(log_metadata, str):
            raise TypeError("log_metadata must be a filename string or None")
        if log_metadata == "":
            raise ValueError("log_metadata must be a nonempty filename")
        self.log_metadata = log_metadata
        # Use the same validated decimal policy for Pareto and scalarized paths.
        self.cost_precision = mosp.validate_cost_precision(cost_precision)
        if isinstance(n_subgraphs, (bool, np.bool_)) or not isinstance(n_subgraphs, (int, np.integer)):
            raise TypeError("n_subgraphs must be an integer")
        if n_subgraphs <= 0:
            raise ValueError("n_subgraphs must be positive to initialize the global graph")

        ranges = []
        for name, value in (("n_nodes_per_subgraph", n_nodes_per_subgraph), ("max_subgraph_outdegree", max_subgraph_outdegree)):
            bounds = value if isinstance(value, tuple) else (value, value)
            if len(bounds) != 2 or any(
                isinstance(bound, (bool, np.bool_)) or not isinstance(bound, (int, np.integer))
                for bound in bounds
            ):
                raise TypeError(f"{name} must be an integer or a tuple of two integers")
            lower, upper = map(int, bounds)
            if lower < 0 or lower > upper:
                raise ValueError(f"{name} requires 0 <= lower <= upper")
            if name == "max_subgraph_outdegree" and lower < 2:
                raise ValueError("max_subgraph_outdegree must be at least 2")
            ranges.append((lower, upper))

        # Validate and normalize fixed or ranged subgraph difficulty settings.
        bounds = task_complexity if isinstance(task_complexity, tuple) else (task_complexity, task_complexity)
        if len(bounds) != 2 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in bounds
        ):
            raise TypeError("task_complexity must be an integer or a tuple of two integers")
        lower, upper = map(int, bounds)
        if lower > upper:
            raise ValueError("task_complexity requires lower <= upper")
        self.task_complexity = (lower, upper) if isinstance(task_complexity, tuple) else lower

        self.subgraphs = {}
        self.themes = themes
        
        # Copy and validate parallel cost specifications before constructing edges.
        self.edge_cost_names = edge_cost_names
        self.edge_cost_types = edge_cost_types
        if any(not isinstance(name, str) or not name for name in self.edge_cost_names):
            raise ValueError("Edge cost names must be nonempty strings")
        # Prevent an objective total from colliding with the derived summary.
        if "scalar_sum" in self.edge_cost_names:
            raise ValueError("Edge cost name 'scalar_sum' is reserved for the derived total")
        if len(set(self.edge_cost_names)) != len(self.edge_cost_names):
            raise ValueError("Edge cost names must be unique")
        if len(self.edge_cost_names) != len(self.edge_cost_types):
            raise ValueError("Provide one edge cost type for every edge cost name")
        if any(cost_type is not int and cost_type is not float for cost_type in self.edge_cost_types):
            raise ValueError("Edge cost types must be int or float")
        
        self.n_subgraphs = n_subgraphs
        self.n_nodes_per_subgraph = n_nodes_per_subgraph
        self.max_subgraph_outdegree = max_subgraph_outdegree
        
        if n_subgraphs and not self.themes:
            raise ValueError("themes must be nonempty when assigning node themes")
        if any(not isinstance(theme, str) for theme in self.themes):
            raise TypeError("themes must contain only strings")
        rng = np.random.default_rng(seed)
        for index in range(1, n_subgraphs + 1):
            n, d = (lower if lower == upper else int(rng.integers(lower, upper + 1)) for lower, upper in ranges)
            self.subgraphs[index] = self.make_subgraph(n=n, d=d, seed=rng)

        self.graph = self.make_graph(self.subgraphs.values(), seed=rng)
        if self.mdp:
            self.assign_transition_probs(self.graph, seed=rng)
        if any(self.restriction):
            nodes = list(self.graph)
            # Normalize only floating-point roundoff after validating the sum.
            probabilities = np.array(self.restriction) / sum(self.restriction)
            samples = rng.choice(("none", "discouraged", "forbidden"), size=len(nodes), p=probabilities)
            nx.set_node_attributes(
                self.graph, {node: str(sample) for node, sample in zip(nodes, samples)},
                name="restriction",
            )
        # Read inherited assignments; do not reassign themes on the global graph.
        self.node_themes = nx.get_node_attributes(self.graph, "theme")
        # Store the directed adjacency array using NetworkX's default node order.
        self.adjacency_matrix = nx.to_numpy_array(self.graph)
        # Store each node's outgoing neighbors as a list.
        self.dict_of_lists = nx.to_dict_of_lists(self.graph)
        # Preserve the complete edge attribute dictionaries for outgoing neighbors.
        self.dict_of_dicts = nx.to_dict_of_dicts(self.graph)
        self.source = None
        self.target = None
        farthest_distance = 0
        n_ties = 0
        for source, distances in nx.all_pairs_shortest_path_length(self.graph):
            for target, distance in distances.items():
                if distance > farthest_distance:
                    farthest_distance = distance
                    n_ties = 1
                    self.source, self.target = source, target
                elif distance == farthest_distance and distance > 0:
                    # Reservoir sampling selects uniformly among tied pairs
                    # without storing the potentially quadratic set of pairs.
                    n_ties += 1
                    if rng.integers(n_ties) == 0:
                        self.source, self.target = source, target
        if self.source is None:
            raise ValueError("Graph must contain a directed path between two distinct nodes")

        # Enumerate all simple directed paths for the selected source and target.
        # Bound enumeration BEFORE materializing exponentially many paths.
        # Counts saturate at the limit, so even rejected DAGs remain cheap.
        if max_paths is not None:
            if max_paths < 1:
                raise ValueError("max_paths must be positive")
            counts = {node: 0 for node in self.graph}
            counts[self.target] = 1
            for node in reversed(list(nx.topological_sort(self.graph))):
                if node != self.target:
                    counts[node] = min(max_paths + 1, sum(counts[v] for v in self.graph[node]))
            if counts[self.source] > max_paths:
                raise ValueError("path_limit")
        self.all_paths = self.get_all_paths(self.source, self.target)
        # Reuse the enumerated paths to retain every path with the fewest edges.
        shortest_paths = self.brute_force_all_shortest_paths(paths=self.all_paths)
        # Match each unweighted shortest path with its scalar hop count.
        self.all_shortest_paths = {
            "paths": shortest_paths,
            "costs": [{"distance": len(path) - 1} for path in shortest_paths],
        }

        # Use named objectives, or a single unit-weight objective for edge hops.
        objectives = self.edge_cost_names or ["weight"]
        # Apply fallback unit costs on a copy when objective attributes are absent.
        cost_graph = self.graph.copy()
        for _, _, data in cost_graph.edges(data=True):
            for objective in objectives:
                data.setdefault(objective, 1)
        # Select the ground-truth backend while preserving distinct paths tied on optimal costs.
        pareto_paths = mosp.solve_pareto_paths(
            cost_graph, self.source, self.target, weight=objectives,
            paths=self.all_paths, cost_precision=self.cost_precision,
            pareto_solver=self.pareto_solver,
        )
        self.multiobjective_shortest_paths = {
            "paths": [path for path, _ in pareto_paths],
            # Match each path with its objective totals and their scalar sum.
            "costs": [{**dict(zip(objectives, cost)), "scalar_sum": self._scalarized_path_cost(path)}
                      for path, cost in pareto_paths],
        }
        # Wrap the single scalarized shortest path and its cost in matching lists.
        scalar_path = self.scalarized_dijkstra(self.source, self.target)
        self.scalar_dijkstra = {
            "paths": [scalar_path],
            "costs": [{"scalar_sum": self._scalarized_path_cost(scalar_path)}],
        }
        # Log only after every computed attribute is available.
        if self.log_metadata is not None:
            self._write_metadata()

    def _print_metadata(self, stream=None, *, image=None):
        """Pretty-print complete graph metadata to the console or a text stream."""
        # Include the image location when main has actually exported its PNG.
        if image is not None:
            print(f"PNG saved to: {image}\n\n", file=stream)

        def display_value(value):
            # Expand NetworkX objects so their nodes, edges, and attributes are visible.
            if isinstance(value, nx.Graph):
                return {
                    "graph_attributes": value.graph,
                    "nodes": list(value.nodes(data=True)),
                    "edges": list(value.edges(data=True)),
                }
            # Expand graphs stored inside dictionaries such as self.subgraphs as well.
            if isinstance(value, dict):
                return {key: display_value(item) for key, item in value.items()}
            return value

        # Print every instance attribute without truncating NumPy adjacency arrays.
        with np.printoptions(threshold=np.inf):
            for name, value in vars(self).items():
                print(f"{name}:", file=stream)
                pprint(display_value(value), width=100, sort_dicts=False, stream=stream)
                print("\n", file=stream)  # Separate each attribute from the next with blank lines.

    def _write_metadata(self, *, image=None):
        """Replace the configured log with the complete current instance attributes."""
        # Use the same formatter as main, including full arrays and graph contents.
        with open(self.log_metadata, "w", encoding="utf-8") as stream:
            self._print_metadata(stream, image=image)


    def make_subgraph(self, n, d=2, *, seed=None):
        """Return a DAG on nodes 0..n-1 with at most d outgoing edges each.

        Randomly order nodes and add edges only to later nodes in that order.
        Each node gets min(d, number of later nodes) children, at least two
        whenever two later nodes exist. Include the next node to guarantee weak
        connectivity, then sample remaining children without replacement.
        The penultimate node has one child and the final node is a sink.
        This does not sample uniformly over all DAGs. Requires 2 <= d <= n - 1.
        seed accepts anything supported by numpy.random.default_rng.
        Assign balanced node themes, choosing remainder themes from those least
        represented in previously constructed subgraphs, with random tie breaks.
        Sample each configured edge cost before returning the subgraph.
        Store task_complexity as a graph attribute using the fixed constructor
        value or a uniform sample from its inclusive configured range.
        """
        for name, value in (("n", n), ("d", d)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        if not 2 <= d <= n - 1:
            raise ValueError("Require 2 <= d <= n - 1 (at least three nodes)")

        rng = np.random.default_rng(seed)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))
        order = rng.permutation(n)
        for position, node in enumerate(order):
            later = order[position + 1:]
            count = min(d, len(later))
            if count:
                # A forward chain connects all nodes without introducing cycles.
                children = [later[0], *rng.choice(later[1:], size=count - 1, replace=False)]
                graph.add_edges_from((int(node), int(child)) for child in children)
        if not self.test_acyclic(graph):
            raise ValueError("Generated subgraph is not acyclic")
        # Give every distinct theme the same base number of nodes.
        themes = list(dict.fromkeys(self.themes))
        base, remainder = divmod(n, len(themes))
        previous_counts = Counter(
            data.get("theme") for subgraph in self.subgraphs.values()
            for _, data in subgraph.nodes(data=True)
        )
        # Prefer globally underrepresented themes for the rounding remainder;
        # shuffle first so ties are resolved randomly rather than by theme name.
        priority = rng.permutation(len(themes)).tolist()
        priority.sort(key=lambda index: previous_counts[themes[index]])
        extra = set(priority[:remainder])
        assignments = [theme for index, theme in enumerate(themes)
                       for _ in range(base + (index in extra))]
        # Randomize which nodes receive each theme while preserving exact quotas.
        rng.shuffle(assignments)
        nx.set_node_attributes(graph, dict(zip(graph, assignments)), name="theme")
        # Initialize internal edge costs here so composition can preserve them.
        self._assign_random_edge_costs(graph, graph.edges, rng)
        # Store a fixed difficulty or independently sample the inclusive range.
        if isinstance(self.task_complexity, tuple):
            lower, upper = self.task_complexity
            complexity = lower if lower == upper else int(rng.integers(lower, upper + 1))
        else:
            complexity = self.task_complexity
        graph.graph["task_complexity"] = complexity
        return graph


    def test_connected(self, graph):
        """Return whether graph has exactly one weakly connected component."""
        return nx.number_weakly_connected_components(graph) == 1


    @staticmethod
    def test_acyclic(graph):
        """Return True exactly when graph contains no cycles, including self-loops.

        Directed graphs are checked for directed cycles. Undirected graphs are
        checked for undirected cycles; an empty graph is acyclic in either case.
        """
        if graph.is_directed():
            return nx.is_directed_acyclic_graph(graph)
        return len(graph) == 0 or nx.is_forest(graph)


    def make_graph(self, subgraphs, *, seed=None):
        """Combine nonempty, weakly connected DAGs into a weakly connected DAG.

        Relabel nodes as (subgraph_index, original_node), indexed from 1.
        Preserve node and edge attributes without modifying inputs. Reject
        cyclic inputs and parallel edges. Sample costs only for new connectors.

        Randomly order subgraphs and connect each to its successor. For every
        group except the last, sample ceil(0.3*n)..floor(0.6*n) distinct sources
        (one for a singleton). Each source gets one child in a later group;
        the first source connects to the immediately following group to ensure
        weak connectivity. The final group has no outgoing connectors.
        Edges always advance in the group order, so no cycle can cross groups.
        A single input is copied without connectors; an empty list is rejected.
        New costs are int in [0, 5] or float in [0.0, 5.0].
        seed accepts anything supported by numpy.random.default_rng.
        """
        subgraphs = list(subgraphs)
        if not subgraphs:
            raise ValueError("At least one subgraph is required")
        for index, subgraph in enumerate(subgraphs, start=1):
            if not subgraph.is_directed() or subgraph.is_multigraph():
                raise ValueError(f"Subgraph {index} must be a DiGraph without parallel edges")
            if not self.test_connected(subgraph):
                raise ValueError(f"Subgraph {index} must be nonempty and weakly connected")
            if not self.test_acyclic(subgraph):
                raise ValueError(f"Subgraph {index} must be acyclic")

        rng = np.random.default_rng(seed)
        graph = nx.DiGraph()
        groups = []
        for index, subgraph in enumerate(subgraphs, start=1):
            # Match the 1-based keys used by the stored subgraph metadata.
            nodes = [(index, node) for node in subgraph]
            groups.append(nodes)
            graph.add_nodes_from(((index, node), data.copy()) for node, data in subgraph.nodes(data=True))
            graph.add_edges_from(
                ((index, u), (index, v), data.copy()) for u, v, data in subgraph.edges(data=True)
            )
        # A forward chain of groups ensures connectivity without a closing ring.
        order = rng.permutation(len(groups)).tolist()
        for position, index in enumerate(order[:-1]):
            nodes = groups[index]
            n = len(nodes)
            lower = (3 * n + 9) // 10
            upper = max(lower, (6 * n) // 10)
            count = int(rng.integers(lower, upper + 1))
            sources = rng.choice(n, size=count, replace=False)
            later_nodes = [node for other in order[position + 1:] for node in groups[other]]
            for offset, source in enumerate(sources):
                candidates = groups[order[position + 1]] if offset == 0 else later_nodes
                child = candidates[int(rng.integers(len(candidates)))]
                graph.add_edge(nodes[int(source)], child)
        if not self.test_acyclic(graph):
            raise ValueError("Generated graph is not acyclic")
        if not self.test_connected(graph):
            raise ValueError("Generated graph is not weakly connected")
        # Node identifiers include their group, so select only the new connectors.
        inter_edges = [(parent, child) for parent, child in graph.edges if parent[0] != child[0]]
        self._assign_random_edge_costs(graph, inter_edges, rng)
        return graph


    def _assign_random_edge_costs(self, graph, edges, rng):
        """Assign typed random costs to selected edges when edge_costs is enabled."""
        if not self.edge_costs:
            return
        # Materialize the selection so each objective updates exactly the same edges.
        edges = list(edges)
        for name, cost_type in zip(self.edge_cost_names, self.edge_cost_types):
            if cost_type is int:
                # NumPy's exclusive upper bound of 6 includes integer cost 5.
                values = {edge: int(rng.integers(0, 6)) for edge in edges}
            else:
                # Extend the upper bound by one float step to allow endpoint 5;
                # clamp rounding at the endpoint to keep every result in range.
                upper = np.nextafter(5.0, np.inf)
                values = {edge: float(min(rng.uniform(0.0, upper), 5.0)) for edge in edges}
            nx.set_edge_attributes(graph, values, name=name)


    def assign_transition_probs(self, graph=None, *, seed=None):
        """Assign outgoing transition probabilities in place and return graph.

        Default to self.graph. A single child has probability 1.0. For two to
        ten children, reserve 0.1 for each edge and randomly split the remaining
        mass using a Dirichlet distribution. This guarantees values in [0.1,
        0.9] and a sum of 1 up to floating-point roundoff. Ten children each get
        0.1. Sinks need no attributes. Reject infeasible degrees before editing.
        seed accepts anything supported by numpy.random.default_rng.
        """
        graph = self.graph if graph is None else graph
        if not graph.is_directed() or graph.is_multigraph():
            raise ValueError("Transition probabilities require a DiGraph without parallel edges")
        for node, degree in graph.out_degree():
            if degree > 10:
                raise ValueError(
                    f"Node {node!r} has {degree} outgoing edges; at most 10 can each have probability >= 0.1"
                )
        rng = np.random.default_rng(seed)
        attributes = {}
        for node in graph:
            edges = list(graph.out_edges(node))
            degree = len(edges)
            if degree == 0:
                continue
            if degree == 1:
                probabilities = [1.0]
            elif degree == 10:
                probabilities = [0.1] * degree
            else:
                probabilities = 0.1 + (1.0 - 0.1 * degree) * rng.dirichlet(np.ones(degree))
            attributes.update({edge: float(probability) for edge, probability in zip(edges, probabilities)})
        nx.set_edge_attributes(graph, attributes, name="transition_prob")
        return graph


    def plot_graph(self, graph, filename="graph.png", *, seed=0, edge_cost=None, color: str = "restriction",
                   transition_probs: bool = False):
        """Save a PNG with arrows and node names; return the output filename.

        Nodes labeled (subgraph_index, original_node) are grouped by subgraph,
        with each group on its own circle and group centers on a larger circle.
        A standalone subgraph uses one circle. This layout is deterministic;
        seed is retained for compatibility and does not affect positions.
        color names a node attribute in the plotted graph, defaulting to
        "restriction". Equal attribute values share a color and legend entry.
        Use color="theme" for theme coloring. Missing values use light gray.
        Reject names absent from every node and non-string names. Attribute
        values must be hashable categories.
        A node's 'name' attribute is its label, falling back to its node identifier.
        Set edge_cost to an edge attribute name (e.g. "distance" or "weight")
        or a list of names. Lists display values in order as (1,3,2), with no
        spaces. Edges missing any requested attribute are left unlabeled.
        None or an empty list hides edge costs. Cost labels use red text.
        transition_probs=True shows existing transition_prob values in blue,
        rounded to three decimal places for display only, independently of edge costs. Missing transition probabilities are omitted.
        When self.edge_costs is False, hide all cost labels and cost legends
        regardless of edge_cost or any existing edge attributes.
        Separate node-color and edge-cost legends sit side by side above the graph.
        The cost legend maps zero-based tuple indices to names in label order.
        Label boxes never overlap: try clear locations along/beside the edge,
        then use a separate callout column with leader lines if necessary.
        Edges use distinct, evenly spaced attachment points around each node,
        with radial entry/exit segments separating incoming heads and outgoing
        tails. Node sizes grow with incident edge count to maintain clearance.
        Bends use tangent quadratic Bezier curves checked against node clearance;
        short straight endpoint segments preserve the separated heads and tails.
        Edges detour around unrelated nodes with clearance for arrowheads.
        Routing uses display coordinates so clearance matches the PNG's marker
        sizes. Crowded layouts expand the figure to leave routing space.
        Uses a noninteractive canvas, so this also works without a display.
        """
        if not isinstance(transition_probs, bool):
            raise TypeError("transition_probs must be a boolean")
        if not isinstance(color, str):
            raise TypeError("color must be a node attribute name string")
        node_values = nx.get_node_attributes(graph, color)
        if not node_values:
            raise ValueError(f"No nodes have the color attribute {color!r}")
        if edge_cost is None:
            cost_names = []
        elif isinstance(edge_cost, str):
            cost_names = [edge_cost]
        elif isinstance(edge_cost, list) and all(isinstance(name, str) for name in edge_cost):
            cost_names = edge_cost
        else:
            raise TypeError("edge_cost must be a string, a list of strings, or None")
        if not self.edge_costs:
            cost_names = []

        from matplotlib import colormaps
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import hsv_to_rgb
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch, Patch
        from matplotlib.path import Path

        figure = Figure(figsize=(12, 9), dpi=150)
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        groups = {}
        for node in graph:
            group = node[0] if isinstance(node, tuple) and len(node) == 2 else 0
            groups.setdefault(group, []).append(node)

        # Allow larger subgraphs more room, and keep neighboring circles apart.
        radii = {group: max(1.0, len(nodes) / 10) for group, nodes in groups.items()}
        n_groups = len(groups)
        spacing = 0.0 if n_groups <= 1 else 1.6 * max(radii.values()) / np.sin(np.pi / n_groups)
        centers = nx.circular_layout(list(groups), scale=spacing)
        positions = {}
        for group, nodes in groups.items():
            positions.update(nx.circular_layout(nodes, scale=radii[group], center=centers[group]))
        labels = {node: str(data.get("name", node)) for node, data in graph.nodes(data=True)}
        # Preserve the established theme palette when coloring by theme.
        values = list(dict.fromkeys([
            *(self.themes if color == "theme" else []), *node_values.values(),
        ]))
        value_colors = {
            value: colormaps["Set3"](index) if len(values) <= 12 else
            hsv_to_rgb((index / len(values), 0.35, 0.95))
            for index, value in enumerate(values)
        }
        node_colors = [value_colors[node_values[node]] if node in node_values else "lightgray"
                       for node in graph]
        max_degree = max((degree for _, degree in graph.degree()), default=0)
        marker_radius = np.sqrt(700) / 2
        if max_degree > 1:
            # Ten points between neighboring ports leaves room for a 12-point
            # arrow's head, the outgoing stroke, and a visible gap.
            marker_radius = max(marker_radius, 10 / (2 * np.sin(np.pi / max_degree)))
        node_size = (2 * marker_radius) ** 2
        nx.draw_networkx_nodes(
            graph, pos=positions, ax=axes, node_color=node_colors, node_size=node_size,
        )
        nx.draw_networkx_labels(graph, pos=positions, labels=labels, ax=axes, font_size=10)
        present_values = set(node_values.values())
        legend_handles = [
            Patch(facecolor=value_colors[value], label=str(value))
            for value in values if value in present_values
        ]
        if len(node_values) < len(graph):
            legend_handles.append(Patch(facecolor="lightgray", label="Unassigned"))
        # Put node colors to the left and edge-cost names to the right.
        # Raise both boxes equally to leave more whitespace above the graph.
        legend_y = 1.08
        if legend_handles:
            node_legend = axes.legend(
                handles=legend_handles, title=color.capitalize(),
                loc="lower right" if cost_names else "lower center",
                bbox_to_anchor=(0.49 if cost_names else 0.5, legend_y),
            )
            # Preserve the node legend when axes.legend creates the cost legend.
            if cost_names:
                axes.add_artist(node_legend)
        if cost_names:
            # Text-only red entries match the ordering and color of edge labels.
            cost_handles = [
                Line2D([], [], linestyle="none", label=(
                    f"[{index}] {name}" if isinstance(edge_cost, list) else name
                )) for index, name in enumerate(cost_names)
            ]
            axes.legend(
                handles=cost_handles,
                title="Edge costs (0-based tuple index)" if isinstance(edge_cost, list) else "Edge cost",
                loc="lower left" if legend_handles else "lower center",
                bbox_to_anchor=(0.51 if legend_handles else 0.5, legend_y),
                handlelength=0, handletextpad=0, labelcolor="red",
            )
        axes.set_aspect("equal")
        axes.margins(0.15)
        axes.set_axis_off()
        figure.canvas.draw()

        # Matplotlib marker sizes are in points squared; route in pixels to
        # account for the actual output size, aspect ratio, and node radius.
        pixels_per_point = figure.dpi / 72
        node_radius = (marker_radius + 1.5) * pixels_per_point
        clearance = node_radius + 18 * pixels_per_point
        nodes = list(graph)
        if len(nodes) > 1:
            centers = axes.transData.transform([positions[node] for node in nodes])
            nearest = min(
                np.max(np.abs(centers[i + 1:] - center), axis=1).min()
                for i, center in enumerate(centers[:-1])
            )
            # Disjoint clearance boxes guarantee a corridor around every node.
            scale = max(1.0, (2 * clearance + 4 * pixels_per_point) / nearest)
            if scale > 1:
                figure.set_size_inches(figure.get_size_inches() * scale)
                figure.canvas.draw()
        axes.set_autoscale_on(False)
        routes = self._route_plot_edges(graph, positions, axes, clearance, node_radius)
        inverse = axes.transData.inverted()
        node_centers = axes.transData.transform([positions[node] for node in nodes]) if nodes else np.empty((0, 2))
        edge_labels = []
        for (u, v), route in routes.items():
            display_path = self._smooth_plot_route(
                route, node_centers, node_radius + 6 * pixels_per_point,
                bend_radius=24 * pixels_per_point,
            )
            path = display_path.transformed(inverse)
            axes.add_patch(FancyArrowPatch(
                path=path, arrowstyle="-|>", mutation_scale=12,
                linewidth=1, color="black", zorder=1, clip_on=False,
            ))
            show_cost = cost_names and all(name in graph[u][v] for name in cost_names)
            show_transition = transition_probs and "transition_prob" in graph[u][v]
            if show_cost or show_transition:
                # Share collision-aware placement for cost and transition labels.
                route = np.vstack([display_path.vertices[:1]] + [
                    curve(np.linspace(0, 1, 17))[1:]
                    for curve, code in display_path.iter_bezier() if code != Path.MOVETO
                ])
                if show_cost:
                    values = [str(graph[u][v][name]) for name in cost_names]
                    label = "(" + ",".join(values) + ")" if isinstance(edge_cost, list) else values[0]
                    edge_labels.append((label, route))
                if show_transition:
                    label = f'{graph[u][v]["transition_prob"]:.3f}'
                    edge_labels.append((label, route, "blue"))
        self._place_plot_edge_labels(axes, edge_labels, node_centers, node_radius)
        figure.savefig(filename, format="png", dpi=figure.dpi, bbox_inches="tight")
        return filename


    def _place_plot_edge_labels(self, axes, labels, centers, node_radius):
        """Place measured text boxes without collisions, returning text artists.

        Work at the output DPI and keep data limits frozen. Rotated text and
        its background box are both measured in display pixels. If all nearby
        candidates collide, an unbounded callout column guarantees space for
        every label. Saving with bbox_inches='tight' includes those callouts.
        """
        from matplotlib.legend import Legend
        from matplotlib.transforms import Bbox

        renderer = axes.figure.canvas.get_renderer()
        pixels_per_point = axes.figure.dpi / 72
        gap = 2 * pixels_per_point
        inverse = axes.transData.inverted()
        occupied = [text.get_window_extent(renderer).padded(gap) for text in axes.texts]
        occupied.extend(Bbox.from_extents(
            center[0] - node_radius, center[1] - node_radius,
            center[0] + node_radius, center[1] + node_radius,
        ).padded(gap) for center in centers)
        # Both legends occupy space, including the one preserved with add_artist.
        occupied.extend(artist.get_window_extent(renderer).padded(gap)
                        for artist in axes.get_children() if isinstance(artist, Legend))
        column_x = axes.bbox.x1 + 24 * pixels_per_point
        column_y = axes.bbox.y1
        artists = []
        for label, route, *label_color in labels:
            lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
            cumulative = np.cumsum(lengths)
            text = axes.text(
                0, 0, label, fontsize=8, color=label_color[0] if label_color else "red", rotation_mode="anchor",
                ha="center", va="center", zorder=4, clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
            )
            chosen_box = None
            for fraction in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9):
                distance = fraction * cumulative[-1]
                segment = min(int(np.searchsorted(cumulative, distance)), len(lengths) - 1)
                direction = (route[segment + 1] - route[segment]) / lengths[segment]
                anchor = route[segment] + direction * (distance - lengths[:segment].sum())
                normal = np.array([-direction[1], direction[0]])
                angle = (np.degrees(np.arctan2(direction[1], direction[0])) + 90) % 180 - 90
                text.set_rotation(angle)
                for offset in (0, 8, -8, 16, -16, 24, -24):
                    location = anchor + offset * pixels_per_point * normal
                    text.set_position(inverse.transform(location))
                    text.update_bbox_position_size(renderer)
                    box = text.get_bbox_patch().get_window_extent(renderer).padded(gap)
                    if not any(box.overlaps(other) for other in occupied):
                        chosen_box = box
                        break
                if chosen_box is not None:
                    break
            if chosen_box is None:
                # Horizontal callouts have no width limit and retain a leader
                # to the edge's midpoint. Scan downward until no box overlaps.
                distance = cumulative[-1] / 2
                segment = min(int(np.searchsorted(cumulative, distance)), len(lengths) - 1)
                direction = (route[segment + 1] - route[segment]) / lengths[segment]
                anchor = route[segment] + direction * (distance - lengths[:segment].sum())
                text.set_rotation(0)
                text.set_ha("left")
                while True:
                    location = np.array([column_x, column_y])
                    text.set_position(inverse.transform(location))
                    text.update_bbox_position_size(renderer)
                    box = text.get_bbox_patch().get_window_extent(renderer).padded(gap)
                    if not any(box.overlaps(other) for other in occupied):
                        chosen_box = box
                        column_y -= box.height + gap
                        break
                    column_y -= box.height + gap
            # Connect displaced labels from the nearest box boundary, keeping
            # the leader outside the text itself. Edge-centered labels need none.
            leader_start = np.clip(anchor, chosen_box.p0, chosen_box.p1)
            if np.linalg.norm(leader_start - anchor) > pixels_per_point:
                axes.annotate(
                    "", xy=inverse.transform(anchor), xytext=inverse.transform(leader_start),
                    arrowprops={"arrowstyle": "-", "color": "0.5", "linewidth": 0.6},
                    annotation_clip=False, zorder=3,
                )
            occupied.append(chosen_box)
            artists.append(text)
        return artists


    @staticmethod
    def _smooth_plot_route(route, centers, protected_radius, *, bend_radius):
        """Round route corners with tangent, obstacle-safe quadratic Beziers.

        A Bezier stays inside its control points' bounding box. Only accept a
        rounding whose entire box avoids the protected disks around all nodes,
        including space occupied by other incoming heads and outgoing tails.
        """
        from matplotlib.path import Path

        vertices = [route[0]]
        codes = [Path.MOVETO]
        for previous, corner, following in zip(route, route[1:], route[2:]):
            incoming = corner - previous
            outgoing = following - corner
            incoming_length = np.linalg.norm(incoming)
            outgoing_length = np.linalg.norm(outgoing)
            incoming /= incoming_length
            outgoing /= outgoing_length
            if np.dot(incoming, outgoing) > 1 - 1e-10:
                # Collinear segments already have a continuous tangent.
                vertices.append(corner)
                codes.append(Path.LINETO)
                continue
            # Leave a straight portion between neighboring rounded bends and
            # retain radial endpoint segments for the separated heads/tails.
            trim = min(bend_radius, 0.4 * incoming_length, 0.4 * outgoing_length)
            while True:
                before = corner - trim * incoming
                after = corner + trim * outgoing
                controls = np.array([before, corner, after])
                nearest = np.clip(centers, controls.min(axis=0), controls.max(axis=0))
                if np.all(np.linalg.norm(centers - nearest, axis=1) >= protected_radius):
                    break
                trim *= 0.5
                if trim < 1e-6:
                    raise ValueError("Insufficient node clearance to smooth an edge bend")
            vertices.extend([before, corner, after])
            codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])
        vertices.append(route[-1])
        codes.append(Path.LINETO)
        return Path(vertices, codes)


    @staticmethod
    def _plot_segment_clear(start, end, centers, clearance, ignored=()):
        """Check that a display-space segment avoids node clearance boxes."""
        # Slightly inset boxes allow travel along their boundaries.
        lower = centers - clearance + 1e-7
        upper = centers + clearance - 1e-7
        enter = np.zeros(len(centers))
        leave = np.ones(len(centers))
        active = np.ones(len(centers), dtype=bool)
        for axis, delta in enumerate(end - start):
            if abs(delta) < 1e-10:
                active &= (start[axis] > lower[:, axis]) & (start[axis] < upper[:, axis])
            else:
                first = (lower[:, axis] - start[axis]) / delta
                second = (upper[:, axis] - start[axis]) / delta
                enter = np.maximum(enter, np.minimum(first, second))
                leave = np.minimum(leave, np.maximum(first, second))
        active[list(ignored)] = False
        return not np.any(active & (enter < leave))

    def _route_plot_edges(self, graph, positions, axes, clearance, node_radius):
        """Route display-space edges through distinct node attachment ports.

        Radial endpoint segments stay inside their own clearance boxes. All
        other segments avoid every box, protecting arrowheads from passing
        edges as well as from the outgoing tails at the same node.
        """
        nodes = list(graph)
        indices = {node: index for index, node in enumerate(nodes)}
        centers = axes.transData.transform([positions[node] for node in nodes]) if nodes else np.empty((0, 2))
        corners = (centers[:, None, :] + clearance * np.array([
            [-1, -1], [-1, 1], [1, 1], [1, -1],
        ])).reshape(-1, 2)
        attachments = {edge: np.empty((4, 2)) for edge in graph.edges}
        for node in nodes:
            center = centers[indices[node]]
            incidents = []
            for u, v in graph.out_edges(node):
                if u == v:
                    raise ValueError("Obstacle-aware plotting requires a graph without self-loops")
                direction = centers[indices[v]] - center
                incidents.append((np.arctan2(direction[1], direction[0]), (u, v), 0))
            for u, v in graph.in_edges(node):
                direction = centers[indices[u]] - center
                incidents.append((np.arctan2(direction[1], direction[0]), (u, v), 3))
            if not incidents:
                continue
            incidents.sort(key=lambda incident: incident[0])
            # Preserve the neighbors' angular order, then rotate the uniformly
            # spaced ports toward them to reduce unnecessary bends.
            angles = 2 * np.pi * np.arange(len(incidents)) / len(incidents)
            desired = np.array([incident[0] for incident in incidents])
            angles += np.angle(np.exp(1j * (desired - angles)).sum())
            for angle, (_, edge, endpoint) in zip(angles, incidents):
                unit = np.array([np.cos(angle), np.sin(angle)])
                attachments[edge][endpoint] = center + node_radius * unit
                port = center + clearance * unit / np.max(np.abs(unit))
                attachments[edge][1 if endpoint == 0 else 2] = port

        visibility = None
        routes = {}
        for edge, attachment in attachments.items():
            start, end = attachment[1:3]
            if self._plot_segment_clear(start, end, centers, clearance):
                routes[edge] = attachment
                continue
            if visibility is None:
                visibility = nx.Graph()
                visibility.add_nodes_from(range(len(corners)))
                for i, corner in enumerate(corners):
                    for j in range(i + 1, len(corners)):
                        if self._plot_segment_clear(corner, corners[j], centers, clearance):
                            visibility.add_edge(i, j, weight=float(np.linalg.norm(corner - corners[j])))
            # Endpoint ports lie on their node's clearance boundary. Visibility
            # connections cannot cut through that node, including on departure.
            start_id, end_id = len(corners), len(corners) + 1
            for endpoint, port in ((start_id, start), (end_id, end)):
                visibility.add_node(endpoint)
                for index, corner in enumerate(corners):
                    if self._plot_segment_clear(port, corner, centers, clearance):
                        visibility.add_edge(endpoint, index, weight=float(np.linalg.norm(port - corner)))
            path = nx.shortest_path(visibility, start_id, end_id, weight="weight")
            route = np.vstack([attachment[:2], corners[path[1:-1]], attachment[2:]])
            # A port may coincide with a corner; omit duplicate waypoints.
            routes[edge] = route[np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-7]]
            visibility.remove_nodes_from((start_id, end_id))
        return routes


    def add_edge_costs(self, graph, costs):
        """Add named edge attributes in place and return graph.

        costs maps attribute names to a constant, a mapping keyed by (u, v),
        or a callable accepting (u, v). For example::

            self.add_edge_costs(graph, {
                "distance": lambda u, v: float(rng.uniform(1, 10)),
                "time": {(u, v): 2.0 for u, v in graph.edges},
                "capacity": 1,
            })

        Pass a cost name as 'weight' to routing functions to choose the objective.
        Compute attributes before applying them, so failures do not partially edit
        the graph. Costs should be numeric; capacities should be nonnegative.
        """
        attributes = {}
        for name, specification in costs.items():
            if not isinstance(name, str):
                raise TypeError("Cost names must be strings")
            for u, v in graph.edges:
                if callable(specification):
                    value = specification(u, v)
                elif hasattr(specification, "keys"):
                    value = specification[u, v]
                else:
                    value = specification
                attributes.setdefault((u, v), {})[name] = value
        for (u, v), data in attributes.items():
            graph[u][v].update(data)
        return graph


    def get_parents(self, graph, node):
        """Return a list of the node's direct predecessors."""
        return list(graph.predecessors(node))


    def get_children(self, graph, node):
        """Return a list of the node's direct successors."""
        return list(graph.successors(node))


    def get_all_paths(self, source = None, target = None):
        """Return all simple directed node paths from source to target in self.graph.

        Paths never repeat a node, so cycles do not create infinite walks.
        Return [] when no directed path exists, and [[source]] when source
        equals target. Missing endpoints raise NetworkX NodeNotFound.
        """
        if source is None:
            source = self.source
        if target is None:
            target = self.target
            
        if source not in self.graph:
            raise nx.NodeNotFound(f"Source {source} is not in graph")
        if target not in self.graph:
            raise nx.NodeNotFound(f"Target {target} is not in graph")
        return list(nx.all_simple_paths(self.graph, source, target))


    def brute_force_all_shortest_paths(self, source=None, target=None, paths=None, get_all_paths=False):
        """Return all paths tied for the minimum number of edges.

        Supply a node-path list, or set get_all_paths=True to enumerate paths
        from self.source to self.target in self.graph. The flag takes precedence
        over a supplied list. Preserve input order and return [] for an empty
        path list. Edge weights are ignored.
        """
        if source is None:
            source = self.source
        if target is None:
            target = self.target
            
        if get_all_paths:
            paths = self.get_all_paths(source, target)
        elif paths is None:
            raise ValueError("Provide a paths list or set get_all_paths=True")
        if not paths:
            return []
        
        minimum_length = min(len(path) - 1 for path in paths)
        return [path for path in paths if len(path) - 1 == minimum_length]


    def _scalarized_edge_cost(self, parent, child, *, as_units=False):
        """Sum one edge's objectives for both Dijkstra and reported path costs."""
        # Read named objectives or the vector-valued cost on the original edge.
        data = self.graph[parent][child]
        cost = ([data.get(name, 1) for name in self.edge_cost_names]
                if self.edge_cost_names else data.get("cost", data.get("weight", 1)))
        vector = np.asarray(cost, dtype=object)
        if vector.ndim == 0:
            vector = vector.reshape(1)
        # Validate and quantize each objective before summing, just as MOSP does.
        if vector.ndim != 1 or not vector.size:
            raise ValueError(f"Edge {(parent, child)!r} needs finite nonnegative costs")
        units = sum(mosp.cost_to_units(value, self.cost_precision) for value in vector.tolist())
        return units if as_units else mosp.cost_from_units(units, self.cost_precision)


    def _scalarized_path_cost(self, path):
        """Report the path's scalar total without resumming rounded float outputs."""
        # Share exact edge totals between Pareto summaries and scalarized Dijkstra.
        units = sum(self._scalarized_edge_cost(u, v, as_units=True) for u, v in zip(path, path[1:]))
        return mosp.cost_from_units(units, self.cost_precision)


    def scalarized_dijkstra(self, source=None, target=None):
        """Return the shortest node path after summing each edge's objectives.

        Use self.edge_cost_names in their configured order. Without named objectives,
        use the edge's 'cost' tuple, falling back to 'weight' or unit cost.
        Missing named costs default to 1, matching MOSP initialization. Costs
        must be finite and nonnegative. Quantize each objective using
        self.cost_precision and run Dijkstra on exact integer units, matching
        the Pareto computation. Does not modify self.graph. Missing
        endpoints and unreachable targets raise NetworkX exceptions.
        """
        # Default to the endpoints chosen when this generator was initialized.
        source = self.source if source is None else source
        target = self.target if target is None else target
        # Copy edge attribute dictionaries so scalarization cannot edit self.graph.
        scalar_graph = self.graph.copy()
        for parent, child, data in scalar_graph.edges(data=True):
            # Store the summed integer units only on the copied graph.
            data["scalarized_weight"] = self._scalarized_edge_cost(parent, child, as_units=True)

        # Run Dijkstra entirely on the copy using its scalar edge weights.
        return nx.dijkstra_path(scalar_graph, source, target, weight="scalarized_weight")


    def get_shortest_path(self, graph, source, target, method="dijkstra", *, weight="weight"):
        """Return (node_path, total_cost) using the selected algorithm.

        bfs minimizes edge count, but reports the selected path's weighted cost.
        dijkstra requires nonnegative
        costs. bellman-ford permits negative costs, but not reachable negative
        cycles. Missing weight attributes default to 1, following NetworkX.
        If no directed path exists, print a message and return None.
        NetworkX exceptions for missing nodes still propagate.
        """
        if source not in graph:
            raise nx.NodeNotFound(f"Source {source} is not in graph")
        if target not in graph:
            raise nx.NodeNotFound(f"Target {target} is not in graph")
        if method not in ("bfs", "bellman-ford", "dijkstra"):
            raise ValueError("method must be 'bellman-ford', 'bfs', or 'dijkstra'")
        try:
            if method == "bfs":
                if weight is not None:
                    print("Warning: BFS returns the shortest path by total edge count; edge weights are ignored when choosing the path.")
                path = nx.shortest_path(graph, source, target, weight=None)
            else:
                path = nx.shortest_path(graph, source, target, weight=weight, method=method)
        except nx.NetworkXNoPath:
            print("No directed path between source and target")
            return None
        if weight is None:
            total_cost = len(path) - 1
        elif callable(weight):
            total_cost = sum(weight(u, v, graph[u][v]) for u, v in zip(path, path[1:]))
        else:
            total_cost = sum(graph[u][v].get(weight, 1) for u, v in zip(path, path[1:]))
        return path, total_cost


    def min_cost_max_flow(self, graph, source, target, *, capacity="capacity", weight="weight"):
        """Return [(node_path, flow_amount), ...] for a minimum-cost maximum flow.

        Missing capacities default to 1 and missing costs to 1. Integer capacities
        and costs are recommended for NetworkX flow algorithms. The result can
        contain several paths because maximum flow may split across routes.
        Returns [] if no flow is possible. Any circulation in the optimal flow is
        excluded from this source-to-target path decomposition.
        """
        if not graph.is_directed() or graph.is_multigraph():
            raise nx.NetworkXError("Expected a directed graph without parallel edges")
        if source not in graph or target not in graph:
            raise nx.NodeNotFound("Both source and target must be in the graph")
        if source == target:
            raise nx.NetworkXError("Source and target must be different")

        working = graph.copy()
        for _, _, data in working.edges(data=True):
            data.setdefault(capacity, 1)
        flow = nx.max_flow_min_cost(working, source, target, capacity=capacity, weight=weight)
        remaining = nx.DiGraph()
        remaining.add_nodes_from(graph)
        for u, neighbors in flow.items():
            for v, amount in neighbors.items():
                if amount > 0:
                    remaining.add_edge(u, v, remaining=amount)

        # Limit extraction to the net source flow; optimal negative-cost cycles
        # can also pass through the source or target.
        flow_left = sum(flow[source].values()) - sum(neighbors.get(source, 0) for neighbors in flow.values())
        paths = []
        while flow_left > 0:
            path = nx.shortest_path(remaining, source, target)
            amount = min(flow_left, *(remaining[u][v]["remaining"] for u, v in zip(path, path[1:])))
            paths.append((path, amount))
            for u, v in zip(path, path[1:]):
                remaining[u][v]["remaining"] -= amount
                if remaining[u][v]["remaining"] <= 0:
                    remaining.remove_edge(u, v)
            flow_left -= amount
        return paths


def main():
    # Construct the graph and compute all attributes through the initializer.
    generator = GraphGenerator(n_subgraphs = 2, 
                 n_nodes_per_subgraph = (5,10),
                 max_subgraph_outdegree = 3,
                 restriction = (0.7, 0.2, 0.1),
                 themes = ["physics", "spatial", "quantitative"],
                 edge_cost_names = ["water", "time", "energy"],
                 edge_cost_types = [int, int, int],
                 log_metadata = "graph_metadata.txt")

    # Export the graph with its named edge costs to a PNG in the current directory.
    image = generator.plot_graph(generator.graph, 
                                 filename="graph.png",
                                 edge_cost=generator.edge_cost_names,
                                 color="restriction",
                                 transition_probs=True)

    # Reuse exactly the constructor's formatting for console and file output.
    generator._print_metadata(image=image)
    if generator.log_metadata is not None:
        generator._write_metadata(image=image)


if __name__ == "__main__":
    # Support both direct script execution and python -m invocation.
    main()
