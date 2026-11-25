"""Render persisted ground-truth instances through the user's plotting API.

Images are evaluator artifacts only. They are never part of policy observations.
"""
from pathlib import Path

import networkx as nx

# Use the exact implementation requested by the user, in graph_world/, rather
# than reimplementing weighted plotting or constructing a new random graph.
from generate_graph_world import GraphGenerator


class SavedGraphPlotter(GraphGenerator):
    def __init__(self):
        # plot_graph needs these display settings, not constructor-generated
        # topology/endpoints/oracles. Inheriting its renderer preserves all edge
        # routing and collision-aware weight-label behavior on the SAVED graph.
        self.edge_costs = True
        self.themes = []


def export_graph_image(instance, filename):
    """Export precisely this instance's IDs, edges and ordered objective costs."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    graph = instance.graph.copy()
    # Source/target coloring remains readable even after serialization discarded
    # optional generator node annotations. It never modifies the actual instance.
    nx.set_node_attributes(graph, {
        node: 'source' if node == instance.source else 'target' if node == instance.target else 'node'
        for node in graph
    }, 'role')
    SavedGraphPlotter().plot_graph(graph, filename=str(filename),
        edge_cost=list(instance.names), color='role', transition_probs=False)
    return filename
