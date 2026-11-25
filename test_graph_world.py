"""Offline tests for graph-world API requests and scoring; no live API calls."""

import base64
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from io import StringIO
import hashlib
import json
import ast

import networkx as nx
import numpy as np
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

if __package__:
    from . import query_graph_world as evaluator
    from . import score_graph_world
else:
    import query_graph_world as evaluator
    import score_graph_world


class GraphWorldTests(unittest.TestCase):
    def model_answer(self, graph, paths):
        return repr({
            'pred_dict_of_lists': nx.to_dict_of_lists(graph),
            'pred_dict_of_dicts': {node: {child: data.get('weight', 1) for child, data in children.items()}
                                   for node, children in nx.to_dict_of_dicts(graph).items()},
            'pred_adjacency_matrix': nx.to_numpy_array(graph, nodelist=sorted(graph)).tolist(),
            'pred_shortest_paths': paths,
        })

    def scoring_fixture(self):
        a, b, c = (1, 0), (1, 1), (1, 2)
        graph = nx.DiGraph([(a, b), (a, c), (b, c)])
        truth = {**evaluator.graph_representations(graph), 'shortest_paths': [[a, c]]}
        prediction = evaluator.prediction_export(graph, [[a, c]], list(graph))
        return json.loads(json.dumps(truth)), json.loads(json.dumps(prediction))

    def task_answer(self, generator, prompt_type, n_paths=1):
        # Return only the fields requested by this task, using independent oracle attributes.
        answer = {}
        for attribute in score_graph_world.PROMPT_ATTRIBUTES[prompt_type]:
            value = getattr(generator, attribute)
            if isinstance(value, dict) and 'paths' in value:
                value = value['paths']
                if prompt_type == 'shortest_paths':
                    value = value[-n_paths:]
                elif prompt_type == 'scalarized':
                    value = value[:1]
            answer['pred_' + attribute] = score_graph_world.json_compatible(value)
        return repr(answer)

    def test_mdp_initialization_and_disabled_flag(self):
        generator = evaluator.GraphGenerator(2, 5, 2)
        self.assertIs(generator.mdp, True)
        self.assertEqual(set(nx.get_edge_attributes(generator.graph, 'transition_prob')), set(generator.graph.edges))
        self.assertEqual(generator.dict_of_dicts, nx.to_dict_of_dicts(generator.graph))
        self.assertTrue(nx.is_directed_acyclic_graph(generator.graph))
        with patch.object(evaluator.GraphGenerator, 'assign_transition_probs') as assign:
            disabled = evaluator.GraphGenerator(1, 5, 2, mdp=False)
        assign.assert_not_called()
        self.assertFalse(nx.get_edge_attributes(disabled.graph, 'transition_prob'))
        for invalid in (None, 0, 1, 'true'):
            with self.assertRaisesRegex(TypeError, 'mdp'):
                evaluator.GraphGenerator(mdp=invalid)

    def test_transition_probabilities_bounds_sums_and_seed(self):
        generator = evaluator.GraphGenerator(1, 5, 2, mdp=False)
        for degree in range(1, 11):
            for seed in range(10):
                graph = nx.DiGraph((0, child) for child in range(1, degree + 1))
                nx.set_edge_attributes(graph, 7, 'time')
                result = generator.assign_transition_probs(graph, seed=seed)
                self.assertIs(result, graph)
                probabilities = list(nx.get_edge_attributes(graph, 'transition_prob').values())
                self.assertTrue(all(type(probability) is float for probability in probabilities))
                self.assertAlmostEqual(sum(probabilities), 1.0)
                if degree == 1:
                    self.assertEqual(probabilities, [1.0])
                else:
                    self.assertTrue(all(0.1 <= probability <= 0.9 for probability in probabilities))
                self.assertTrue(all(data['time'] == 7 for _, _, data in graph.edges(data=True)))
                copy = graph.copy()
                generator.assign_transition_probs(copy, seed=seed)
                self.assertTrue(nx.utils.graphs_equal(graph, copy))
        self.assertIs(generator.assign_transition_probs(), generator.graph)
        self.assertEqual(list(generator.assign_transition_probs(nx.DiGraph()).edges), [])
        graph = nx.DiGraph([(0, 1), *((1, child) for child in range(2, 13))])
        original = graph.copy()
        with self.assertRaisesRegex(ValueError, 'at most 10'):
            generator.assign_transition_probs(graph)
        self.assertTrue(nx.utils.graphs_equal(graph, original))

    def test_plot_transition_probabilities_in_blue(self):
        generator = evaluator.GraphGenerator(1, 5, 2, edge_costs=True)
        graph = nx.DiGraph([(0, 1, {'time': 3, 'transition_prob': 0.1236789})])
        nx.set_node_attributes(graph, 'none', 'restriction')
        with tempfile.TemporaryDirectory() as directory:
            for enabled in (False, True):
                with patch.object(generator, '_place_plot_edge_labels',
                                  wraps=generator._place_plot_edge_labels) as place:
                    generator.plot_graph(graph, Path(directory) / f'prob_{enabled}.png',
                                         edge_cost='time', transition_probs=enabled)
                axes = place.call_args.args[0]
                red = [text.get_text() for text in axes.texts if text.get_color() == 'red']
                blue = [text.get_text() for text in axes.texts if text.get_color() == 'blue']
                self.assertEqual(red, ['3'])
                self.assertEqual(blue, ['0.124'] if enabled else [])
            generator.edge_costs = False
            with patch.object(generator, '_place_plot_edge_labels',
                              wraps=generator._place_plot_edge_labels) as place:
                generator.plot_graph(graph, Path(directory) / 'prob_only.png', transition_probs=True)
            axes = place.call_args.args[0]
            self.assertEqual([text.get_text() for text in axes.texts if text.get_color() == 'blue'], ['0.124'])
        self.assertEqual(graph[0][1]['transition_prob'], 0.1236789)
        with self.assertRaisesRegex(TypeError, 'transition_probs'):
            generator.plot_graph(graph, transition_probs='true')

    def test_restriction_annotations_and_disabled_flag(self):
        generator = evaluator.GraphGenerator(3, 5, 2)
        self.assertEqual(generator.restriction, (0.7, 0.2, 0.1))
        attributes = nx.get_node_attributes(generator.graph, 'restriction')
        self.assertEqual(set(attributes), set(generator.graph))
        self.assertTrue(set(attributes.values()) <= {'none', 'discouraged', 'forbidden'})
        self.assertTrue(all(type(value) is str for value in attributes.values()))
        self.assertTrue(all(not nx.get_node_attributes(graph, 'restriction')
                            for graph in generator.subgraphs.values()))
        disabled = evaluator.GraphGenerator(1, 5, 2, restriction=(0.0, 0.0, 0.0))
        self.assertFalse(nx.get_node_attributes(disabled.graph, 'restriction'))

    def test_restriction_probability_validation(self):
        for invalid in (None, True, 0.2, [0.8, 0.1, 0.1], (), (1.0, 0.0),
                        (1.0, 0.0, 0.0, 0.0), (True, 0.0, 0.0), ('1', 0.0, 0.0)):
            with self.subTest(value=invalid), self.assertRaisesRegex(TypeError, 'restriction'):
                evaluator.GraphGenerator(restriction=invalid)
        for probabilities in ((-0.1, 0.5, 0.6), (1.1, 0.0, 0.0), (0.1, 0.1, 0.1),
                              (0.0, 0.0, 0.1), (float('nan'), 0.0, 1.0),
                              (float('inf'), 0.0, 0.0)):
            with self.subTest(value=probabilities), self.assertRaisesRegex(ValueError, 'restriction'):
                evaluator.GraphGenerator(restriction=probabilities)

    def test_restriction_categorical_sampling(self):
        for probabilities, expected in (((1.0, 0.0, 0.0), 'none'),
                                        ((0.0, 1.0, 0.0), 'discouraged'),
                                        ((0.0, 0.0, 1.0), 'forbidden')):
            generator = evaluator.GraphGenerator(1, 5, 2, restriction=probabilities)
            attributes = nx.get_node_attributes(generator.graph, 'restriction')
            self.assertEqual(attributes, dict.fromkeys(generator.graph, expected))
        graph = nx.path_graph(4, create_using=nx.DiGraph)
        samples = ['none', 'discouraged', 'forbidden', 'none']
        rng = Mock()
        rng.choice.return_value = np.array(samples)
        with patch.object(np.random, 'default_rng', return_value=rng), \
                patch.object(evaluator.GraphGenerator, 'make_subgraph', return_value=graph.copy()), \
                patch.object(evaluator.GraphGenerator, 'make_graph', return_value=graph):
            generator = evaluator.GraphGenerator(1, 4, 2, restriction=(0.5, 0.3, 0.2), mdp=False)
        self.assertEqual(rng.choice.call_args.args, (('none', 'discouraged', 'forbidden'),))
        self.assertEqual(rng.choice.call_args.kwargs['size'], 4)
        np.testing.assert_allclose(rng.choice.call_args.kwargs['p'], generator.restriction)
        self.assertEqual(nx.get_node_attributes(generator.graph, 'restriction'), dict(enumerate(samples)))

    def test_plot_colors_by_selected_node_attribute(self):
        from matplotlib.legend import Legend
        from matplotlib.colors import to_rgba

        generator = evaluator.GraphGenerator(1, 5, 2)
        nx.set_node_attributes(generator.graph, {node: index % 2
                                               for index, node in enumerate(generator.graph)}, 'category')
        with tempfile.TemporaryDirectory() as directory:
            for attribute in ('restriction', 'theme', 'category'):
                with self.subTest(attribute=attribute), \
                        patch.object(nx, 'draw_networkx_nodes', wraps=nx.draw_networkx_nodes) as draw:
                    kwargs = {} if attribute == 'restriction' else {'color': attribute}
                    generator.plot_graph(generator.graph, Path(directory) / f'{attribute}.png', **kwargs)
                colors = draw.call_args.kwargs['node_color']
                values = nx.get_node_attributes(generator.graph, attribute)
                nodes = list(generator.graph)
                for i, first in enumerate(nodes):
                    for j, second in enumerate(nodes):
                        self.assertEqual(to_rgba(colors[i]) == to_rgba(colors[j]),
                                         values[first] == values[second])
                axes = draw.call_args.kwargs['ax']
                legends = [artist for artist in axes.get_children() if isinstance(artist, Legend)]
                self.assertEqual(len(legends), 1)
                self.assertEqual(legends[0].get_title().get_text(), attribute.capitalize())
                self.assertEqual({text.get_text() for text in legends[0].get_texts()},
                                 {str(value) for value in values.values()})
        for invalid in (None, False, 3):
            with self.assertRaises(TypeError):
                generator.plot_graph(generator.graph, color=invalid)
        with self.assertRaisesRegex(ValueError, 'No nodes'):
            generator.plot_graph(generator.graph, color='unknown')
        disabled = evaluator.GraphGenerator(1, 5, 2, restriction=(0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, 'restriction'):
            disabled.plot_graph(disabled.graph)

    def test_edge_costs_default_disabled(self):
        generator = evaluator.GraphGenerator(2, 5, 2)
        self.assertIs(generator.edge_costs, False)
        for graph in [generator.graph, *generator.subgraphs.values(),
                      generator.make_subgraph(5, 2, seed=42),
                      generator.make_graph(generator.subgraphs.values(), seed=42)]:
            self.assertTrue(all(not (set(data) & set(generator.edge_cost_names))
                                for _, _, data in graph.edges(data=True)))
        # Unit fallback costs preserve the path pipeline without editing edges.
        self.assertEqual(generator.multiobjective_shortest_paths['paths'],
                         generator.all_shortest_paths['paths'])
        self.assertIn(generator.scalar_dijkstra['paths'][0], generator.all_shortest_paths['paths'])
        for invalid in (None, 0, 1, 'false', []):
            with self.subTest(value=invalid), self.assertRaisesRegex(TypeError, 'edge_costs'):
                evaluator.GraphGenerator(edge_costs=invalid)

    def test_plot_respects_edge_costs_flag(self):
        from matplotlib.legend import Legend

        # Existing attributes must also stay hidden when cost generation is off.
        graph = nx.DiGraph([(0, 1, {'time': 3})])
        nx.set_node_attributes(graph, 'physics', name='theme')
        with tempfile.TemporaryDirectory() as directory:
            for enabled in (False, True):
                generator = evaluator.GraphGenerator(1, 5, 2, edge_costs=enabled)
                with patch.object(generator, '_place_plot_edge_labels',
                                  wraps=generator._place_plot_edge_labels) as place:
                    filename = Path(directory) / f'costs_{enabled}.png'
                    generator.plot_graph(graph, filename, edge_cost=['time'], color='theme')
                self.assertTrue(filename.is_file())
                axes, labels, *_ = place.call_args.args
                self.assertEqual([text for text, _ in labels], ['(3)'] if enabled else [])
                titles = [artist.get_title().get_text() for artist in axes.get_children()
                          if isinstance(artist, Legend)]
                self.assertEqual(any(title.startswith('Edge cost') for title in titles), enabled)

    def test_acyclic_detects_cycles_and_accepts_empty_graphs(self):
        check = evaluator.GraphGenerator.test_acyclic
        for graph in (nx.DiGraph(), nx.Graph(), nx.MultiDiGraph(), nx.MultiGraph(),
                      nx.DiGraph([(0, 1), (0, 2), (1, 2)]), nx.path_graph(4)):
            with self.subTest(graph=graph):
                self.assertIs(check(graph), True)
        # Include disconnected cycles, reciprocal edges, loops, and parallel edges.
        for graph in (nx.DiGraph([(0, 1), (1, 2), (2, 0)]),
                      nx.DiGraph([(0, 1), (1, 0)]), nx.DiGraph([(0, 0)]),
                      nx.cycle_graph(3), nx.Graph([(0, 0)]),
                      nx.MultiGraph([(0, 1), (0, 1)])):
            graph.add_node('isolated')
            with self.subTest(graph=graph):
                self.assertIs(check(graph), False)

    def test_generated_subgraphs_and_global_graph_are_acyclic(self):
        generator = evaluator.GraphGenerator(3, 5, 2)
        self.assertTrue(nx.is_directed_acyclic_graph(generator.graph))
        self.assertTrue(all(nx.is_directed_acyclic_graph(g) for g in generator.subgraphs.values()))
        # Exercise sparse and dense DAGs with many independently seeded orders.
        for seed in range(30):
            for n, d in ((3, 2), (5, 2), (5, 3), (5, 4)):
                with self.subTest(seed=seed, n=n, d=d):
                    graph = generator.make_subgraph(n, d, seed=seed)
                    self.assertTrue(nx.is_directed_acyclic_graph(graph))
                    order = list(nx.topological_sort(graph))
                    self.assertEqual([graph.out_degree(node) for node in order],
                                     [min(d, n - i - 1) for i in range(n)])
                    if d or n == 1:
                        self.assertTrue(nx.is_weakly_connected(graph))
            # Singleton groups and mixed node types also compose into a DAG.
            groups = [nx.DiGraph([(0, 1), (1, 2)]),
                      nx.DiGraph([('a', 'b')]), nx.empty_graph(1, create_using=nx.DiGraph)]
            for inputs in (groups, [groups[-1], groups[-1]]):
                graph = generator.make_graph(inputs, seed=seed)
                self.assertTrue(nx.is_directed_acyclic_graph(graph))
                self.assertTrue(nx.is_weakly_connected(graph))

    def test_minimum_subgraph_outdegree_and_renamed_setting(self):
        for invalid in (0, 1, (0, 2), (1, 3)):
            with self.subTest(setting=invalid), self.assertRaisesRegex(
                ValueError, 'max_subgraph_outdegree must be at least 2'
            ):
                evaluator.GraphGenerator(max_subgraph_outdegree=invalid)
        for setting in (2, (2, 3)):
            generator = evaluator.GraphGenerator(
                n_subgraphs=3, n_nodes_per_subgraph=5,
                max_subgraph_outdegree=setting,
            )
            self.assertEqual(generator.max_subgraph_outdegree, setting)
            self.assertNotIn('out_degree', vars(generator))
            for graph in generator.subgraphs.values():
                order = list(nx.topological_sort(graph))
                self.assertTrue(all(graph.out_degree(node) >= 2 for node in order[:-2]))
                self.assertEqual([graph.out_degree(node) for node in order[-2:]], [1, 0])
        for n, d in ((0, 0), (1, 0), (2, 1), (5, 0), (5, 1), (5, 5)):
            with self.subTest(n=n, d=d), self.assertRaisesRegex(ValueError, '2 <= d <= n - 1'):
                generator.make_subgraph(n, d)

    def test_make_graph_rejects_cyclic_inputs(self):
        generator = evaluator.GraphGenerator(1, 5, 2)
        for graph in (nx.DiGraph([(0, 1), (1, 2), (2, 0)]),
                      nx.DiGraph([(0, 1), (1, 0)]), nx.DiGraph([(0, 0)])):
            for groups in ([graph], [generator.graph, graph]):
                with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, 'acyclic'):
                    generator.make_graph(groups)

    def test_metadata_logging_filename_and_disabled_default(self):
        # Use a temporary working directory to check real filename handling safely.
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                generator = evaluator.GraphGenerator(1, 5, 2)
                self.assertIsNone(generator.log_metadata)
                self.assertEqual(list(Path(directory).iterdir()), [])

                # A custom filename enables logging with the same console formatting.
                filename = 'custom graph metadata.txt'
                generator = evaluator.GraphGenerator(1, 5, 2, log_metadata=filename)
                log = Path(filename)
                self.assertTrue(log.is_file())
                self.assertEqual(generator.log_metadata, filename)
                contents = log.read_text(encoding='utf-8')
                for name in vars(generator):
                    self.assertIn(f'{name}:\n', contents)
                self.assertIn('graph_attributes', contents)
                self.assertIn('task_complexity', contents)
                output = StringIO()
                generator._print_metadata(output)
                self.assertEqual(contents, output.getvalue())

                # None neither overwrites an existing log nor creates a new one.
                log.write_text('preserve existing log', encoding='utf-8')
                evaluator.GraphGenerator(1, 5, 2, log_metadata=None)
                self.assertEqual(log.read_text(), 'preserve existing log')
                log.unlink()
                evaluator.GraphGenerator(1, 5, 2)
                self.assertFalse(log.exists())
            finally:
                os.chdir(original_directory)

    def test_metadata_logging_requires_filename_or_none(self):
        # Reject booleans and other types instead of treating them as filenames.
        for value in (False, True, 0, 1, ['metadata.txt']):
            with self.subTest(value=value), self.assertRaises(TypeError):
                evaluator.GraphGenerator(log_metadata=value)
        with self.assertRaises(ValueError):
            evaluator.GraphGenerator(log_metadata='')

    def test_subgraph_themes_are_balanced_and_inherited(self):
        # Three five-node subgraphs need coordinated remainders for global balance.
        themes = ['physics', 'spatial', 'physics', 'quantitative']
        with patch.object(nx, 'set_node_attributes', wraps=nx.set_node_attributes) as assign:
            generator = evaluator.GraphGenerator(3, 5, 2, themes=themes)
        unique = set(themes)
        expected = {}
        for group, subgraph in generator.subgraphs.items():
            # Every subgraph uses floor/ceil quotas over distinct themes.
            assignments = nx.get_node_attributes(subgraph, 'theme')
            self.assertEqual(set(assignments), set(subgraph))
            counts = [sum(theme == name for theme in assignments.values()) for name in unique]
            self.assertLessEqual(max(counts) - min(counts), 1)
            expected.update({(group, node): theme for node, theme in assignments.items()})
        self.assertEqual(generator.node_themes, expected)
        self.assertEqual(nx.get_node_attributes(generator.graph, 'theme'), expected)
        counts = [sum(theme == name for theme in expected.values()) for name in unique]
        self.assertEqual(counts, [5, 5, 5])
        # Theme writes happen on subgraphs only; composition just copies attributes.
        theme_calls = [call for call in assign.call_args_list if call.kwargs.get('name') == 'theme']
        self.assertEqual(len(theme_calls), 3)
        self.assertTrue(all(call.args[0] is not generator.graph for call in theme_calls))

    def test_balanced_themes_cover_rounding_and_seeded_subgraphs(self):
        generator = evaluator.GraphGenerator(1, 5, 2)
        for size in (3, 4, 5, 6, 8):
            # Cover exact division and nonzero theme remainders.
            with self.subTest(size=size):
                first = generator.make_subgraph(size, d=2, seed=123)
                second = generator.make_subgraph(size, d=2, seed=123)
                self.assertTrue(nx.utils.graphs_equal(first, second))
                # Graph-level complexity is an inclusive 1..10 integer, also seeded.
                self.assertIs(type(first.graph['task_complexity']), int)
                self.assertTrue(1 <= first.graph['task_complexity'] <= 10)
                assignments = nx.get_node_attributes(first, 'theme')
                self.assertEqual(len(assignments), size)
                counts = [sum(theme == name for theme in assignments.values()) for name in generator.themes]
                self.assertLessEqual(max(counts) - min(counts), 1)
        # Different sampled subgraph sizes must also retain global rounding balance.
        generator = evaluator.GraphGenerator(2, (5, 6), 2)
        counts = [sum(theme == name for theme in generator.node_themes.values()) for name in generator.themes]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_configurable_subgraph_task_complexity(self):
        # Fixed values must be identical across all generated subgraphs.
        generator = evaluator.GraphGenerator(2, 5, 2, task_complexity=7)
        self.assertEqual(generator.task_complexity, 7)
        self.assertEqual([graph.graph['task_complexity'] for graph in generator.subgraphs.values()], [7, 7])
        for setting, lower, upper in (((3, 6), 3, 6), ((8, 8), 8, 8), (np.int64(4), 4, 4), ((-2, 1), -2, 1)):
            # Check configured ranges, single-value ranges, and integer normalization.
            with self.subTest(setting=setting):
                generator = evaluator.GraphGenerator(1, 5, 2, task_complexity=setting)
                for graph in generator.subgraphs.values():
                    self.assertIs(type(graph.graph['task_complexity']), int)
                    self.assertTrue(lower <= graph.graph['task_complexity'] <= upper)
                first = generator.make_subgraph(5, 2, seed=123)
                second = generator.make_subgraph(5, 2, seed=123)
                self.assertTrue(nx.utils.graphs_equal(first, second))
                self.assertTrue(lower <= first.graph['task_complexity'] <= upper)

    def test_invalid_task_complexity_configuration(self):
        # Reject noninteger values and tuples that do not contain exactly two bounds.
        for invalid in (True, np.bool_(True), 1.5, '3', [1, 3], (), (1,), (1, 2, 3), (1, 2.5), (False, 3)):
            with self.subTest(setting=invalid), self.assertRaises(TypeError):
                evaluator.GraphGenerator(1, 5, 2, task_complexity=invalid)
        with self.assertRaises(ValueError):
            evaluator.GraphGenerator(1, 5, 2, task_complexity=(5, 2))

    def test_internal_costs_are_inherited_and_connector_costs_are_sampled(self):
        # Check both internal and connecting arcs have typed, independently sampled costs.
        generator = evaluator.GraphGenerator(2, 5, 2, edge_cost_names=['time', 'energy'],
                                             edge_cost_types=[int, float], edge_costs=True)
        self.assertTrue(any(u[0] != v[0] for u, v in generator.graph.edges))
        # Global node prefixes must match the actual 1-based metadata keys.
        self.assertEqual({node[0] for node in generator.graph}, set(generator.subgraphs))
        for group, subgraph in generator.subgraphs.items():
            for node, data in subgraph.nodes(data=True):
                self.assertEqual({key: value for key, value in generator.graph.nodes[group, node].items()
                                  if key != 'restriction'}, data)
        for _, _, data in generator.graph.edges(data=True):
            self.assertIs(type(data['time']), int)
            self.assertIs(type(data['energy']), float)
            self.assertTrue(0 <= data['time'] <= 5)
            self.assertTrue(0.0 <= data['energy'] <= 5.0)

        # Internal costs already exist in each subgraph and survive relabeling exactly.
        for group, subgraph in generator.subgraphs.items():
            for parent, child, data in subgraph.edges(data=True):
                self.assertIs(type(data['time']), int)
                self.assertIs(type(data['energy']), float)
                self.assertEqual({key: value for key, value in generator.graph[(group, parent)][(group, child)].items()
                                  if key != 'transition_prob'}, data)

        # Initialization must compute path costs from the assigned edge attributes.
        for record in (generator.all_shortest_paths, generator.multiobjective_shortest_paths,
                       generator.scalar_dijkstra):
            # Each path index must have one dictionary of scalar cost values.
            self.assertEqual(set(record), {'paths', 'costs'})
            self.assertTrue(all(isinstance(path, list) for path in record['paths']))
            self.assertIsInstance(record['costs'], list)
            self.assertEqual(len(record['paths']), len(record['costs']))
            for costs in record['costs']:
                self.assertIsInstance(costs, dict)
                self.assertTrue(all(isinstance(cost, (int, float)) for cost in costs.values()))
        # Independently round each original edge before checking reported totals.
        quantum = Decimal(1).scaleb(-generator.cost_precision)
        for path, costs in zip(generator.multiobjective_shortest_paths['paths'],
                               generator.multiobjective_shortest_paths['costs']):
            self.assertEqual(set(costs), set(generator.edge_cost_names) | {'scalar_sum'})
            for name in generator.edge_cost_names:
                # Verify each dictionary belongs to the path at the same index.
                actual = sum(Decimal(str(generator.graph[u][v][name])).quantize(quantum)
                             for u, v in zip(path, path[1:]))
                self.assertEqual(costs[name], float(actual))
            self.assertEqual(costs['scalar_sum'],
                             float(sum(Decimal(str(costs[name])) for name in generator.edge_cost_names)))
        scalar_path = generator.scalar_dijkstra['paths'][0]
        scalar_cost = float(sum(Decimal(str(generator.graph[u][v][name])).quantize(quantum)
                                for u, v in zip(scalar_path, scalar_path[1:])
                                for name in generator.edge_cost_names))
        self.assertEqual(generator.scalar_dijkstra['costs'], [{'scalar_sum': scalar_cost}])
        self.assertEqual(generator.all_shortest_paths['costs'],
                         [{'distance': len(path) - 1} for path in generator.all_shortest_paths['paths']])
        expected = nx.dijkstra_path_length(generator.graph, generator.source, generator.target,
                                           weight=lambda u, v, data: sum(
                                               Decimal(str(data[name])).quantize(quantum)
                                               for name in generator.edge_cost_names))
        self.assertEqual(scalar_cost, float(expected))
        self.assertEqual(scalar_cost, min(costs['scalar_sum']
                                        for costs in generator.multiobjective_shortest_paths['costs']))

        # Fixed seeds reproduce topology and attributes without modifying inputs.
        originals = [graph.copy() for graph in generator.subgraphs.values()]
        first = generator.make_graph(generator.subgraphs.values(), seed=42)
        second = generator.make_graph(generator.subgraphs.values(), seed=42)
        self.assertTrue(nx.utils.graphs_equal(first, second))
        for graph, original in zip(generator.subgraphs.values(), originals):
            self.assertTrue(nx.utils.graphs_equal(graph, original))

        # Explicit input costs and unrelated attributes are copied without resampling.
        subgraph = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
        nx.set_edge_attributes(subgraph, 99, name='time')
        nx.set_edge_attributes(subgraph, 99.0, name='energy')
        subgraph[0][1]['note'] = 'preserved'
        original = subgraph.copy()
        graph = generator.make_graph([subgraph], seed=42)
        for parent, child, data in subgraph.edges(data=True):
            self.assertEqual(graph[(1, parent)][(1, child)], data)
        self.assertTrue(nx.utils.graphs_equal(subgraph, original))

        # Only the added inter-subgraph edges get new costs in the sampling range.
        graph = generator.make_graph([subgraph, subgraph.copy()], seed=42)
        for parent, child, data in graph.edges(data=True):
            if parent[0] == child[0]:
                self.assertEqual(data, subgraph[parent[1]][child[1]])
            else:
                self.assertIs(type(data['time']), int)
                self.assertIs(type(data['energy']), float)
                self.assertTrue(0 <= data['time'] <= 5)
                self.assertTrue(0.0 <= data['energy'] <= 5.0)
        self.assertTrue(nx.utils.graphs_equal(subgraph, original))

    def test_edge_cost_configuration_validation(self):
        # Supply one type per name, matching the constructor's cost configuration.
        generator = evaluator.GraphGenerator(1, 5, 2, edge_cost_names=['time'], edge_cost_types=[int], edge_costs=True)
        self.assertEqual(generator.edge_cost_types, [int])
        self.assertTrue(all(type(data['time']) is int for _, _, data in generator.graph.edges(data=True)))
        for names, types in ((['time'], []), (['time', 'energy'], [int]),
                             (['time'], [str]), (['time'], [bool]),
                             (['time', 'time'], [int, int]), ([''], [int]),
                             (['scalar_sum'], [int])):
            # Reject mismatches and unsupported names/types before generating a graph.
            with self.subTest(names=names, types=types), self.assertRaises(ValueError):
                evaluator.GraphGenerator(1, 5, 2, edge_cost_names=names, edge_cost_types=types)

    def test_generator_initializes_multiobjective_paths_without_editing_costs(self):
        # Two equal-hop routes trade off time against distance.
        graph = nx.DiGraph()
        graph.add_edge('s', 'a', time=1, distance=5)
        graph.add_edge('a', 't', time=1, distance=5)
        graph.add_edge('s', 'b', time=5, distance=1)
        graph.add_edge('b', 't', time=5, distance=1)
        original = graph.copy()
        with patch.object(evaluator.GraphGenerator, 'make_graph', return_value=graph):
            generator = evaluator.GraphGenerator(1, 5, 2, restriction=(0.0, 0.0, 0.0), mdp=False, edge_cost_names=['time', 'distance'], edge_cost_types=[int, int])
        self.assertEqual((generator.source, generator.target), ('s', 't'))
        self.assertEqual(generator.multiobjective_shortest_paths,
                         {'paths': [['s', 'a', 't'], ['s', 'b', 't']],
                          'costs': [{'time': 2, 'distance': 10, 'scalar_sum': 12},
                                    {'time': 10, 'distance': 2, 'scalar_sum': 12}]})
        self.assertEqual(generator.all_shortest_paths,
                         {'paths': [['s', 'a', 't'], ['s', 'b', 't']], 'costs': [{'distance': 2}, {'distance': 2}]})
        nx.set_node_attributes(original, generator.node_themes, name='theme')
        self.assertTrue(nx.utils.graphs_equal(graph, original))

        # Unit defaults make unweighted routes minimize hops without adding attributes.
        unweighted = nx.DiGraph(graph.edges)
        for objectives, expected_cost in ((None, (2,)), (['time', 'distance'], (2, 2))):
            with patch.object(evaluator.GraphGenerator, 'make_graph', return_value=unweighted):
                generator = evaluator.GraphGenerator(1, 5, 2, restriction=(0.0, 0.0, 0.0), mdp=False, edge_cost_names=objectives or [],
                                                     edge_cost_types=[int] * len(objectives or []))
            self.assertEqual(generator.multiobjective_shortest_paths['paths'],
                             generator.all_shortest_paths['paths'])
            self.assertEqual(len(generator.multiobjective_shortest_paths['paths']), 2)
            names = objectives or ['weight']
            self.assertEqual(generator.multiobjective_shortest_paths['costs'],
                             [{**dict(zip(names, expected_cost)), 'scalar_sum': sum(expected_cost)}] * 2)
            self.assertTrue(all(not data for _, _, data in unweighted.edges(data=True)))

    def test_generator_reuses_paths_and_applies_precision_consistently(self):
        # All four routes tie on a two-decimal grid; source and target are unambiguous.
        graph = nx.DiGraph()
        for node, first, second in (('a', 0.1, 0.2), ('b', 0.3, 0.0),
                                    ('c', 0.1 + 0.2, 0.0), ('d', 0.30001, 0.0)):
            graph.add_edge('s', node, time=first, energy=0.1)
            graph.add_edge(node, 't', time=second, energy=0.2)
        original = graph.copy()
        with patch.object(evaluator.GraphGenerator, 'make_graph', return_value=graph), \
                patch.object(nx, 'all_simple_paths', wraps=nx.all_simple_paths) as enumerate_paths:
            generator = evaluator.GraphGenerator(1, 5, 2, restriction=(0.0, 0.0, 0.0), mdp=False, edge_cost_names=['time', 'energy'],
                                                 edge_cost_types=[float, float], cost_precision=2)
        # Initialization must traverse the graph for simple paths exactly once.
        self.assertEqual(enumerate_paths.call_count, 1)
        self.assertEqual(generator.cost_precision, 2)
        self.assertEqual(generator.multiobjective_shortest_paths['paths'], generator.all_paths)
        self.assertEqual(generator.multiobjective_shortest_paths['costs'],
                         [{'time': 0.3, 'energy': 0.3, 'scalar_sum': 0.6}] * 4)
        self.assertIn(generator.scalar_dijkstra['paths'][0], generator.multiobjective_shortest_paths['paths'])
        self.assertEqual(generator.scalar_dijkstra['costs'], [{'scalar_sum': 0.6}])
        self.assertTrue(nx.utils.graphs_equal(graph, original))

    def test_high_precision_scalar_totals_do_not_resum_float_outputs(self):
        # Rounding the objective totals to output floats must not affect scalar sums.
        graph = nx.DiGraph()
        graph.add_edge('s', 'a', time=0.10000000000000006, energy=0.01)
        graph.add_edge('a', 't', time=0.20000000000000007, energy=0.0)
        with patch.object(evaluator.GraphGenerator, 'make_graph', return_value=graph):
            generator = evaluator.GraphGenerator(1, 5, 2, restriction=(0.0, 0.0, 0.0), mdp=False, edge_cost_names=['time', 'energy'],
                                                 edge_cost_types=[float, float], cost_precision=17)
        expected = float(Decimal('0.10000000000000006') + Decimal('0.20000000000000007') + Decimal('0.01'))
        self.assertEqual(generator.multiobjective_shortest_paths['costs'][0]['scalar_sum'], expected)
        self.assertEqual(generator.scalar_dijkstra['costs'], [{'scalar_sum': expected}])

    def test_generator_rejects_invalid_precision_and_reserved_name_before_generation(self):
        # Invalid policies and summary-name collisions must fail before expensive work.
        with patch.object(evaluator.GraphGenerator, 'make_graph', side_effect=AssertionError('Unexpected graph')):
            for invalid in (-1, True, 1.5, '2', None):
                with self.subTest(precision=invalid), self.assertRaisesRegex(ValueError, 'cost_precision'):
                    evaluator.GraphGenerator(cost_precision=invalid)
            with self.assertRaisesRegex(ValueError, 'scalar_sum.*reserved'):
                evaluator.GraphGenerator(edge_cost_names=['scalar_sum'], edge_cost_types=[int])

    def test_scalarized_dijkstra_initialization_and_vector_costs(self):
        # Summing objectives must choose b, even though a minimizes time alone.
        graph = nx.DiGraph()
        graph.add_edge('s', 'a', time=1, distance=8)
        graph.add_edge('a', 't', time=1, distance=8)
        graph.add_edge('s', 'b', time=5, distance=1)
        graph.add_edge('b', 't', time=5, distance=1)
        original = graph.copy()
        with patch.object(evaluator.GraphGenerator, 'make_graph', return_value=graph):
            generator = evaluator.GraphGenerator(1, 5, 2, restriction=(0.0, 0.0, 0.0), mdp=False, edge_cost_names=['time', 'distance'], edge_cost_types=[int, int])
        self.assertEqual(generator.scalar_dijkstra, {'paths': [['s', 'b', 't']], 'costs': [{'scalar_sum': 12}]})
        self.assertEqual(generator.scalarized_dijkstra(), ['s', 'b', 't'])
        nx.set_node_attributes(original, generator.node_themes, name='theme')
        self.assertTrue(nx.utils.graphs_equal(graph, original))

        # A tuple-valued cost can favor a longer route over the direct edge.
        generator.edge_cost_names = []
        generator.graph = nx.DiGraph()
        generator.graph.add_edge('s', 't', cost=(10, 10))
        generator.graph.add_edge('s', 'a', cost=(1, 2))
        generator.graph.add_edge('a', 't', cost=(1, 2))
        original = generator.graph.copy()
        self.assertEqual(generator.scalarized_dijkstra('s', 't'), ['s', 'a', 't'])
        self.assertTrue(nx.utils.graphs_equal(generator.graph, original))

        # Reject malformed weights instead of running Dijkstra with invalid costs.
        for invalid in ((-1, 2), (float('nan'), 1), (float('inf'), 1), (), ((1, 2),)):
            generator.graph['s']['t']['cost'] = invalid
            with self.subTest(cost=invalid), self.assertRaises(ValueError):
                generator.scalarized_dijkstra('s', 't')

        # Without costs the same helper minimizes hop count using unit weights.
        generator.graph = nx.DiGraph(original.edges)
        self.assertEqual(generator.scalarized_dijkstra(), ['s', 't'])
        with self.assertRaises(nx.NodeNotFound):
            generator.scalarized_dijkstra('missing', 't')
        with self.assertRaises(nx.NetworkXNoPath):
            generator.scalarized_dijkstra('t', 's')

    def test_graph_edit_distance_wrapper_and_networkx_options(self):
        target = np.array([[0, 1], [0, 0]])
        reversed_edge = target.T
        self.assertEqual(score_graph_world.graph_edit_distance(target, reversed_edge), 0.0)
        self.assertEqual(score_graph_world.structural_hamming_distance(target, reversed_edge), 2)
        self.assertEqual(score_graph_world.graph_edit_distance(target, np.zeros((2, 2))), 1.0)
        self.assertEqual(score_graph_world.graph_edit_distance(target, target * 5), 0.0)
        self.assertEqual(score_graph_world.graph_edit_distance(np.zeros((2, 2)), np.zeros((3, 3))), 1.0)
        self.assertEqual(score_graph_world.graph_edit_distance([], []), 0.0)
        # Upper bounds preserve NetworkX's None result when no solution fits.
        self.assertIsNone(score_graph_world.graph_edit_distance(target, np.zeros((2, 2)), upper_bound=0))
        with patch.object(score_graph_world.nx, 'graph_edit_distance', return_value=3.0) as ged:
            self.assertEqual(score_graph_world.graph_edit_distance(target, reversed_edge, timeout=0.5), 3.0)
            graphs = ged.call_args.args
            self.assertTrue(all(graph.is_directed() for graph in graphs))
            self.assertEqual(set(graphs[0].edges), {(0, 1)})
            self.assertEqual(set(graphs[1].edges), {(1, 0)})
            self.assertEqual(ged.call_args.kwargs, {'timeout': 0.5})
        for invalid in ([[0, 1]], [[float('nan')]], [1, 2]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                score_graph_world.graph_edit_distance(target, invalid)

    def test_graph_edit_distance_is_scored_for_each_representation(self):
        truth, prediction = self.scoring_fixture()
        prediction['pred_adjacency_matrix'] = np.zeros((3, 3)).tolist()
        prediction['pred_dict_of_lists'] = {'(1, 0)': [[1, 1]], '(1, 1)': [], '(1, 2)': []}
        scores = score_graph_world.score_prediction(truth, prediction)
        self.assertEqual(scores['pred_adjacency_matrix']['graph_edit_distance'], 3.0)
        self.assertEqual(scores['pred_dict_of_lists']['graph_edit_distance'], 2.0)
        self.assertEqual(scores['pred_dict_of_dicts']['graph_edit_distance'], 0.0)
        records = {'1': {**truth, 'openai/test': prediction},
                   '2': {**truth, 'openai/test': evaluator.prediction_export(None, None, [])}}
        summary = score_graph_world.score_replicates(records)['summary']['openai/test']
        self.assertEqual(summary['pred_adjacency_matrix']['ged_count'], 1)
        self.assertEqual(summary['pred_adjacency_matrix']['mean_graph_edit_distance'], 3.0)
        missing = score_graph_world.score_replicates({'1': records['2']})
        self.assertIsNone(missing['summary']['openai/test']['pred_adjacency_matrix']['mean_graph_edit_distance'])

    def test_independent_graph_scores_and_strict_dictionary_equality(self):
        truth, prediction = self.scoring_fixture()
        scores = score_graph_world.score_prediction(truth, prediction)
        for field in score_graph_world.GRAPH_FIELDS:
            self.assertEqual(scores[field], {'exact_match': 1, 'structural_hamming_distance': 0, 'graph_edit_distance': 0.0})
        # Reordering children fails literal dictionary equality but preserves structure.
        prediction['pred_dict_of_lists']['(1, 0)'].reverse()
        scores = score_graph_world.score_prediction(truth, prediction)
        self.assertEqual(scores['pred_dict_of_lists'], {'exact_match': 0, 'structural_hamming_distance': 0, 'graph_edit_distance': 0.0})
        self.assertEqual(scores['pred_dict_of_dicts']['exact_match'], 1)
        # Change only one dictionary's weight; SHD ignores weights.
        prediction['pred_dict_of_dicts']['(1, 0)']['(1, 1)'] = 9
        scores = score_graph_world.score_prediction(truth, prediction)
        self.assertEqual(scores['pred_dict_of_dicts'], {'exact_match': 0, 'structural_hamming_distance': 0, 'graph_edit_distance': 0.0})
        self.assertEqual(scores['pred_adjacency_matrix']['exact_match'], 1)
        # Reverse an edge in the matrix only: one missing and one extra edge.
        prediction['pred_adjacency_matrix'][0][1] = 0
        prediction['pred_adjacency_matrix'][1][0] = 1
        scores = score_graph_world.score_prediction(truth, prediction)
        self.assertEqual(scores['pred_adjacency_matrix'], {'exact_match': 0, 'structural_hamming_distance': 2, 'graph_edit_distance': 0.0})
        prediction['pred_dict_of_lists']['(1, 0)'].remove([1, 1])
        prediction['pred_dict_of_lists']['(1, 1)'].append([1, 0])
        scores = score_graph_world.score_prediction(truth, prediction)
        self.assertEqual(scores['pred_dict_of_lists']['structural_hamming_distance'], 2)
        self.assertEqual(scores['pred_dict_of_dicts']['structural_hamming_distance'], 0)

    def test_matrix_alignment_extra_nodes_and_malformed_predictions(self):
        truth, prediction = self.scoring_fixture()
        original = np.asarray(prediction['pred_adjacency_matrix'])
        prediction['pred_node_order'].reverse()
        prediction['pred_adjacency_matrix'] = original[::-1, ::-1].tolist()
        self.assertEqual(score_graph_world.score_prediction(truth, prediction)['pred_adjacency_matrix']['exact_match'], 1)
        prediction['pred_node_order'].append([9, 9])
        prediction['pred_adjacency_matrix'] = np.pad(original[::-1, ::-1], ((0, 1), (0, 1))).tolist()
        prediction['pred_adjacency_matrix'][3][0] = 1
        score = score_graph_world.score_prediction(truth, prediction)['pred_adjacency_matrix']
        self.assertEqual(score, {'exact_match': 0, 'structural_hamming_distance': 1, 'graph_edit_distance': 2.0})
        prediction['pred_dict_of_lists']['(9, 9)'] = [[1, 0]]
        self.assertEqual(score_graph_world.score_prediction(truth, prediction)['pred_dict_of_lists']['structural_hamming_distance'], 1)
        prediction['pred_adjacency_matrix'] = [[1]]
        prediction['pred_dict_of_dicts'] = {'(1, 0)': {'(1, 1)': 'invalid weight'}}
        prediction['pred_dict_of_lists'] = None
        scores = score_graph_world.score_prediction(truth, prediction)
        for field in score_graph_world.GRAPH_FIELDS:
            self.assertEqual(scores[field]['exact_match'], 0)
            self.assertIsNone(scores[field]['structural_hamming_distance'])
            self.assertIsNone(scores[field]['graph_edit_distance'])
        self.assertIn('error', scores['pred_adjacency_matrix'])
        self.assertIn('error', scores['pred_dict_of_dicts'])

    def test_dictionary_conversion_includes_missing_nodes_and_weights(self):
        truth, prediction = self.scoring_fixture()
        prediction['pred_dict_of_dicts'] = {'(1, 1)': {'(1, 0)': 2}, '(1, 2)': {}}
        scores = score_graph_world.score_prediction(truth, prediction)
        self.assertEqual(scores['pred_dict_of_dicts'], {'exact_match': 0, 'structural_hamming_distance': 4, 'graph_edit_distance': 2.0})
        # Missing an isolated node fails exact dictionary matching even at zero SHD.
        truth['dict_of_lists']['(2, 0)'] = []
        truth['dict_of_dicts']['(2, 0)'] = {}
        truth['node_order'].append([2, 0])
        truth['adjacency_matrix'] = np.pad(truth['adjacency_matrix'], ((0, 1), (0, 1))).tolist()
        scores = score_graph_world.score_prediction(truth, self.scoring_fixture()[1])
        self.assertEqual(scores['pred_dict_of_lists'], {'exact_match': 0, 'structural_hamming_distance': 0, 'graph_edit_distance': 1.0})
        self.assertEqual(scores['pred_adjacency_matrix'], {'exact_match': 0, 'structural_hamming_distance': 0, 'graph_edit_distance': 1.0})

    def test_shortest_path_scoring_and_summary(self):
        a, b, c, d = (1, 0), (1, 1), (1, 2), (1, 3)
        paths = [[a, b, d], [a, c, d]]
        score = score_graph_world.score_shortest_paths(paths, [paths[1]], n_paths=1)
        self.assertEqual(score, {'correct': 1, 'exact_match': 0, 'path_correctness': [True]})
        self.assertEqual(score_graph_world.score_shortest_paths(paths, paths[::-1], 2)['exact_match'], 1)
        score = score_graph_world.score_shortest_paths(paths, [paths[0], paths[0]], 2)
        self.assertEqual(score, {'correct': 0, 'exact_match': 0, 'path_correctness': [True, True]})
        self.assertEqual(score_graph_world.score_shortest_paths(paths, [[a, d]], 1)['correct'], 0)
        truth, prediction = self.scoring_fixture()
        prediction.pop('pred_adjacency_matrix')  # Legacy spelling still scores.
        records = {'1': {**truth, 'openai/test': prediction},
                   '2': {**truth, 'openai/test': evaluator.prediction_export(None, None, [])}}
        summary = score_graph_world.score_replicates(records)['summary']['openai/test']
        for field in (*score_graph_world.GRAPH_FIELDS, 'pred_shortest_paths'):
            self.assertEqual(summary[field]['attempted'], 2)
            self.assertEqual(summary[field]['accuracy'], 0.5)
        self.assertEqual(summary['pred_adjacency_matrix']['shd_count'], 1)
        self.assertEqual(summary['pred_adjacency_matrix']['mean_structural_hamming_distance'], 0)
        self.assertEqual(summary['pred_adjacency_matrix']['ged_count'], 1)
        self.assertEqual(summary['pred_adjacency_matrix']['ged_total'], 0.0)
        self.assertEqual(summary['pred_adjacency_matrix']['mean_graph_edit_distance'], 0.0)

    def test_score_saved_run_uses_exported_fields_without_rebuilding(self):
        truth, prediction = self.scoring_fixture()
        prediction['pred_dict_of_lists']['(1, 0)'] = []
        records = {'1': {**truth, 'openai/test': prediction}}
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluator, 'query_model') as query:
            export = Path(directory) / 'graph_replicates.json'
            export.write_text(json.dumps(records))
            with patch('sys.argv', ['query_graph_world.py', '--score-run', directory]), redirect_stdout(StringIO()):
                evaluator.main()
            query.assert_not_called()
            self.assertEqual(json.loads(export.read_text()), records)
            scores = json.loads((Path(directory) / 'scores.json').read_text())
            model = scores['replicates']['1']['openai/test']
            self.assertEqual(model['pred_dict_of_lists']['structural_hamming_distance'], 2)
            self.assertEqual(model['pred_dict_of_dicts']['exact_match'], 1)
            self.assertEqual(model['pred_adjacency_matrix']['exact_match'], 1)

    def test_weighted_graph_serialization_and_prediction_alignment(self):
        a, b, isolated, invented = (1, 0), (1, 1), (1, 2), (9, 9)
        graph = nx.DiGraph()
        graph.add_nodes_from([a, b, isolated])
        graph.add_edge(a, b, weight=2.5)
        graph.add_edge(b, a)
        forms = json.loads(json.dumps(evaluator.graph_representations(graph)))
        self.assertEqual(forms['adjacency_matrix'], [[0, 2.5, 0], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(forms['dict_of_dicts'], {repr(a): {repr(b): 2.5},
                                                repr(b): {repr(a): 1}, repr(isolated): {}})
        adjacency = {ast.literal_eval(node): [tuple(child) for child in children]
                     for node, children in forms['dict_of_lists'].items()}
        restored = nx.from_dict_of_lists(adjacency, create_using=nx.DiGraph)
        self.assertEqual(set(restored.edges), set(graph.edges))
        weights = {ast.literal_eval(node): {ast.literal_eval(child): {'weight': weight}
                                           for child, weight in children.items()}
                   for node, children in forms['dict_of_dicts'].items()}
        restored = nx.from_dict_of_dicts(weights, create_using=nx.DiGraph)
        np.testing.assert_array_equal(nx.to_numpy_array(restored), forms['adjacency_matrix'])
        predicted = nx.DiGraph([(b, invented)])
        result = evaluator.prediction_export(predicted, [[b, invented]], [a, b, isolated])
        self.assertEqual(result['pred_node_order'], [a, b, isolated, invented])
        self.assertEqual(result['pred_adjacency_array'][1][3], 1)
        self.assertEqual(result['pred_dict_of_lists'][repr(a)], [])
        self.assertEqual(list(predicted), [b, invented])

    def test_prediction_parser_preserves_independent_fields(self):
        graph = nx.DiGraph([((1, 0), (1, 1))])
        prediction = evaluator.parse_prediction(self.model_answer(graph, [[(1, 0), (1, 1)]]))
        self.assertEqual(prediction['pred_dict_of_lists'], {'(1, 0)': [[1, 1]], '(1, 1)': []})
        self.assertEqual(prediction['pred_node_order'], [(1, 0), (1, 1)])
        for invalid in ('[]', 'not python', repr({})):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluator.parse_prediction(invalid)
        raw = ast.literal_eval(self.model_answer(graph, [[(1, 0), (1, 1)]]))
        raw['pred_adjacency_matrix'] = [[0, 0], [1, 0]]
        raw['pred_dict_of_dicts'] = None
        prediction = evaluator.parse_prediction(repr(raw))
        self.assertEqual(prediction['pred_adjacency_matrix'], raw['pred_adjacency_matrix'])
        self.assertIsNone(prediction['pred_dict_of_dicts'])
        self.assertEqual(prediction['pred_dict_of_lists']['(1, 0)'], [[1, 1]])

    def test_live_evaluation_keeps_disagreeing_predictions(self):
        graph = evaluator.GraphGenerator(1, 5, 2)
        def query(provider, model, key, png, prompt, *args, **kwargs):
            prompt_type = next(task for task, attributes in score_graph_world.PROMPT_ATTRIBUTES.items()
                               if "'pred_" + attributes[0] + "'" in prompt)
            answer = ast.literal_eval(self.task_answer(graph, prompt_type))
            if prompt_type == 'grap_rep':
                answer['pred_adjacency_matrix'] = np.zeros((len(graph.graph), len(graph.graph))).tolist()
            return repr(answer), None
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret'}, clear=True), \
                patch.object(evaluator, 'GraphGenerator', return_value=graph), \
                patch.object(evaluator, 'query_model', side_effect=query), redirect_stdout(StringIO()):
            report = evaluator.run_evaluation([('openai', 'test')], 1, directory)
            run_dir = Path(report['run_directory'])
            scores = json.loads((run_dir / 'scores.json').read_text())
            graph_scores = scores['grap_rep']['scores']['1']['openai/test']
            self.assertFalse(graph_scores['adjacency_matrix']['correct'])
            self.assertEqual(graph_scores['adjacency_matrix']['structural_hamming_distance'], graph.graph.number_of_edges())
            self.assertEqual(graph_scores['dict_of_lists'], {'correct': True})
            self.assertEqual(graph_scores['dict_of_dicts'], {'correct': True})
            self.assertEqual(scores['shortest_paths']['scores']['1']['openai/test']['all_shortest_paths']['precision'], 1)
            evaluator.export_saved_run(run_dir)
            self.assertEqual(json.loads((run_dir / 'scores.json').read_text()), scores)


    def test_export_existing_legacy_run_without_api_calls(self):
        a, b = [1, 0], [1, 1]
        report = {
            'config': {'models': [['openai', 'legacy'], ['openai', 'failed']]},
            'replicates': [{
                'replicate': 1, 'source': a, 'sink': b, 'nodes': [a, b], 'edges': [[a, b]],
                'shortest_length': 1, 'shortest_paths': [[a, b]],
                'responses': [{'provider': 'openai', 'model': 'legacy', 'paths': [[a, b]],
                               'response': '[(1, 0), (1, 1)]'}],
            }],
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluator, 'query_model') as query:
            results = Path(directory) / 'run_metadata.json'
            results.write_text(json.dumps(report))
            with patch('sys.argv', ['query_graph_world.py', '--export-run', directory]), redirect_stdout(StringIO()):
                evaluator.main()
            query.assert_not_called()
            self.assertEqual(json.loads(results.read_text()), report)
            scores = json.loads((Path(directory) / 'scores.json').read_text())
            self.assertEqual(scores['replicates']['1']['openai/legacy']['pred_shortest_paths']['correct'], 1)
            self.assertIsNone(scores['replicates']['1']['openai/legacy']['pred_adjacency_matrix']['structural_hamming_distance'])
            record = json.loads((Path(directory) / 'graph_replicates.json').read_text())['1']
            self.assertEqual(record['adjacency_matrix'], [[0, 1], [0, 0]])
            self.assertEqual(record['openai/legacy']['pred_shortest_paths'], [[a, b]])
            self.assertIsNone(record['openai/legacy']['pred_dict_of_lists'])
            self.assertIsNone(record['openai/failed']['pred_shortest_paths'])

    def test_dotenv_preserves_environment_and_does_not_execute_values(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / '.env'
            env.write_text(
                '# comment\nEXISTING=replace\nexport NEW="a value" # comment\n'
                "LITERAL='$(not-a-shell-command)'\nPLAIN=abc # comment\n", encoding='utf-8',
            )
            with patch.dict(os.environ, {'EXISTING': 'keep'}, clear=True):
                evaluator.load_env(env)
                self.assertEqual(os.environ['EXISTING'], 'keep')
                self.assertEqual(os.environ['NEW'], 'a value')
                self.assertEqual(os.environ['LITERAL'], '$(not-a-shell-command)')
                self.assertEqual(os.environ['PLAIN'], 'abc')

    def test_strict_safe_path_parsing(self):
        self.assertEqual(evaluator.parse_path(' [(0, 1), (2, 3)]\n'), [(0, 1), (2, 3)])
        self.assertEqual(evaluator.parse_path('[]'), [])
        for text in (
            '[[0, 1], [2, 3]]', '[(True, 1)]', '[(0, 1.0)]', '[(0, 1, 2)]',
            '```python\n[(0,1)]\n```', 'The path is [(0,1)]',
            "__import__('os').system('should-never-run')",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                evaluator.parse_path(text)

    def test_all_provider_payloads_include_png_and_task(self):
        answer = '[(0, 0), (1, 2)]'
        fixtures = {
            'openai': {'status': 'completed', 'output': [{'content': [
                {'type': 'output_text', 'text': answer},
            ]}], 'usage': {'output_tokens': 12}},
            'anthropic': {'stop_reason': 'end_turn', 'content': [
                {'type': 'thinking', 'thinking': 'private'}, {'type': 'text', 'text': answer},
            ]},
            'gemini': {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
                {'text': 'private', 'thought': True}, {'text': answer},
            ]}}]},
        }
        png = b'\x89PNG\r\n\x1a\nimage-bytes'
        prompt = evaluator.make_prompt((0, 0), (1, 2))
        for provider, fixture in fixtures.items():
            with self.subTest(provider=provider), patch.object(evaluator, 'post_json', return_value=fixture) as post:
                text, _ = evaluator.query_model(provider, 'test-model', 'fake-key', png, prompt)
                self.assertEqual(text, answer)
                url, headers, payload, timeout = post.call_args.args
                self.assertTrue(url.startswith('https://'))
                self.assertNotIn('fake-key', url)
                self.assertNotIn('fake-key', json.dumps(payload))
                if provider == 'openai':
                    content = payload['input'][0]['content']
                    encoded = content[0]['image_url'].split(',', 1)[1]
                    self.assertEqual(content[1]['text'], prompt)
                    self.assertFalse(payload['store'])
                elif provider == 'anthropic':
                    content = payload['messages'][0]['content']
                    encoded = content[0]['source']['data']
                    self.assertEqual(content[1]['text'], prompt)
                else:
                    content = payload['contents'][0]['parts']
                    encoded = content[0]['inlineData']['data']
                    self.assertEqual(content[1]['text'], prompt)
                self.assertEqual(base64.b64decode(encoded), png)

    def test_effort_payloads_and_omitted_defaults(self):
        answer = '[(0, 0), (1, 2)]'
        fixture = {
            'status': 'completed', 'output': [{'content': [{'type': 'output_text', 'text': answer}]}],
            'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': answer}],
            'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': answer}]}}],
        }
        for provider, levels in evaluator.EFFORT_LEVELS.items():
            for effort in (None, *levels):
                with self.subTest(provider=provider, effort=effort), patch.object(evaluator, 'post_json', return_value=fixture) as post:
                    evaluator.query_model(provider, 'test-model', 'fake-key', b'png', 'task', effort=effort)
                    payload = post.call_args.args[2]
                    if provider == 'openai':
                        self.assertEqual(payload.get('reasoning'), None if effort is None else {'effort': effort})
                        self.assertNotIn('output_config', payload)
                    elif provider == 'anthropic':
                        self.assertEqual(payload.get('output_config'), None if effort is None else {'effort': effort})
                        self.assertNotIn('reasoning', payload)
                    else:
                        self.assertEqual(payload['generationConfig'].get('thinkingConfig'),
                                         None if effort is None else {'thinkingLevel': effort})
                        self.assertNotIn('reasoning', payload)

    def test_cli_accepts_model_lists_repeated_flags_and_efforts(self):
        argv = [
            'query_graph_world.py', '--model', 'openai:first', 'anthropic:second',
            '--model', 'gemini:third', 'openai:first', '--openai-effort', 'high',
            '--anthropic-effort', 'medium', '--gemini-effort', 'low',
            '--n-replicates', '2', '--nodes-per-subgraph', '5', '6',
        ]
        with patch('sys.argv', argv), patch.object(evaluator, 'load_env'), \
                patch.object(evaluator, 'run_evaluation') as run:
            evaluator.main()
            self.assertEqual(run.call_args.args[0], [('openai', 'first'), ('anthropic', 'second'), ('gemini', 'third')])
            self.assertEqual(run.call_args.args[1], 2)
            self.assertEqual(run.call_args.kwargs['n_nodes_per_subgraph'], (5, 6))
            self.assertEqual(run.call_args.kwargs['openai_effort'], 'high')
            self.assertEqual(run.call_args.kwargs['anthropic_effort'], 'medium')
            self.assertEqual(run.call_args.kwargs['gemini_effort'], 'low')
        with patch('sys.argv', ['query_graph_world.py', '--model', 'openai:first']), \
                patch.object(evaluator, 'load_env'), patch.object(evaluator, 'run_evaluation') as run:
            evaluator.main()
            for provider in evaluator.KEY_NAMES:
                self.assertEqual(run.call_args.kwargs[provider + '_effort'], evaluator.DEFAULT_EFFORTS[provider])

    def test_cli_uses_requested_default_models_when_flag_is_omitted(self):
        with patch('sys.argv', ['query_graph_world.py']), patch.object(evaluator, 'load_env'), \
                patch.object(evaluator, 'run_evaluation') as run:
            evaluator.main()
            self.assertEqual(run.call_args.args[0], [
                ('openai', 'gpt-6.1-sol'),
                ('anthropic', 'claude-fable-5-1'),
                ('gemini', 'gemini-4-argon'),
            ])
            self.assertEqual(run.call_args.args[2], Path(evaluator.__file__).resolve().parent / 'api_results')

    def test_tuple_cli_ranges_and_legacy_range_formats(self):
        for node_values, degree_values, expected_nodes, expected_degree in (
            (['(5,8)'], ['(1,2)'], (5, 8), (1, 2)),
            (['(5, 8)'], ['(1, 2)'], (5, 8), (1, 2)),
            (['5', '8'], ['1', '2'], (5, 8), (1, 2)),
            (['5'], ['2'], 5, 2),
        ):
            argv = ['query_graph_world.py', '--nodes-per-subgraph', *node_values, '--out-degree', *degree_values]
            with self.subTest(argv=argv), patch('sys.argv', argv), patch.object(evaluator, 'load_env'), \
                    patch.object(evaluator, 'run_evaluation') as run:
                evaluator.main()
                self.assertEqual(run.call_args.kwargs['n_nodes_per_subgraph'], expected_nodes)
                self.assertEqual(run.call_args.kwargs['out_degree'], expected_degree)
                self.assertEqual(run.call_args.kwargs['openai_effort'], 'max')
                self.assertEqual(run.call_args.kwargs['anthropic_effort'], 'max')
                self.assertEqual(run.call_args.kwargs['gemini_effort'], 'high')
        for values in (['(8,5)'], ['(5,)'], ['(5,8,9)'], ['(True,8)'], ['(5.5,8)'], ['(-1,8)'], ['[5,8]']):
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluator.integer_or_range(values, '--nodes-per-subgraph')

    def test_n_paths_prompt_grammar_and_response_format(self):
        one = evaluator.make_prompt((0, 0), (1, 2), prompt_type="shortest_paths")
        self.assertIn('Find 1 shortest directed path from', one)
        self.assertNotIn('Find 1 shortest directed paths', one)
        self.assertIn('If several shortest paths exist, return any 1.', one)
        many = evaluator.make_prompt((0, 0), (1, 2), n_paths=2, prompt_type="shortest_paths")
        self.assertIn('Find 2 shortest directed paths from', many)
        self.assertIn('If several shortest paths exist, return any 2.', many)
        self.assertIn('exactly 2 distinct paths', many)
        paths = [[(0, 0), (1, 2)], [(0, 0), (0, 1), (1, 2)]]
        self.assertEqual(evaluator.parse_path(repr(paths), n_paths=2), paths)
        for invalid in (repr(paths[:1]), repr(paths[0]), '[]', '[[[0,0],[1,2]],[[0,0],[1,2]]]'):
            with self.assertRaises(ValueError):
                evaluator.parse_path(invalid, n_paths=2)

    def test_n_paths_cli_default_aliases_and_validation(self):
        for flag in ('--n_paths', '--n-paths'):
            with patch('sys.argv', ['query_graph_world.py', flag, '2']), patch.object(evaluator, 'load_env'), \
                    patch.object(evaluator, 'run_evaluation') as run:
                evaluator.main()
                self.assertEqual(run.call_args.kwargs['n_paths'], 2)
        with patch('sys.argv', ['query_graph_world.py']), patch.object(evaluator, 'load_env'), \
                patch.object(evaluator, 'run_evaluation') as run:
            evaluator.main()
            self.assertEqual(run.call_args.kwargs['n_paths'], 1)
        with patch('sys.argv', ['query_graph_world.py', '--n_paths', '0']), redirect_stderr(StringIO()), \
                patch.object(evaluator, 'run_evaluation') as run:
            with self.assertRaises(SystemExit):
                evaluator.main()
            run.assert_not_called()

    def test_multiple_paths_scoring_and_insufficient_ground_truth(self):
        generator = evaluator.GraphGenerator(1, 5, 2)
        source, a, b, target = (0, 0), (0, 1), (0, 2), (0, 3)
        generator.graph = nx.DiGraph([(source, a), (a, target), (source, b), (b, target), (a, b)])
        nx.set_node_attributes(generator.graph, 'none', name='restriction')
        generator.source, generator.target = source, target
        first, second = [source, a, target], [source, b, target]
        # Keep the cached oracle attributes consistent with this hand-built fixture graph.
        generator.adjacency_matrix = nx.to_numpy_array(generator.graph)
        generator.dict_of_lists = nx.to_dict_of_lists(generator.graph)
        generator.dict_of_dicts = nx.to_dict_of_dicts(generator.graph)
        generator.all_shortest_paths = {'paths': [first, second], 'costs': [{'distance': 2}] * 2}
        answers = {'correct': [second, first], 'duplicate': [first, first],
                   'longer': [first, [source, a, b, target]], 'too-few': [first]}
        def query(provider, model, key, png, prompt, *args, **kwargs):
            prompt_type = next(task for task, attributes in score_graph_world.PROMPT_ATTRIBUTES.items()
                               if "'pred_" + attributes[0] + "'" in prompt)
            if prompt_type == 'shortest_paths':
                self.assertIn('Find 2 shortest directed paths from', prompt)
                return repr({'pred_all_shortest_paths': answers[model]}), None
            return self.task_answer(generator, prompt_type), None
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret'}, clear=True), \
                patch.object(evaluator, 'GraphGenerator', return_value=generator), \
                patch.object(evaluator, 'query_model', side_effect=query), redirect_stdout(StringIO()):
            report = evaluator.run_evaluation([('openai', name) for name in answers], 1, directory, n_paths=2)
            scores = json.loads((Path(report['run_directory']) / 'scores.json').read_text())
            shortest = scores['shortest_paths']['scores']['1']
            self.assertEqual(shortest['openai/correct']['all_shortest_paths'], {'precision': 1, 'recall': 1, 'f1': 1})
            for name in ('duplicate', 'longer'):
                self.assertEqual(shortest['openai/' + name]['all_shortest_paths'],
                                 {'precision': 0.5, 'recall': 0.5, 'f1': 0.5})
            self.assertEqual(shortest['openai/too-few']['all_shortest_paths'],
                             {'precision': 1, 'recall': 0.5, 'f1': 2 / 3})
            with patch.object(evaluator, 'query_model') as query_mock:
                with self.assertRaisesRegex(ValueError, 'only 2 distinct shortest paths'), redirect_stderr(StringIO()):
                    evaluator.run_evaluation([('openai', 'correct')], 1, directory, n_paths=3)
                query_mock.assert_not_called()


    def test_incomplete_or_empty_provider_outputs_are_errors(self):
        for provider, fixture in (
            ('openai', {'status': 'incomplete', 'output': []}),
            ('openai', {'status': 'completed', 'output': []}),
            ('anthropic', {'stop_reason': 'max_tokens', 'content': []}),
            ('gemini', {'candidates': []}),
            ('gemini', {'candidates': [{'finishReason': 'MAX_TOKENS'}]}),
        ):
            with self.subTest(provider=provider), patch.object(evaluator, 'post_json', return_value=fixture):
                with self.assertRaises(RuntimeError):
                    evaluator.query_model(provider, 'test', 'fake-key', b'png', 'task')

    def test_run_directories_use_second_precision_and_avoid_collisions(self):
        fixed = evaluator.datetime(2026, 10, 23, 10, 11, 12, tzinfo=evaluator.ZoneInfo('America/New_York'))
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluator, 'datetime') as clock:
            clock.now.return_value = fixed
            first = evaluator.create_run_directory(directory)
            marker = first / 'preserved.txt'
            marker.write_text('preserve')
            second = evaluator.create_run_directory(directory)
            self.assertEqual(first.name, 'run_2026-10-23_10-11-12')
            self.assertEqual(second.name, 'run_2026-10-23_10-11-12_2')
            self.assertEqual(marker.read_text(), 'preserve')
            self.assertTrue(second.is_dir())

    def test_log_captures_stdout_stderr_and_failure_and_restores_streams(self):
        out, err = StringIO(), StringIO()
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'log.txt'
            with redirect_stdout(out), redirect_stderr(err):
                with self.assertRaisesRegex(RuntimeError, 'test failure'):
                    with evaluator.capture_log(log):
                        print('stdout line')
                        print('stderr line', file=sys.stderr)
                        raise RuntimeError('test failure')
                self.assertIs(sys.stdout, out)
                self.assertIs(sys.stderr, err)
            text = log.read_text()
            self.assertIn('stdout line', text)
            self.assertIn('stderr line', text)
            self.assertIn('RuntimeError: test failure', text)

    def test_prompt_types_use_templates_and_graph_attribute_names(self):
        # Every selector must format its matching template without import-time variables.
        expected = {'grap_rep': evaluator.PROMPT_GRAPH_REP,
                    'shortest_paths': evaluator.PROMPT_SHORTEST_PATHS,
                    'pareto_frontier': evaluator.PROMPT_PARETO_FRONTIER,
                    'scalarized': evaluator.PROMPT_SCALARIZED}
        for task, template in expected.items():
            prompt = evaluator.make_prompt((1, 0), (2, 1), n_paths=2, prompt_type=task,
                                            node_order=[(1, 0), (2, 1)], edge_cost_names=['time'], cost_precision=3)
            self.assertEqual(prompt, template.format(source=(1, 0), target=(2, 1), n_paths=2,
                             path_word='paths', node_order=[(1, 0), (2, 1)], edge_cost_names=['time'], cost_precision=3))
            for attribute in score_graph_world.PROMPT_ATTRIBUTES[task]:
                self.assertIn("'pred_" + attribute + "'", prompt)
        with self.assertRaisesRegex(ValueError, 'prompt_type'):
            evaluator.make_prompt((1, 0), (2, 1), prompt_type='unknown')
        for invalid in (0, -1, True, 1.5):
            with self.subTest(count=invalid), self.assertRaises(ValueError):
                evaluator.make_prompt((1, 0), (2, 1), n_paths=invalid)

    def test_task_prediction_parser_preserves_each_field(self):
        # JSON and Python tuple encodings must yield the same normalized prediction.
        answer = {'pred_all_shortest_paths': [[(1, 0), (2, 1)]]}
        parsed = evaluator.parse_prompt_prediction(repr(answer), 'shortest_paths')
        self.assertEqual(parsed, evaluator.parse_prompt_prediction(json.dumps(answer), 'shortest_paths'))
        self.assertEqual(parsed, {'pred_all_shortest_paths': [[[1, 0], [2, 1]]]})
        for invalid in ('{}', '[]', '```python\n{}\n```', repr({'pred_shortest_paths': []})):
            with self.subTest(answer=invalid), self.assertRaises(ValueError):
                evaluator.parse_prompt_prediction(invalid, 'shortest_paths')
        raw = {'pred_adjacency_matrix': [[0, 0], [1, 0]],
               'pred_dict_of_lists': {(1, 0): [(2, 1)], (2, 1): []},
               'pred_dict_of_dicts': None}
        parsed = evaluator.parse_prompt_prediction(repr(raw), 'grap_rep')
        self.assertEqual(parsed['pred_adjacency_matrix'], raw['pred_adjacency_matrix'])
        self.assertIsNone(parsed['pred_dict_of_dicts'])
        self.assertEqual(parsed['pred_dict_of_lists'], {'(1, 0)': [[2, 1]], '(2, 1)': []})

    def test_exact_path_list_precision_recall_duplicates_and_invalid_entries(self):
        # Match complete paths; duplicates and malformed entries cannot inflate recall.
        a, b, c, d = (1, 0), (1, 1), (1, 2), (1, 3)
        first, second = [a, b, d], [a, c, d]
        truth = {'paths': [first, second], 'costs': [{'distance': 2}] * 2}
        self.assertEqual(score_graph_world.score_path_lists(truth, [first, first, [a, d]]),
                         {'precision': 1 / 3, 'recall': 0.5, 'f1': 0.4})
        self.assertEqual(score_graph_world.score_path_lists(truth, [first, None]),
                         {'precision': 0.5, 'recall': 0.5, 'f1': 0.5})
        self.assertEqual(score_graph_world.score_path_lists(truth, {'paths': [second, first], 'costs': []}),
                         {'precision': 1, 'recall': 1, 'f1': 1})
        for prediction in (None, [], [first[:-1]], [['invalid']], 'invalid'):
            with self.subTest(prediction=prediction):
                self.assertEqual(score_graph_world.score_path_lists(truth, prediction),
                                 {'precision': 0, 'recall': 0, 'f1': 0})
        self.assertEqual(score_graph_world.score_path_lists([], []), {'precision': 1, 'recall': 1, 'f1': 1})
        self.assertEqual(score_graph_world.score_path_lists([], None), {'precision': 0, 'recall': 0, 'f1': 0})

    def test_task_metrics_compare_full_dictionaries_and_matrix_values(self):
        # All ground-truth edge attributes must participate in dictionary correctness.
        graph = evaluator.GraphGenerator(1, 5, 2)
        truth = evaluator.graph_ground_truth(graph)
        prediction = {}
        for task in evaluator.PROMPTS:
            prediction.update(evaluator.parse_prompt_prediction(self.task_answer(graph, task), task))
        scores = score_graph_world.score_prompt_responses(truth, prediction)
        self.assertEqual(set(scores), set(evaluator.PROMPTS))
        for attribute in ('dict_of_lists', 'dict_of_dicts'):
            self.assertEqual(scores['grap_rep']['scores'][attribute], {'correct': True})
        parent = next(node for node, children in prediction['pred_dict_of_dicts'].items() if children)
        child = next(iter(prediction['pred_dict_of_dicts'][parent]))
        prediction['pred_dict_of_dicts'][parent][child]['water'] += 1
        # A weight-only matrix error fails correctness but preserves the edge structure.
        matrix = prediction['pred_adjacency_matrix']
        row, column = next((i, j) for i, values in enumerate(matrix) for j, value in enumerate(values) if value)
        matrix[row][column] = 2
        scores = score_graph_world.score_prompt_responses(truth, prediction)['grap_rep']['scores']
        self.assertEqual(scores['dict_of_dicts'], {'correct': False})
        self.assertEqual(scores['dict_of_lists'], {'correct': True})
        self.assertEqual(scores['adjacency_matrix'],
                         {'correct': False, 'structural_hamming_distance': 0, 'graph_edit_distance': 0})
        # Literal child-list order is significant, while dictionary key order is not.
        a, b, c = (1, 0), (1, 1), (1, 2)
        self.assertEqual(score_graph_world.score_dictionary({a: [b, c]}, {a: [c, b]}), {'correct': False})
        self.assertEqual(score_graph_world.score_dictionary({a: [b], b: []}, {b: [], a: [b]}), {'correct': True})

    def test_matrix_scores_normalize_predictions_to_ground_truth_dtype(self):
        # Include float32 precision and large integers that forcing float64 would lose.
        cases = [
            (np.array([[0, 1], [0, 0]], dtype=np.float64), np.array([[0, 1], [0, 0]], dtype=np.int64), True),
            (np.array([[0, 1], [0, 0]], dtype=np.int64), np.array([[0, 1], [0, 0]], dtype=np.float64), True),
            (np.array([[0, 0.1], [0, 0]], dtype=np.float32), np.array([[0, 0.1], [0, 0]], dtype=np.float64), True),
            (np.array([[0, 1], [0, 0]], dtype=np.int32), np.array([[0, 1.9], [0, 0]], dtype=np.float64), True),
            (np.array([[0, 1], [0, 0]], dtype=np.int32), np.array([[0, 2], [0, 0]], dtype=np.float64), False),
            (np.array([[0, 0.5], [0, 0]], dtype=np.float64), np.array([[0, 1], [0, 0]], dtype=np.int64), False),
            (np.array([[0, 2 ** 53 + 1], [0, 0]], dtype=np.int64),
             np.array([[0, 2 ** 53], [0, 0]], dtype=np.int64), False),
        ]
        for truth, prediction, expected in cases:
            with self.subTest(truth=truth, prediction=prediction):
                original_truth, original_prediction = truth.copy(), prediction.copy()
                score = score_graph_world.score_adjacency_matrix(truth, prediction)
                self.assertIs(score['correct'], expected)
                record = {'node_order': [(1, 0), (1, 1)], 'adjacency_matrix': truth, 'shortest_paths': []}
                legacy = score_graph_world.score_prediction(record, {'pred_adjacency_matrix': prediction})
                self.assertEqual(legacy['pred_adjacency_matrix']['exact_match'], int(expected))
                if expected:
                    self.assertEqual(score['structural_hamming_distance'], 0)
                    self.assertEqual(score['graph_edit_distance'], 0)
                np.testing.assert_array_equal(truth, original_truth)
                np.testing.assert_array_equal(prediction, original_prediction)

    def test_matrix_dtype_conversion_rejects_nonfinite_or_overflowing_values(self):
        # Reject invalid float inputs before a cast could conceal them as integer sentinels.
        for dtype, value in ((np.int64, float('nan')), (np.int64, float('inf')),
                             (np.int64, 1e100), (np.float32, 1e100)):
            with self.subTest(dtype=dtype, value=value):
                truth = np.array([[0, 1], [0, 0]], dtype=dtype)
                prediction = [[0, value], [0, 0]]
                self.assertEqual(score_graph_world.score_adjacency_matrix(truth, prediction),
                                 {'correct': False, 'structural_hamming_distance': None, 'graph_edit_distance': None})

    def test_task_matrix_scores_invalid_shapes_and_additional_isolates(self):
        truth = [[0, 1], [0, 0]]
        extra = [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
        self.assertEqual(score_graph_world.score_adjacency_matrix(truth, extra),
                         {'correct': False, 'structural_hamming_distance': 0, 'graph_edit_distance': 1})
        for invalid in (None, [], [[0, 1]], [[0, 1], [0]], [[0, float('nan')], [0, 0]]):
            with self.subTest(prediction=invalid):
                self.assertEqual(score_graph_world.score_adjacency_matrix(truth, invalid),
                                 {'correct': False, 'structural_hamming_distance': None, 'graph_edit_distance': None})

    def test_multiturn_provider_payloads_keep_image_before_all_questions(self):
        # Inspect real payload construction without sending any network requests.
        fixture = {'status': 'completed', 'output': [{'content': [{'type': 'output_text', 'text': 'answer'}]}],
                   'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': 'answer'}],
                   'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': 'answer'}]}}]}
        history = [{'prompt': 'graph question', 'response': 'graph answer'},
                   {'prompt': 'shortest question', 'response': 'shortest answer'},
                   {'prompt': 'pareto question', 'response': 'pareto answer'}]
        png = b'graph-image-for-history'
        encoded = base64.b64encode(png).decode()
        for provider in evaluator.KEY_NAMES:
            with self.subTest(provider=provider), patch.object(evaluator, 'post_json', return_value=fixture) as post:
                evaluator.query_model(provider, 'test', 'test-key', png, 'scalar question', history=history)
                payload = post.call_args.args[2]
                messages = payload[{'openai': 'input', 'anthropic': 'messages', 'gemini': 'contents'}[provider]]
                assistant = 'model' if provider == 'gemini' else 'assistant'
                self.assertEqual([message['role'] for message in messages], ['user', assistant] * 3 + ['user'])
                self.assertEqual(json.dumps(messages).count(encoded), 1)
                self.assertIn('scalar question', json.dumps(messages[-1]))
                self.assertIn('pareto answer', json.dumps(messages[-2]))
                first = messages[0]['parts' if provider == 'gemini' else 'content']
                self.assertIn(encoded, json.dumps(first[0]))
                self.assertIn('graph question', json.dumps(first[1]))
        self.assertEqual(len(history), 3)

    def test_response_filenames_are_safe_and_collision_free(self):
        models = [('openai', 'model'), ('anthropic', 'model'), ('openai', 'a/b'), ('openai', 'a_b')]
        names = evaluator.response_filenames(models)
        self.assertEqual(len(set(names.values())), len(models))
        self.assertTrue(all(Path(name).name == name and name.endswith('_responses.json')
                            for name in names.values()))
        self.assertEqual(evaluator.response_filenames([('openai', 'test-model')])[('openai', 'test-model')],
                         'test-model_responses.json')

    def test_prompt_failure_does_not_discard_other_task_answers(self):
        graph = evaluator.GraphGenerator(1, 5, 2)
        def query(provider, model, key, png, prompt, *args, history=None, **kwargs):
            task = next(task for task, attributes in score_graph_world.PROMPT_ATTRIBUTES.items()
                        if "'pred_" + attributes[0] + "'" in prompt)
            if task == 'shortest_paths':
                raise RuntimeError('failed shortest request')
            return self.task_answer(graph, task), None
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret'}, clear=True), \
                patch.object(evaluator, 'GraphGenerator', return_value=graph), \
                patch.object(evaluator, 'query_model', side_effect=query) as calls, redirect_stdout(StringIO()):
            report = evaluator.run_evaluation([('openai', 'test')], 1, directory)
            scores = json.loads((Path(report['run_directory']) / 'scores.json').read_text())
            self.assertEqual(calls.call_count, 4)
            self.assertEqual([len(call.kwargs['history']) for call in calls.call_args_list], [0, 1, 1, 2])
            self.assertTrue(scores['grap_rep']['scores']['1']['openai/test']['adjacency_matrix']['correct'])
            self.assertEqual(scores['shortest_paths']['scores']['1']['openai/test']['all_shortest_paths']['f1'], 0)
            self.assertEqual(scores['pareto_frontier']['scores']['1']['openai/test']['multiobjective_shortest_paths']['f1'], 1)
            self.assertEqual(scores['scalarized']['scores']['1']['openai/test']['scalar_dijkstra']['f1'], 1)

    def test_average_prompt_scores_include_distances_counts_and_macro_f1(self):
        # Use distinct denominators to verify missing distances are never counted as zero.
        tasks = {
            'grap_rep': {'scores': {
                '1': {'openai/test': {'adjacency_matrix': {'correct': True, 'structural_hamming_distance': 2,
                                                         'graph_edit_distance': 1},
                                      'dict_of_lists': {'correct': True}},
                      'openai/missing': {'adjacency_matrix': {'correct': False, 'structural_hamming_distance': None,
                                                            'graph_edit_distance': None}}},
                '2': {'openai/test': {'adjacency_matrix': {'correct': False, 'structural_hamming_distance': 6,
                                                         'graph_edit_distance': None},
                                      'dict_of_lists': {'correct': False}},
                      'openai/missing': {'adjacency_matrix': {'correct': False, 'structural_hamming_distance': None,
                                                            'graph_edit_distance': None}}}}},
            'shortest_paths': {'scores': {
                '1': {'openai/test': {'all_shortest_paths': {'precision': 1, 'recall': 0.5, 'f1': 2 / 3}}},
                '2': {'openai/test': {'all_shortest_paths': {'precision': 0.5, 'recall': 1, 'f1': 2 / 3}}}}},
        }
        summary = score_graph_world.summarize_prompt_scores(tasks)
        self.assertEqual(summary['openai/test']['pred_adjacency_matrix'],
                         {'attempted': 2, 'accuracy': 0.5, 'mean_structural_hamming_distance': 4,
                          'mean_graph_edit_distance': 1, 'shd_count': 2, 'ged_count': 1})
        self.assertEqual(summary['openai/test']['pred_dict_of_lists'], {'attempted': 2, 'accuracy': 0.5})
        self.assertEqual(summary['openai/test']['pred_all_shortest_paths'],
                         {'attempted': 2, 'precision': 0.75, 'recall': 0.75, 'f1': 2 / 3})
        self.assertEqual(summary['openai/missing']['pred_adjacency_matrix'],
                         {'attempted': 2, 'accuracy': 0, 'mean_structural_hamming_distance': None,
                          'mean_graph_edit_distance': None, 'shd_count': 0, 'ged_count': 0})
        # The same pretty-printed table must reach console and log through capture_log.
        model_summaries = {name: {**score_graph_world.create_model_summary(None),
                                  'attempted': 2, 'scores': metrics} for name, metrics in summary.items()}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as output:
            logfile = Path(directory) / 'log.txt'
            with evaluator.capture_log(logfile):
                evaluator.print_summary(model_summaries)
            self.assertEqual(logfile.read_text(), output.getvalue())
            for label in ('Average scores across replicates', 'mean SHD',
                          'mean GED', 'precision', 'recall', 'f1', 'N/A', '0.666667'):
                self.assertIn(label, output.getvalue())

    def test_mock_run_uses_no_credentials_or_network_and_exports_all_artifacts(self):
        # Every entry point to provider/network code must remain unused in mock mode.
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch.object(evaluator, 'api_key', side_effect=AssertionError('Credential lookup')) as credentials, \
                patch.object(evaluator, 'query_model', side_effect=AssertionError('Model API')) as query, \
                patch.object(evaluator, 'post_json', side_effect=AssertionError('HTTP request')) as post, \
                patch.object(evaluator.urllib.request, 'urlopen', side_effect=AssertionError('Network')) as network, \
                redirect_stdout(StringIO()) as output:
            report = evaluator.run_evaluation(evaluator.DEFAULT_MODELS, 2, directory,
                                              n_subgraphs=1, n_nodes_per_subgraph=3, out_degree=2, mock=True)
            for forbidden in (credentials, query, post, network):
                forbidden.assert_not_called()
            self.assertTrue(report['config']['mock'])
            run_dir = Path(report['run_directory'])
            self.assertEqual(len(list((run_dir / 'images_seen_by_model').glob('*.png'))), 2)
            scores = json.loads((run_dir / 'scores.json').read_text())
            for number in ('1', '2'):
                for provider, model in evaluator.DEFAULT_MODELS:
                    name = provider + '/' + model
                    self.assertTrue(scores['grap_rep']['scores'][number][name]['adjacency_matrix']['correct'])
                    for task, attribute in (('shortest_paths', 'all_shortest_paths'),
                                             ('pareto_frontier', 'multiobjective_shortest_paths'),
                                             ('scalarized', 'scalar_dijkstra')):
                        self.assertEqual(scores[task]['scores'][number][name][attribute]['f1'], 1)
                    responses = json.loads((run_dir / report['response_files'][name]).read_text())
                    self.assertEqual(list(responses[number]), list(evaluator.PROMPTS))
                    self.assertTrue(all(entry['mock'] and entry['usage']['mock']
                                        for entry in responses[number].values()))
                    self.assertTrue((run_dir / report['prompt_files'][name]).is_file())
            self.assertTrue(all(response['mock'] for replicate in report['replicates']
                                for response in replicate['responses']))
            self.assertIn('MOCK RUN: synthetic oracle-based responses', output.getvalue())
            self.assertEqual((run_dir / 'log.txt').read_text(), output.getvalue())
            evaluator.export_saved_run(run_dir)
            self.assertEqual(json.loads((run_dir / 'scores.json').read_text()), scores)

    def test_mock_cli_skips_env_loading_and_forwards_flag(self):
        # Both public flag spellings must select the safe, credential-free execution path.
        for flag in ('--mock', '--mock-run'):
            with self.subTest(flag=flag), patch('sys.argv', ['query_graph_world.py', flag, '--n-replicates', '1']), \
                    patch.object(evaluator, 'load_env', side_effect=AssertionError('Read credentials')) as load, \
                    patch.object(evaluator, 'run_evaluation') as run:
                evaluator.main()
                load.assert_not_called()
                self.assertTrue(run.call_args.kwargs['mock'])
                self.assertEqual(run.call_args.args[1], 1)

    def test_mock_response_respects_requested_path_count(self):
        # Synthetic shortest responses obey n_paths without modifying the cached oracle.
        first = [(1, 0), (1, 1), (1, 3)]
        second = [(1, 0), (1, 2), (1, 3)]
        truth = {'all_shortest_paths': {'paths': [first, second], 'costs': [{'distance': 2}] * 2}}
        raw, usage = evaluator.mock_response(truth, 'shortest_paths', n_paths=2)
        parsed = evaluator.parse_prompt_prediction(raw, 'shortest_paths')
        self.assertEqual(parsed['pred_all_shortest_paths'], [[list(node) for node in path] for path in (first, second)])
        self.assertEqual(usage, {'mock': True})
        self.assertEqual(truth['all_shortest_paths']['paths'], [first, second])

    def test_end_to_end_scoring_and_saved_artifacts(self):
        real_generator = evaluator.GraphGenerator
        generated = []
        def create_graph(**kwargs):
            graph = real_generator(**kwargs)
            generated.append(graph)
            print('Created replicate graph')
            return graph
        def fake_query(provider, model, key, png, prompt, timeout, max_tokens, *, effort=None, history=None):
            self.assertTrue(png.startswith(b'\x89PNG\r\n\x1a\n'))
            graph = generated[-1]
            task = next(task for task, attributes in score_graph_world.PROMPT_ATTRIBUTES.items()
                        if "'pred_" + attributes[0] + "'" in prompt)
            self.assertEqual(prompt, evaluator.make_prompt(graph.source, graph.target, prompt_type=task,
                              node_order=list(graph.graph), edge_cost_names=graph.edge_cost_names,
                              cost_precision=graph.cost_precision))
            if model != 'error':
                self.assertEqual(len(history), list(evaluator.PROMPTS).index(task))
            else:
                self.assertEqual(history, [])
            if model == 'correct':
                return self.task_answer(graph, task), {'tokens': 1}
            if model == 'wrong':
                answer = ast.literal_eval(self.task_answer(graph, task))
                for field in answer:
                    answer[field] = ([[0] * len(graph.graph)] * len(graph.graph) if field == 'pred_adjacency_matrix'
                                     else {} if field.startswith('pred_dict_') else [[]])
                return repr(answer), None
            if model == 'format':
                return 'Here is the path: []', None
            raise RuntimeError(f'Provider failure with {key}')
        models = [('openai', 'correct'), ('anthropic', 'wrong'), ('gemini', 'format'), ('openai', 'error')]
        keys = {'OPENAI_API_KEY': 'test-openai-secret', 'ANTHROPIC_API_KEY': 'test-anthropic-secret',
                'GEMINI_API_KEY': 'test-gemini-secret'}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, keys, clear=True), \
                patch.object(evaluator, 'GraphGenerator', side_effect=create_graph), \
                patch.object(evaluator, 'query_model', side_effect=fake_query) as query, \
                patch.object(real_generator, 'plot_graph', autospec=True, side_effect=real_generator.plot_graph) as render, \
                redirect_stdout(StringIO()) as console:
            report = evaluator.run_evaluation(models, 2, directory, n_subgraphs=1,
                                              openai_effort='high', anthropic_effort='medium', gemini_effort='low')
            self.assertEqual(query.call_count, 32)
            self.assertEqual(len(generated), 2)
            self.assertEqual(render.call_count, 2)
            self.assertEqual([call.kwargs['effort'] for call in query.call_args_list],
                             ['high'] * 4 + ['medium'] * 4 + ['low'] * 4 + ['high'] * 4
                             + ['high'] * 4 + ['medium'] * 4 + ['low'] * 4 + ['high'] * 4)
            self.assertEqual(report['summary']['openai/correct']['accuracy'], 1)
            self.assertEqual(report['summary']['gemini/format']['format_errors'], 2)
            self.assertEqual(report['summary']['openai/error']['api_errors'], 2)
            run_dir = Path(report['run_directory'])
            self.assertEqual(run_dir.parent, Path(directory))
            scores = json.loads((run_dir / 'scores.json').read_text())
            exported = json.loads((run_dir / 'graph_replicates.json').read_text())
            self.assertEqual(set(scores), set(evaluator.PROMPTS))
            self.assertEqual(scores, score_graph_world.score_replicates(exported))
            for task in scores.values():
                self.assertEqual(set(task), {'true', 'pred', 'scores'})
                self.assertEqual(set(task['true']), {'1', '2'})
            for index, graph in enumerate(generated, 1):
                number = str(index)
                truth = evaluator.graph_ground_truth(graph)
                self.assertEqual(scores['grap_rep']['true'][number]['dict_of_dicts'], truth['dict_of_dicts'])
                matrix = scores['grap_rep']['scores'][number]['openai/correct']['adjacency_matrix']
                self.assertEqual(matrix, {'correct': True, 'structural_hamming_distance': 0, 'graph_edit_distance': 0})
                for attribute in ('dict_of_lists', 'dict_of_dicts'):
                    self.assertEqual(scores['grap_rep']['scores'][number]['openai/correct'][attribute], {'correct': True})
                for task, attribute in (('pareto_frontier', 'multiobjective_shortest_paths'), ('scalarized', 'scalar_dijkstra')):
                    self.assertEqual(scores[task]['scores'][number]['openai/correct'][attribute],
                                     {'precision': 1, 'recall': 1, 'f1': 1})
                self.assertEqual(scores['shortest_paths']['scores'][number]['openai/correct']['all_shortest_paths']['recall'],
                                 1 / len(graph.all_shortest_paths['paths']))
                # Every model's raw answers and parsed predictions survive in one response file.
                for provider, model in models:
                    transcript_path = run_dir / report['prompt_files'][provider + '/' + model]
                    self.assertEqual(transcript_path.name, model + '_prompts.txt')
                    transcript = transcript_path.read_text()
                    self.assertIn('Model: ' + provider + '/' + model, transcript)
                    section = transcript.split('Replicate ' + number + '\n', 1)[1].split('\nReplicate ', 1)[0]
                    replicate = report['replicates'][index - 1]
                    self.assertIn('Image: ' + replicate['image'], section)
                    self.assertIn('Image SHA-256: ' + replicate['image_sha256'], section)
                    self.assertEqual(section.count('User prompt ('), 4)
                    for task, prompt in replicate['prompts'].items():
                        self.assertIn(f'User prompt ({task}):\n{prompt}\n', section)
                    saved = json.loads((run_dir / report['response_files'][provider + '/' + model]).read_text())
                    self.assertEqual(list(saved[number]), list(evaluator.PROMPTS))
                    for task, entry in saved[number].items():
                        self.assertEqual(entry['image_sha256'], report['replicates'][index - 1]['image_sha256'])
                        if model == 'correct':
                            self.assertIsNotNone(entry['prediction'])
                            self.assertIsNone(entry['error'])
                        elif model == 'error':
                            self.assertEqual(entry['error_type'], 'api')
                            self.assertIn('[REDACTED]', entry['error'])
                # Check prompt order, identical image bytes, and fresh histories for each conversation.
                calls = query.call_args_list[(index - 1) * 16:index * 16]
                png = (run_dir / report['replicates'][index - 1]['image']).read_bytes()
                self.assertTrue(all(call.args[3] == png for call in calls))
                for offset in range(0, 16, 4):
                    self.assertEqual([call.args[4] for call in calls[offset:offset + 4]],
                                     list(report['replicates'][index - 1]['prompts'].values()))
                    self.assertEqual(calls[offset].kwargs['history'], [])
            # Re-export and rescoring must reproduce complete named-cost ground truth.
            evaluator.export_saved_run(run_dir)
            self.assertEqual(json.loads((run_dir / 'graph_replicates.json').read_text()), exported)
            score_graph_world.score_saved_run(run_dir)
            self.assertEqual(json.loads((run_dir / 'scores.json').read_text()), scores)
            self.assertEqual(len(list((run_dir / 'images_seen_by_model').glob('*.png'))), 2)
            logged = (run_dir / 'log.txt').read_text()
            self.assertEqual(logged, console.getvalue())
            self.assertIn('Created replicate graph', logged)
            self.assertIn('Average scores across replicates', logged)
            for metric in ('mean SHD', 'mean GED', 'precision', 'recall', 'f1'):
                self.assertIn(metric, logged)
            self.assertIn('Here is the path: []', logged)
            for key in keys.values():
                self.assertNotIn(key, (run_dir / 'run_metadata.json').read_text())
                self.assertNotIn(key, logged)



if __name__ == '__main__':
    unittest.main()
