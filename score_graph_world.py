"""Scoring, aggregation, and score-file workflows for graph-world evaluation."""

import ast
from collections import Counter
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

GRAPH_FIELDS = ("pred_adjacency_matrix", "pred_dict_of_dicts", "pred_dict_of_lists")


def pass_at_k(n, c, k, stable: bool = False):
    '''
    Unbiased probability of at least one success among k trials.
    n: total samples generated
    c: number of correct samples
    k: total trials
    '''
    validate_pass_counts(n, c, k)
    if k == 1:
        return c / n
    if stable:
        return 1.0 - np.prod(1 - k / np.arange(n - c + 1, n + 1))
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_power_k(n, c, k):
    '''
    Unbiased probability that all k trials succeed (pass^k).
    n: total samples generated
    c: number of correct samples
    k: total trials
    '''
    validate_pass_counts(n, c, k)
    return (c / n) ** k


def validate_pass_counts(n, c, k):
    '''
    Reject undefined estimators rather than silently extrapolating trials.
    '''
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in (n, c, k))
            or not 0 <= c <= n or not 1 <= k <= n):
        raise ValueError("pass metrics require integer counts with 0 <= c <= n and 1 <= k <= n")


def structural_hamming_distance(target, prediction):
    """Count differing directed edge positions; a reversal counts twice.

    Nonzero weights indicate edges. Weight-only errors affect exact matching,
    but not SHD. Matrices must already use the same node order and shape.
    """
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.ndim != 2 or target.shape[0] != target.shape[1] or target.shape != prediction.shape:
        raise ValueError("SHD requires square matrices with the same shape")
    return int(np.count_nonzero((target != 0) != (prediction != 0)))


def graph_edit_distance(target, prediction, **kwargs):
    """Wrap nx.graph_edit_distance for directed adjacency matrices.

    Matrices may have different node counts. NetworkX defaults allow node
    relabeling and ignore weights; node/edge insertion and deletion cost 1.
    Forward optional NetworkX arguments, including costs, upper_bound, and
    timeout. Without timeout the result is exact; with timeout NetworkX may
    return its best upper bound. None means no distance was returned.
    """
    graphs = []
    for value in (target, prediction):
        matrix = np.asarray(value, dtype=float)
        if matrix.size == 0 and matrix.shape == (0,):
            matrix = matrix.reshape(0, 0)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("GED requires finite square adjacency matrices")
        graphs.append(nx.from_numpy_array(matrix, create_using=nx.DiGraph))
    distance = nx.graph_edit_distance(*graphs, **kwargs)
    return None if distance is None else float(distance)


def score_shortest_paths(ground_truth, predicted, n_paths=1):
    """Score distinct requested paths and exact coverage of all oracle paths."""
    oracle = {tuple(tuple(node) for node in path) for path in ground_truth}
    if predicted is None:
        return {"correct": 0, "exact_match": 0, "path_correctness": None}
    try:
        if not isinstance(predicted, list):
            raise ValueError("Paths must be a list")
        paths = []
        for path in predicted:
            if not isinstance(path, list) or any(
                not isinstance(node, (tuple, list)) or len(node) != 2
                or any(type(value) is not int for value in node) for node in path
            ):
                raise ValueError("Paths must contain integer pairs")
            paths.append(tuple(tuple(node) for node in path))
    except (TypeError, ValueError):
        return {"correct": 0, "exact_match": 0, "path_correctness": None}
    correctness = [path in oracle for path in paths]
    distinct = len(set(paths)) == len(paths)
    return {
        "correct": int(len(paths) == n_paths and distinct and all(correctness)),
        "exact_match": int(distinct and set(paths) == oracle),
        "path_correctness": correctness,
    }


def _node(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(type(x) is not int for x in value):
        raise ValueError("Node names must be integer pairs")
    return tuple(value)


def _numeric_adjacency_matrix(value, dtype=None):
    """Validate numeric adjacency values, optionally casting to the oracle dtype."""
    matrix = np.asarray(value)
    # Validate before conversion so nonfinite floats cannot become finite integer sentinels.
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.dtype.kind not in "biuf"
            or not np.isfinite(matrix).all()):
        raise ValueError("Adjacency matrices must be finite numeric square arrays")
    if dtype is not None:
        # Standardize predicted values to the stored ground-truth datatype before scoring.
        with np.errstate(over="raise", invalid="raise"):
            matrix = matrix.astype(dtype, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError("Converted adjacency values must be finite")
    return matrix


def _matrix(value, nodes, *, dtype=None):
    matrix = np.asarray(value)
    if not nodes and matrix.size == 0:
        matrix = matrix.reshape(0, 0)
    matrix = _numeric_adjacency_matrix(matrix, dtype)
    if matrix.shape != (len(nodes), len(nodes)):
        raise ValueError("Matrix must match its node order")
    return matrix


def _dictionary_graph(value, weighted):
    """Use NetworkX conversions, retaining scalar dict-of-dicts weights."""
    if not isinstance(value, dict):
        raise ValueError("Graph prediction must be a dictionary")
    if weighted:
        adjacency = {}
        for parent, children in value.items():
            if not isinstance(children, dict):
                raise ValueError("Weighted children must be dictionaries")
            adjacency[_node(parent)] = {}
            for child, weight in children.items():
                if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight):
                    raise ValueError("Edge weights must be finite numbers")
                adjacency[_node(parent)][_node(child)] = {"weight": weight}
        return nx.from_dict_of_dicts(adjacency, create_using=nx.DiGraph)
    adjacency = {}
    for parent, children in value.items():
        if not isinstance(children, list):
            raise ValueError("Children must be lists")
        adjacency[_node(parent)] = [_node(child) for child in children]
    return nx.from_dict_of_lists(adjacency, create_using=nx.DiGraph)


def score_prediction(ground_truth, prediction, n_paths=1):
    """Score each exported representation independently, aligned by node label.

    Accept pred_adjacency_array as the legacy spelling of pred_adjacency_matrix.
    Missing/invalid predictions get exact_match=0 and null distances. Extra nodes are
    retained on the union of node sets for SHD, and fail exact matrix matching.
    Dictionary equality ignores key insertion order but preserves list order.
    """
    truth_nodes = [_node(node) for node in ground_truth["node_order"]]
    truth_matrix = _matrix(ground_truth["adjacency_matrix"], truth_nodes)
    scores = {}
    for field in GRAPH_FIELDS:
        value = prediction.get(field)
        if field == "pred_adjacency_matrix" and field not in prediction:
            value = prediction.get("pred_adjacency_array")
        score = {"exact_match": 0, "structural_hamming_distance": None, "graph_edit_distance": None}
        scores[field] = score
        if value is None:
            continue
        try:
            if field == "pred_adjacency_matrix":
                raw_order = prediction.get("pred_node_order")
                pred_nodes = truth_nodes if raw_order is None else [_node(node) for node in raw_order]
                if len(set(pred_nodes)) != len(pred_nodes):
                    raise ValueError("Predicted node order contains duplicates")
                matrix = _matrix(value, pred_nodes, dtype=truth_matrix.dtype)
            else:
                graph = _dictionary_graph(value, weighted=field == "pred_dict_of_dicts")
                pred_nodes = list(graph)
                matrix = _numeric_adjacency_matrix(
                    nx.to_numpy_array(graph, nodelist=pred_nodes), dtype=truth_matrix.dtype)
                # JSON normalizes tuple coordinates to lists for live and saved runs.
                score["exact_match"] = int(json.loads(json.dumps(value)) == json.loads(
                    json.dumps(ground_truth[field.removeprefix("pred_")])
                ))
            nodes = truth_nodes + [node for node in pred_nodes if node not in truth_nodes]
            indices = {node: index for index, node in enumerate(nodes)}
            target = np.zeros((len(nodes), len(nodes)), dtype=truth_matrix.dtype)
            target[:len(truth_nodes), :len(truth_nodes)] = truth_matrix
            aligned = np.zeros_like(target)
            positions = [indices[node] for node in pred_nodes]
            aligned[np.ix_(positions, positions)] = matrix
            score["structural_hamming_distance"] = structural_hamming_distance(target, aligned)
            # GED uses the original matrices, preserving missing/extra nodes
            # rather than padding them with isolates as required for SHD.
            score["graph_edit_distance"] = graph_edit_distance(truth_matrix, matrix)
            if field == "pred_adjacency_matrix":
                score["exact_match"] = int(set(pred_nodes) == set(truth_nodes) and np.array_equal(target, aligned))
        except (ValueError, TypeError, SyntaxError, OverflowError, FloatingPointError, nx.NetworkXException) as exc:
            score.update(exact_match=0, error=str(exc))
    scores["pred_shortest_paths"] = score_shortest_paths(
        ground_truth["shortest_paths"], prediction.get("pred_shortest_paths"), n_paths,
    )
    return scores



# Keep task names and graph-attribute names shared by querying and scoring.
PROMPT_ATTRIBUTES = {
    "graph_rep": ("adjacency_matrix", "dict_of_lists", "dict_of_dicts"),
    "shortest_paths": ("all_shortest_paths",),
    "pareto_frontier": ("multiobjective_shortest_paths",),
    "scalarized": ("scalar_dijkstra",),
}


def json_compatible(value):
    """Encode arrays and tuple node names without changing graph representations."""
    # Convert NumPy values and recursively encode tuple keys as Python literals.
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {repr(key) if isinstance(key, tuple) else str(key): json_compatible(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def score_path_lists(ground_truth, predicted):
    """Score exact complete paths, counting duplicate predictions as false positives.

    Ground truth may be a graph attribute with a 'paths' entry or a path list.
    Costs are retained in exports but do not participate in path matching.
    Two empty valid lists score 1; missing or malformed predictions score 0.
    """
    # Use full paths as multiset members; a correct prefix is not a correct path.
    truth = ground_truth.get("paths") if isinstance(ground_truth, dict) else ground_truth
    prediction = predicted.get("paths") if isinstance(predicted, dict) else predicted
    if not isinstance(truth, list):
        raise ValueError("Ground truth paths must be a list")
    if not isinstance(prediction, list):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def path_key(path):
        # Require the same integer-pair node labels used by the graph generator.
        if not isinstance(path, (tuple, list)) or not path:
            return None
        try:
            return tuple(_node(node) for node in path)
        except (TypeError, ValueError, SyntaxError):
            return None

    true_paths = Counter(path_key(path) for path in truth)
    if None in true_paths:
        raise ValueError("Ground truth contains an invalid path")
    predicted_paths = Counter(path_key(path) for path in prediction)
    # Invalid path entries remain in the precision denominator, never the numerator.
    predicted_paths.pop(None, None)
    matched = sum((true_paths & predicted_paths).values())
    precision = matched / len(prediction) if prediction else float(not truth)
    recall = matched / len(truth) if truth else float(not prediction)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_adjacency_matrix(ground_truth, predicted):
    """Compare values in the requested node order after casting to ground-truth dtype."""
    # Invalid predictions retain false correctness and unavailable distances.
    result = {"correct": False, "structural_hamming_distance": None, "graph_edit_distance": None}
    try:
        truth = _numeric_adjacency_matrix(ground_truth)
        prediction = _numeric_adjacency_matrix(predicted, dtype=truth.dtype)
        result["correct"] = bool(np.array_equal(truth, prediction))
        # Pad positional matrices only for SHD; GED must retain original node counts.
        size = max(len(truth), len(prediction))
        true_aligned = np.zeros((size, size), dtype=truth.dtype)
        pred_aligned = np.zeros_like(true_aligned)
        true_aligned[:len(truth), :len(truth)] = truth
        pred_aligned[:len(prediction), :len(prediction)] = prediction
        result["structural_hamming_distance"] = structural_hamming_distance(true_aligned, pred_aligned)
        result["graph_edit_distance"] = 0.0 if result["correct"] else graph_edit_distance(truth, prediction)
    except (TypeError, ValueError, OverflowError, FloatingPointError, nx.NetworkXException):
        pass
    return result


def score_dictionary(ground_truth, predicted, *, weighted=False):
    """Require complete dictionary equality, preserving outgoing-child list order."""
    def canonical(value):
        # Normalize tuple/list JSON encodings while preserving every edge attribute.
        if not isinstance(value, dict):
            raise ValueError("Expected a dictionary")
        result = {}
        for parent, children in value.items():
            node = _node(parent)
            if node in result:
                raise ValueError("Duplicate node keys")
            if weighted:
                if not isinstance(children, dict):
                    raise ValueError("Expected child dictionaries")
                normalized = {}
                for child, attributes in children.items():
                    child_node = _node(child)
                    if child_node in normalized:
                        raise ValueError("Duplicate child keys")
                    normalized[child_node] = json_compatible(attributes)
                result[node] = normalized
            else:
                if not isinstance(children, list):
                    raise ValueError("Expected child lists")
                result[node] = [_node(child) for child in children]
        return result

    try:
        correct = canonical(ground_truth) == canonical(predicted)
    except (TypeError, ValueError, SyntaxError):
        correct = False
    return {"correct": bool(correct)}


def score_prompt_responses(ground_truth, prediction):
    """Align a model's predictions with graph attributes for all four prompt types."""
    # Each task holds the requested true/pred/scores envelope with attribute names.
    tasks = {}
    for prompt_type, attributes in PROMPT_ATTRIBUTES.items():
        task = {"true": {}, "pred": {}, "scores": {}}
        for attribute in attributes:
            field = "pred_" + attribute
            truth, predicted = ground_truth[attribute], prediction.get(field)
            task["true"][attribute] = json_compatible(truth)
            task["pred"][field] = json_compatible(predicted)
            if attribute == "adjacency_matrix":
                metric = score_adjacency_matrix(truth, predicted)
            elif attribute in ("dict_of_lists", "dict_of_dicts"):
                metric = score_dictionary(truth, predicted, weighted=attribute == "dict_of_dicts")
            else:
                metric = score_path_lists(truth, predicted)
            task["scores"][attribute] = metric
        tasks[prompt_type] = task
    return tasks


def score_prompt_replicates(records):
    """Export prompt-first true/pred/scores, then replicate and model keys."""
    tasks = {name: {"true": {}, "pred": {}, "scores": {}} for name in PROMPT_ATTRIBUTES}
    for number, record in records.items():
        number = str(number)
        for prompt_type, attributes in PROMPT_ATTRIBUTES.items():
            tasks[prompt_type]["true"][number] = {attribute: json_compatible(record[attribute])
                                                   for attribute in attributes}
            tasks[prompt_type]["pred"][number] = {}
            tasks[prompt_type]["scores"][number] = {}
        for name, prediction in record.items():
            # Ground-truth cost dictionaries are not model response dictionaries.
            if not isinstance(prediction, dict) or not any(key.startswith("pred_") for key in prediction):
                continue
            for prompt_type, result in score_prompt_responses(record, prediction).items():
                tasks[prompt_type]["pred"][number][name] = result["pred"]
                tasks[prompt_type]["scores"][number][name] = result["scores"]
    return tasks


def summarize_prompt_scores(tasks):
    """Average per-replicate metrics for each model, excluding unavailable distances."""
    summary = {}
    for task in tasks.values():
        for models in task["scores"].values():
            for name, attributes in models.items():
                for attribute, metric in attributes.items():
                    result = summary.setdefault(name, {}).setdefault(
                        "pred_" + attribute, {"attempted": 0, "totals": {}, "counts": {}})
                    result["attempted"] += 1
                    for field, value in metric.items():
                        # Missing distances are unavailable observations, not zero errors.
                        if value is not None:
                            result["totals"][field] = result["totals"].get(field, 0) + value
                            result["counts"][field] = result["counts"].get(field, 0) + 1
    for attributes in summary.values():
        for attribute, result in attributes.items():
            totals, counts = result.pop("totals"), result.pop("counts")
            for field, total in totals.items():
                # Correctness means accuracy; path metrics are macro-averaged over replicates.
                output = "accuracy" if field == "correct" else (
                    "mean_" + field if field in ("structural_hamming_distance", "graph_edit_distance") else field)
                result[output] = total / counts[field]
            # Preserve explicit availability counts even when every distance is missing.
            if attribute == "pred_adjacency_matrix":
                for field, prefix in (("structural_hamming_distance", "shd"), ("graph_edit_distance", "ged")):
                    result.setdefault("mean_" + field, None)
                    result[prefix + "_count"] = counts.get(field, 0)
    return summary


def score_replicates(records, n_paths=1):
    """Return all per-model scores plus aggregates for each representation."""
    # Preserve the legacy scorer for saved runs created before the four-task workflow.
    if records and all("all_shortest_paths" in record for record in records.values()):
        return score_prompt_replicates(records)
    replicates, summary = {}, {}
    for number, record in records.items():
        replicates[number] = {}
        for name, prediction in record.items():
            if not isinstance(prediction, dict) or not any(key.startswith("pred_") for key in prediction):
                continue
            scores = score_prediction(record, prediction, n_paths)
            replicates[number][name] = scores
            model_summary = summary.setdefault(name, {})
            for field, score in scores.items():
                aggregate = model_summary.setdefault(field, {"attempted": 0, "correct": 0, "exact_matches": 0,
                                                             "shd_count": 0, "shd_total": 0,
                                                             "ged_count": 0, "ged_total": 0.0})
                aggregate["attempted"] += 1
                aggregate["correct"] += score.get("correct", score["exact_match"])
                aggregate["exact_matches"] += score["exact_match"]
                distance = score.get("structural_hamming_distance")
                if distance is not None:
                    aggregate["shd_count"] += 1
                    aggregate["shd_total"] += distance
                edit_distance = score.get("graph_edit_distance")
                if edit_distance is not None:
                    aggregate["ged_count"] += 1
                    aggregate["ged_total"] += edit_distance
    for model in summary.values():
        for aggregate in model.values():
            aggregate["accuracy"] = aggregate["correct"] / aggregate["attempted"]
            aggregate["exact_match_accuracy"] = aggregate["exact_matches"] / aggregate["attempted"]
            aggregate["mean_structural_hamming_distance"] = (
                aggregate["shd_total"] / aggregate["shd_count"] if aggregate["shd_count"] else None
            )
            aggregate["mean_graph_edit_distance"] = (
                aggregate["ged_total"] / aggregate["ged_count"] if aggregate["ged_count"] else None
            )
    return {
        "metadata": {
            "structural_hamming_distance": "Count differing nonzero directed edge positions; reversals count twice; weights ignored",
            "graph_edit_distance": "Exact NetworkX GED on directed graphs; node relabeling allowed; weights ignored; unit node/edge insertion/deletion costs; original node counts retained",
            "dictionary_exact_match": "Entire dictionary equality; key order ignored, child list order preserved",
            "matrix_exact_match": "All values and node sets identical after alignment by node label",
            "shortest_paths_correct": "Exactly n_paths distinct predictions, all in the oracle shortest paths",
            "shortest_paths_exact_match": "Complete oracle path set, ignoring path order, without duplicates",
            "missing_predictions": "Correctness 0; SHD and GED null and excluded from their means",
            "n_paths": n_paths,
        },
        "replicates": replicates, "summary": summary,
    }


def write_scores(run_directory, records, n_paths=1):
    """Persist scores from the same snapshot used by the graph export."""
    scores = score_replicates(records, n_paths)
    path = Path(run_directory) / "scores.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return scores


def score_saved_run(run_directory):
    """Score existing exported predictions independently, without API calls."""
    run_directory = Path(run_directory)
    records = json.loads((run_directory / "graph_replicates.json").read_text(encoding="utf-8"))
    metadata_path = run_directory / "run_metadata.json"
    report = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    write_scores(run_directory, records, report.get("config", {}).get("n_paths", 1))
    return run_directory / "scores.json"



def create_model_summary(effort):
    """Initialize the counters used for one model's evaluation summary."""
    return {
        "correct": 0, "attempted": 0, "accuracy": 0.0, "api_errors": 0, "format_errors": 0,
        "effort": effort,
    }


def update_model_summary(summary, result):
    """Accumulate one response's correctness and error counters in place."""
    summary["attempted"] += 1
    summary["correct"] += int(result["correct"])
    summary["accuracy"] = summary["correct"] / summary["attempted"]
    summary["api_errors"] += int(result["error_type"] == "api")
    summary["format_errors"] += int(result["error_type"] == "format")
