"""Query vision models about graph representations and three path objectives.

Run directly or as a module, for example::

    python -m query_graph_world --model openai:MODEL_ID --n-replicates 10

Each model sees the graph image first, followed by four questions in one
conversation: graph_rep, shortest_paths, pareto_frontier, scalarized. Conversation
history is isolated per model and replicate. Only image/task information is sent;
ground-truth paths and costs are never supplied. Shortest paths minimize hops;
Pareto paths minimize all named costs; scalarized paths minimize their sum.
--n_paths controls the number of unweighted shortest paths requested. The scalar
question requests one path, matching the graph object's scalar_dijkstra attribute.

Use --mock (alias --mock-run) to generate every artifact and score synthetic
oracle-based answers without any API calls, keys, or .env loading. Mock results
are marked in run metadata, each response, and log.txt; these are pipeline checks,
not measurements of model ability. For example::

    python -m query_graph_world --mock --n-replicates 1

Load provider credentials from environment variables or the repository .env:
OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY). Provider
SDKs are not required. No retries are made; later questions continue after errors.
Effort settings and default models are defined below and configurable by CLI.

Each timestamped run contains PNGs, log.txt, ground_truth.txt, run_metadata.json,
graph_replicates.json, scores.json, MODEL_responses.json, and
MODEL_prompts.txt per model. Each replicate is generated and rendered once; all
models receive the same immutable PNG bytes, verified by their recorded SHA-256.
Response files map replicate numbers to prompt types, retaining raw text, parsed
predictions, usage, and errors. Unsafe filename characters are replaced by '_';
colliding model filenames receive provider and identity suffixes.

scores.json has prompt types as its first-level keys. Each maps to true, pred,
and scores dictionaries keyed by replicate number; pred and scores additionally
contain provider/model keys. Attributes retain their GraphGenerator names and
predictions add 'pred_'. Matrices have correct, structural_hamming_distance, and
graph_edit_distance; dictionaries have correct; path lists have precision,
recall, and f1. Full ground-truth path attributes retain paths and costs, but only
complete paths are compared. Duplicate predictions cannot create extra matches.

JSON encodes arrays as lists, tuple node keys as tuple strings, and path nodes
as integer pairs. The graph question explicitly supplies the matrix node order
and the edge-cost names shown in the image legend. --score-run rescores saved
predictions without API calls; --export-run rebuilds exports from run metadata.
The end-of-run tables show per-model averages for every metric across replicates
and are mirrored to log.txt. Missing distances are excluded, with counts shown.
Older saved runs continue to use their original scoring schema.

Provider conversation formats:
https://developers.openai.com/api/docs/guides/conversation-state
https://platform.claude.com/docs/en/build-with-claude/vision
https://ai.google.dev/gemini-api/docs/generate-content/text-generation
"""

import argparse
import ast
import base64
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

import networkx as nx

if __package__:
    from .generate_graph_world import GraphGenerator
    from . import score_graph_world
else:
    from generate_graph_world import GraphGenerator
    import score_graph_world


KEY_NAMES = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

DEFAULT_MODELS = [
    ("openai", "gpt-6.1-sol"),
    ("anthropic", "claude-fable-5-1"),
    ("gemini", "gemini-4-argon"),
]

DEFAULT_EFFORTS = {"openai": "max", "anthropic": "max", "gemini": "high"}

EFFORT_LEVELS = {
    "openai": ("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    "anthropic": ("low", "medium", "high", "xhigh", "max"),
    "gemini": ("minimal", "low", "medium", "high"),
}

# Format templates when called, rather than referencing undefined variables at import.
PROMPT_GRAPH_REP = (
        "You are given a diagram of a directed graph with tuple edge weights. Do the following: "
        "(1) Provide the adjacency matrix for this directed graph as a 2D array of 0s and 1s, "
        "where entry i, j corresponds to an edge from i to j (for rows i and columns j). "
        "(2) Provide the directed graph as a dictionary of lists, mapping each node tuple to its list of child tuples. "
        "(3) Provide the directed graph as a dictionary of dictionaries, mapping each parent tuple to a dictionary "
        "whose keys are child tuples and values are edge weight dictionaries. The edge weight dictionary should map the "
        "name of the edge weight to its corresponding value. "
        "Return only a Python dictionary with exactly three fields: "
        "'pred_dict_of_lists', 'pred_dict_of_dicts', and 'pred_adjacency_matrix'. "
        "Do not explain your reasoning or add Markdown, code fences, costs, or other text."
)

PROMPT_SHORTEST_PATHS = (
        "Find {n_paths} shortest directed {path_word} from source node {source!r} to sink node {target!r}. "
        "In this case, shortest means the minimum number of directed edges, independent of edge weights. "
        "Return only a dictionary with the key 'pred_all_shortest_paths', whose value is a Python list containing "
        "exactly {n_paths} distinct paths. "
        "Each path must be a Python list of tuples containing the node names in path order. "
        "If several such shortest paths exist, return any {n_paths}. "
        "Do not explain your reasoning or add Markdown, code fences, costs, or other text."
)

PROMPT_PARETO_FRONTIER = (
    "Find all Pareto-optimal paths from source node {source!r} to sink node {target!r} with respect to the multiple edge weights. "
    "This is a multi-objective optimization problem. Return a dictionary with the key 'pred_multiobjective_shortest_paths', whose value is a Python list of paths. "
    "Each path must be a list of node tuples in path order. "
    "Do not explain your reasoning or add Markdown, code fences, costs, or other text."
)

PROMPT_SCALARIZED = (
    "Find {n_paths} shortest directed paths from source node {source!r} to sink node {target!r} with respect to the scalarized sum of edge weights. "
    "The scalarized sum for a single edge is computed by summing the edge weights that edge. "
    "For a given path, sum all edge sums along the full path. "
    "Return only a dictionary with the key 'pred_scalar_dijkstra', whose value is a Python list containing "
    "exactly {n_paths} distinct paths. "
    "If several such shortest paths exist, return any {n_paths}. "
    "Do not explain your reasoning or add Markdown, code fences, costs, or other text."
)

PROMPTS = {
    "graph_rep": PROMPT_GRAPH_REP,
    "shortest_paths": PROMPT_SHORTEST_PATHS,
    "pareto_frontier": PROMPT_PARETO_FRONTIER,
    "scalarized": PROMPT_SCALARIZED,
}


def conversation_messages(provider, png, prompt, history=None):
    """Replay previous turns with the image at the start of the conversation."""
    # Every HTTP request is stateless; preserve one logical image-first conversation.
    encoded = base64.b64encode(png).decode("ascii")
    turns = list(history or []) + [{"prompt": prompt, "response": None}]
    messages = []
    for index, turn in enumerate(turns):
        if provider == "openai":
            content = []
            if index == 0:
                content.append({"type": "input_image", "image_url": f"data:image/png;base64,{encoded}", "detail": "high"})
            content.append({"type": "input_text", "text": turn["prompt"]})
            messages.append({"role": "user", "content": content})
            if turn["response"] is not None:
                messages.append({"role": "assistant", "content": turn["response"]})
        elif provider == "anthropic":
            content = []
            if index == 0:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}})
            content.append({"type": "text", "text": turn["prompt"]})
            messages.append({"role": "user", "content": content})
            if turn["response"] is not None:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": turn["response"]}]})
        elif provider == "gemini":
            parts = []
            if index == 0:
                parts.append({"inlineData": {"mimeType": "image/png", "data": encoded}})
            parts.append({"text": turn["prompt"]})
            messages.append({"role": "user", "parts": parts})
            if turn["response"] is not None:
                messages.append({"role": "model", "parts": [{"text": turn["response"]}]})
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    return messages


class Tee:
    """Mirror output to the console and a run's log without terminal escapes."""

    def __init__(self, console, logfile):
        self.console = console
        self.logfile = logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def capture_log(path):
    """Capture all Python stdout/stderr during a run, including fatal errors."""
    with path.open("w", encoding="utf-8", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                yield
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                sys.stdout.flush()
                sys.stderr.flush()


def create_run_directory(parent):
    """Reserve a new timestamped run directory without overwriting prior runs."""
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    name = datetime.now(ZoneInfo("America/New_York")).strftime("run_%Y-%m-%d_%H-%M-%S")
    suffix = 1
    while True:
        directory = parent / (name if suffix == 1 else f"{name}_{suffix}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            suffix += 1


def print_model_progress(name, processed, total):
    """Print a per-model progress bar; newline updates keep log.txt readable."""
    width = 24
    filled = width * processed // total
    bar = "#" * filled + "-" * (width - filled)
    print(f"  Progress {name}: [{bar}] {processed}/{total}", flush=True)


def load_env(path):
    """Load dotenv assignments without executing shell code or replacing env vars."""
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid .env assignment at line {number}; expected NAME=value")
        if value.lstrip().startswith(("'", '"')):
            try:
                parts = shlex.split(value, comments=True, posix=True)
            except ValueError:
                raise ValueError(f"Invalid quoting in .env line {number}") from None
            if len(parts) > 1:
                raise ValueError(f"Invalid .env value at line {number}")
            value = parts[0] if parts else ""
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        os.environ.setdefault(name, value)


def api_key(provider):
    """Read the selected provider's key; never include its value in an error."""
    for name in KEY_NAMES[provider]:
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    raise ValueError(f"Missing {' or '.join(KEY_NAMES[provider])}; set it in .env or the environment")


def post_json(url, headers, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Avoid logging headers or image bodies. The caller redacts keys from
        # any provider error message before displaying or saving it.
        try:
            error = json.loads(exc.read()).get("error", {})
            message = error.get("message", exc.reason) if isinstance(error, dict) else str(error)
        except (ValueError, UnicodeDecodeError):
            message = exc.reason
        raise RuntimeError(f"HTTP {exc.code}: {message}") from None


def query_model(provider, model, key, png, prompt, timeout=600, max_tokens=32768, *, effort=None, history=None):
    """Query one conversation turn; return its visible text and token usage."""
    if provider not in EFFORT_LEVELS:
        raise ValueError(f"Unsupported provider: {provider}")
    if effort is not None and effort not in EFFORT_LEVELS[provider]:
        raise ValueError(f"Unsupported effort for {provider}: {effort}")
    messages = conversation_messages(provider, png, prompt, history)
    if provider == "openai":
        effort_settings = {} if effort is None else {"reasoning": {"effort": effort}}
        result = post_json(
            "https://api.openai.com/v1/responses", {"Authorization": f"Bearer {key}"},
            {"model": model, "store": False, "max_output_tokens": max_tokens, **effort_settings,
             "input": messages}, timeout,
        )
        if result.get("status") != "completed":
            raise RuntimeError(f"OpenAI response was not completed: {result.get('status')}")
        text = "".join(
            block.get("text", "") for item in result.get("output", [])
            for block in item.get("content", []) if block.get("type") == "output_text"
        )
    elif provider == "anthropic":
        effort_settings = {} if effort is None else {"output_config": {"effort": effort}}
        result = post_json(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            {"model": model, "max_tokens": max_tokens, **effort_settings, "messages": messages}, timeout,
        )
        if result.get("stop_reason") != "end_turn":
            raise RuntimeError(f"Anthropic response did not finish normally: {result.get('stop_reason')}")
        text = "".join(block.get("text", "") for block in result.get("content", []) if block.get("type") == "text")
    elif provider == "gemini":
        model_id = urllib.parse.quote(model.removeprefix("models/"), safe="")
        generation_config = {"maxOutputTokens": max_tokens}
        if effort is not None:
            generation_config["thinkingConfig"] = {"thinkingLevel": effort}
        result = post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
            {"x-goog-api-key": key},
            {"contents": messages, "generationConfig": generation_config}, timeout,
        )
        candidates = result.get("candidates", [])
        if not candidates or candidates[0].get("finishReason") != "STOP":
            raise RuntimeError("Gemini response was blocked, incomplete, or missing")
        text = "".join(part.get("text", "") for part in candidates[0].get("content", {}).get("parts", [])
                       if not part.get("thought", False))
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    if not text.strip():
        raise RuntimeError("Provider returned no visible text answer")
    return text, result.get("usage", result.get("usageMetadata"))


def make_prompt(source, target, n_paths=1, prompt_type="graph_rep", *, node_order=None,
                edge_cost_names=None, cost_precision=12):
    """Select a task template and format its endpoints and requested path count."""
    # Validate the selector and count before any request can be made.
    if prompt_type not in PROMPTS:
        raise ValueError("prompt_type must be one of " + ", ".join(PROMPTS))
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    return PROMPTS[prompt_type].format(
        source=source, target=target, n_paths=n_paths, path_word="path" if n_paths == 1 else "paths",
        node_order="sort all node tuples lexicographically" if node_order is None else node_order,
        edge_cost_names=["water", "time", "energy"] if edge_cost_names is None else edge_cost_names,
        cost_precision=cost_precision,
    )


def parse_prompt_prediction(text, prompt_type):
    """Parse a single task without deriving one model representation from another."""
    # Accept Python tuple-key dictionaries and JSON responses, preserving raw text elsewhere.
    try:
        prediction = ast.literal_eval(text.strip())
    except (TypeError, ValueError, SyntaxError, RecursionError):
        try:
            prediction = json.loads(text)
        except (TypeError, ValueError, RecursionError):
            raise ValueError("Response must contain only a Python or JSON dictionary") from None
    required = {"pred_" + attribute for attribute in score_graph_world.PROMPT_ATTRIBUTES[prompt_type]}
    if not isinstance(prediction, dict) or set(prediction) != required:
        raise ValueError("Response needs exactly " + ", ".join(sorted(required)))
    try:
        return json.loads(json.dumps(score_graph_world.json_compatible(prediction), allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError("Predictions must use JSON-compatible values and tuple node keys") from None


def mock_response(ground_truth, prompt_type, n_paths=1):
    """Return a synthetic oracle-based answer to exercise the normal scoring pipeline."""
    # Use known graph attributes rather than credentials, provider code, or network access.
    prediction = {}
    for attribute in score_graph_world.PROMPT_ATTRIBUTES[prompt_type]:
        value = ground_truth[attribute]
        if isinstance(value, dict) and "paths" in value:
            value = value["paths"]
            if prompt_type == "shortest_paths":
                value = value[:n_paths]
            elif prompt_type == "scalarized":
                value = value[:1]
        prediction["pred_" + attribute] = value
    return json.dumps(score_graph_world.json_compatible(prediction), allow_nan=False), {"mock": True}


def graph_ground_truth(generator):
    """Snapshot the actual cached graph attributes, including named edge costs."""
    attributes = {name for names in score_graph_world.PROMPT_ATTRIBUTES.values() for name in names}
    # Record native matrix ordering explicitly; never reorder a graph attribute for scoring.
    result = {name: getattr(generator, name) for name in sorted(attributes)}
    result.update(node_order=list(generator.graph), source=generator.source, target=generator.target,
                  edge_cost_names=generator.edge_cost_names, cost_precision=generator.cost_precision)
    return score_graph_world.json_compatible(result)


def response_filenames(models):
    """Create safe per-model filenames, disambiguating equal or sanitized names."""
    # Prefer the requested model name, adding an identity suffix only on collision.
    stems = {pair: re.sub(r"[^a-zA-Z0-9_.-]+", "_", pair[1]).strip(".") or "model" for pair in models}
    filenames = {}
    for pair, stem in stems.items():
        if list(stems.values()).count(stem) > 1:
            identity = hashlib.sha256(repr(pair).encode()).hexdigest()[:10]
            stem = f"{pair[0]}_{stem}_{identity}"
        filenames[pair] = stem + "_responses.json"
    return filenames


def parse_path(text, n_paths=1):
    """Parse one node path, or exactly n_paths node paths for a multi-path task."""
    if n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    try:
        parsed = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError, TypeError, RecursionError):
        raise ValueError("Response must contain only a Python list of node tuples or paths") from None
    paths = [parsed] if n_paths == 1 else parsed
    if not isinstance(paths, list) or len(paths) != n_paths:
        raise ValueError(f"Response must contain exactly {n_paths} paths")
    for path in paths:
        if not isinstance(path, list) or any(
            not isinstance(node, tuple) or len(node) != 2 or any(type(value) is not int for value in node)
            for node in path
        ):
            raise ValueError("Every path must be a list of tuples of two integers")
    return parsed


def parse_prediction(text, n_paths=1):
    """Preserve each independent model representation for separate scoring.

    Validate the response envelope here; metrics validate individual fields so
    one malformed representation does not erase other, correct predictions.
    """
    try:
        prediction = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError, TypeError, RecursionError):
        raise ValueError("Response must contain only a Python dictionary of graph and paths") from None
    required = {"pred_dict_of_lists", "pred_dict_of_dicts", "pred_adjacency_matrix", "pred_shortest_paths"}
    if not isinstance(prediction, dict) or set(prediction) != required:
        raise ValueError("Response needs " + ", ".join(sorted(required)))
    # JSON cannot store tuple keys. Preserve dictionaries and child-list order;
    # do not reconstruct one prediction using another representation.
    for field in ("pred_dict_of_lists", "pred_dict_of_dicts"):
        value = prediction[field]
        if isinstance(value, dict):
            prediction[field] = {
                repr(node) if isinstance(node, tuple) else str(node): (
                    {repr(child) if isinstance(child, tuple) else str(child): weight
                     for child, weight in children.items()}
                    if field == "pred_dict_of_dicts" and isinstance(children, dict) else children
                ) for node, children in value.items()
            }
    try:
        prediction = json.loads(json.dumps(prediction, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError("Predictions must use JSON-compatible values and tuple node keys") from None
    adjacency = prediction["pred_dict_of_lists"]
    try:
        nodes = sorted(ast.literal_eval(node) for node in adjacency) if isinstance(adjacency, dict) else None
    except (ValueError, SyntaxError, TypeError):
        nodes = None
    prediction["pred_node_order"] = nodes
    prediction["pred_adjacency_array"] = prediction["pred_adjacency_matrix"]
    return prediction


def graph_representations(graph, node_order=None):
    """Return JSON graph forms; matrix rows/columns follow node_order.

    JSON object keys are repr(tuple) strings. Children and paths serialize as
    integer pairs. Missing edge weights use NetworkX's default weight of 1.
    dict_of_dicts stores scalar weights rather than edge attribute dictionaries.
    """
    nodes = list(graph) if node_order is None else list(node_order)
    adjacency = nx.to_dict_of_lists(graph)
    edge_data = nx.to_dict_of_dicts(graph)
    return {
        "node_order": nodes,
        "adjacency_matrix": nx.to_numpy_array(graph, nodelist=nodes).tolist(),
        "dict_of_lists": {repr(node): children for node, children in adjacency.items()},
        "dict_of_dicts": {
            repr(node): {repr(child): data.get("weight", 1) for child, data in children.items()}
            for node, children in edge_data.items()
        },
    }


def prediction_export(graph, paths, ground_truth_nodes):
    """Align predictions to oracle order, appending any model-invented nodes."""
    if graph is None:
        result = dict.fromkeys(("pred_node_order", "pred_adjacency_matrix", "pred_adjacency_array", "pred_dict_of_lists",
                                "pred_dict_of_dicts", "pred_shortest_paths"))
        result["pred_shortest_paths"] = paths
        return result
    nodes = list(ground_truth_nodes)
    nodes.extend(node for node in graph if node not in nodes)
    aligned = graph.copy()
    aligned.add_nodes_from(nodes)
    forms = graph_representations(aligned, nodes)
    return {
        "pred_node_order": forms["node_order"],
        "pred_adjacency_matrix": forms["adjacency_matrix"],
        "pred_adjacency_array": forms["adjacency_matrix"],
        "pred_dict_of_lists": forms["dict_of_lists"],
        "pred_dict_of_dicts": forms["dict_of_dicts"],
        "pred_shortest_paths": paths,
    }


def write_json(path, data):
    """Replace a JSON snapshot only after its full contents have been written."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_saved_run(run_directory):
    """Export recorded results without API calls; legacy graphs remain unknown."""
    run_directory = Path(run_directory)
    report = json.loads((run_directory / "run_metadata.json").read_text(encoding="utf-8"))
    records = {}
    if report.get("config", {}).get("workflow") == "four_prompts":
        for replicate in report["replicates"]:
            record = dict(replicate["ground_truth"])
            for result in replicate["responses"]:
                record[f"{result['provider']}/{result['model']}"] = result["predictions"]
            records[str(replicate["replicate"])] = record
        path = run_directory / "graph_replicates.json"
        write_json(path, records)
        score_graph_world.write_scores(run_directory, records)
        return path
    for replicate in report["replicates"]:
        graph = nx.DiGraph()
        graph.add_nodes_from(tuple(node) for node in replicate["nodes"])
        graph.add_edges_from((tuple(parent), tuple(child)) for parent, child in replicate["edges"])
        record = {
            **graph_representations(graph),
            "source": replicate["source"], "sink": replicate["sink"],
            "shortest_path_length": replicate["shortest_length"],
            "total_shortest_paths": len(replicate["shortest_paths"]),
            "shortest_paths": replicate["shortest_paths"],
        }
        for provider, model in report["config"]["models"]:
            record[f"{provider}/{model}"] = prediction_export(None, None, [])
        for response in replicate["responses"]:
            model_prediction = None
            paths = response.get("paths")
            if paths is None and response.get("path") is not None:
                paths = [response["path"]]
            if response.get("error_type") != "api" and response.get("response"):
                try:
                    model_prediction = parse_prediction(
                        response["response"], n_paths=report["config"].get("n_paths", 1),
                    )
                except ValueError:
                    # Older runs requested paths alone. Preserve the recorded
                    # paths, but never infer a full predicted graph from them.
                    pass
            name = f"{response['provider']}/{response['model']}"
            record[name] = model_prediction if model_prediction is not None else prediction_export(None, paths, [])
        records[str(replicate["replicate"])] = record
    path = run_directory / "graph_replicates.json"
    write_json(path, records)
    score_graph_world.write_scores(run_directory, records, report["config"].get("n_paths", 1))
    return path


def print_summary(summaries):
    width = max(5, *(len(name) for name in summaries))
    print("\nShortest directed path accuracy (API and format errors count as incorrect)")
    print(f"{'Model':<{width}}  {'Effort':<16}  {'Correct':>9}  {'Accuracy':>9}  {'API errors':>10}  {'Invalid outputs':>15}")
    print("-" * (width + 78))
    for name, summary in summaries.items():
        ratio = f"{summary['correct']}/{summary['attempted']}"
        effort = summary["effort"] if summary["effort"] is not None else "provider default"
        print(f"{name:<{width}}  {effort:<16}  {ratio:>9}  {summary['accuracy']:>8.2%}  "
              f"{summary['api_errors']:>10}  {summary['format_errors']:>15}")

    print("\nAverage scores across replicates (path precision/recall/F1 are macro-averages)")
    # Group by prediction category so models can be compared side by side.
    categories = dict.fromkeys(attribute for summary in summaries.values()
                               for attribute in summary.get("scores", {}))
    for attribute in categories:
        columns = [("Replicates", "attempted", 10)]
        if attribute == "pred_adjacency_matrix":
            columns += [("accuracy", "accuracy", 12), ("mean SHD", "mean_structural_hamming_distance", 12),
                        ("SHD count", "shd_count", 10), ("mean GED", "mean_graph_edit_distance", 12),
                        ("GED count", "ged_count", 10)]
        elif attribute in ("pred_dict_of_lists", "pred_dict_of_dicts"):
            columns += [("accuracy", "accuracy", 12)]
        else:
            columns += [(metric, metric, 12) for metric in ("precision", "recall", "f1")]
        print(f"\n{attribute}")
        header = f"{'Model':<{width}}  " + "  ".join(f"{label:>{size}}" for label, _, size in columns)
        print(header)
        print("-" * len(header))
        for name, summary in summaries.items():
            values = summary.get("scores", {}).get(attribute, {})
            cells = []
            for _, metric, size in columns:
                value = values.get(metric)
                # Retain missing-distance markers and their valid observation counts.
                if value is None:
                    cell = "N/A"
                elif metric in ("attempted", "shd_count", "ged_count"):
                    cell = str(value)
                else:
                    cell = f"{value:.6f}"
                cells.append(f"{cell:>{size}}")
            print(f"{name:<{width}}  " + "  ".join(cells))


def append_prompt_transcript(path, replicate, prompt_type, prompt):
    """Record exact user prompts and the shared image identity before each API call."""
    with path.open("a", encoding="utf-8") as stream:
        # Separate graph conversations and identify the exact PNG shown to the model.
        if prompt_type == next(iter(PROMPTS)):
            stream.write(f"Replicate {replicate['replicate']}\nImage: {replicate['image']}\n")
            stream.write(f"Image SHA-256: {replicate['image_sha256']}\n\n")
        stream.write(f"User prompt ({prompt_type}):\n{prompt}\n\n")


def run_evaluation(models, n_replicates, output_dir, *, n_subgraphs=3,
                   n_nodes_per_subgraph=5, out_degree=2, themes=None,
                   timeout=600, max_tokens=32768, openai_effort=DEFAULT_EFFORTS["openai"],
                   anthropic_effort=DEFAULT_EFFORTS["anthropic"], gemini_effort=DEFAULT_EFFORTS["gemini"], n_paths=1,
                   mock=False):
    """Ask four ordered questions per model/graph and checkpoint responses and scores."""
    # Validate configuration before creating files or making any provider requests.
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    if n_replicates < 1 or not models:
        raise ValueError("Provide at least one replicate and one provider/model pair")
    models = list(dict.fromkeys(models))
    efforts = {"openai": openai_effort, "anthropic": anthropic_effort, "gemini": gemini_effort}
    for provider, effort in efforts.items():
        if effort is not None and effort not in EFFORT_LEVELS[provider]:
            raise ValueError(f"Unsupported effort for {provider}: {effort}")
    themes = ["physics", "spatial", "quantitative"] if themes is None else themes
    # Mock runs never look up credentials or enter the provider request code.
    keys = {} if mock else {provider: api_key(provider) for provider, _ in models}
    output_dir = create_run_directory(output_dir)
    with capture_log(output_dir / "log.txt"):
        summaries = {f"{provider}/{model}": score_graph_world.create_model_summary(efforts[provider])
                     for provider, model in models}
        filenames = response_filenames(models)
        # Reuse collision-safe model stems for both response files and prompt transcripts.
        prompt_files = {pair: filename.removesuffix("_responses.json") + "_prompts.txt"
                        for pair, filename in filenames.items()}
        responses = {f"{provider}/{model}": {} for provider, model in models}
        report = {
            "run_directory": str(output_dir), "timezone": "America/New_York",
            "config": {"workflow": "four_prompts", "mock": mock, "prompt_types": list(PROMPTS), "models": models,
                       "n_replicates": n_replicates, "n_subgraphs": n_subgraphs, "n_paths": n_paths,
                       "n_nodes_per_subgraph": n_nodes_per_subgraph, "out_degree": out_degree,
                       "themes": themes, "timeout": timeout, "max_tokens": max_tokens, "efforts": efforts},
            "response_files": {f"{provider}/{model}": filenames[provider, model] for provider, model in models},
            "prompt_files": {f"{provider}/{model}": prompt_files[provider, model] for provider, model in models},
            "replicates": [], "summary": summaries,
        }
        graph_replicates = {}
        graph_export_file = output_dir / "graph_replicates.json"
        ground_truth_file = output_dir / "ground_truth.txt"
        ground_truth_file.write_text("Ground truth shortest directed paths (minimum number of edges)\n\n", encoding="utf-8")
        print(f"Run directory: {output_dir}", flush=True)
        if mock:
            print("MOCK RUN: synthetic oracle-based responses; no model APIs or credentials used.", flush=True)
        (output_dir / "images_seen_by_model").mkdir()
        for provider, model in models:
            print_model_progress(f"{provider}/{model}", 0, n_replicates)
            write_json(output_dir / filenames[provider, model], {})
            (output_dir / prompt_files[provider, model]).write_text(f"Model: {provider}/{model}\n\n", encoding="utf-8")
        for index in range(1, n_replicates + 1):
            # Generate and render outside the model loop so random graphs cannot differ by model.
            generator = GraphGenerator(n_subgraphs=n_subgraphs, n_nodes_per_subgraph=n_nodes_per_subgraph,
                                       max_subgraph_outdegree=out_degree, themes=themes)
            # Reuse initialization's complete path oracle instead of enumerating again.
            shortest_paths = generator.all_shortest_paths["paths"]
            with ground_truth_file.open("a", encoding="utf-8") as ground_truth:
                ground_truth.write(f"Replicate {index}\nSource: {generator.source!r}\nTarget: {generator.target!r}\n")
                length = len(shortest_paths[0]) - 1 if shortest_paths else None
                ground_truth.write(f"Shortest length (edges): {length}\nNumber of shortest paths: {len(shortest_paths)}\n")
                ground_truth.write("Paths:\n")
                for path in shortest_paths:
                    ground_truth.write(f"{path!r}\n")
                ground_truth.write("\n")
            if len(shortest_paths) < n_paths:
                raise ValueError(f"Replicate {index} has only {len(shortest_paths)} distinct shortest paths; "
                                 f"n_paths={n_paths} requires at least {n_paths}. Reduce --n_paths.")
            truth = graph_ground_truth(generator)
            # Keep each attribute prediction independent, including missing/error values.
            empty_prediction = {"pred_" + attribute: None for attributes in score_graph_world.PROMPT_ATTRIBUTES.values()
                                for attribute in attributes}
            exported = {**truth, **{f"{provider}/{model}": dict(empty_prediction) for provider, model in models}}
            graph_replicates[str(index)] = exported
            write_json(graph_export_file, graph_replicates)
            image = Path("images_seen_by_model") / f"replicate_{index:04d}.png"
            generator.plot_graph(generator.graph, output_dir / image, edge_cost=generator.edge_cost_names)
            # Read the PNG once so every model and conversation uses identical image bytes.
            png = (output_dir / image).read_bytes()
            image_sha256 = hashlib.sha256(png).hexdigest()
            prompts = {prompt_type: make_prompt(
                generator.source, generator.target, n_paths=1 if prompt_type == "scalarized" else n_paths,
                prompt_type=prompt_type, node_order=list(generator.graph),
                edge_cost_names=generator.edge_cost_names, cost_precision=generator.cost_precision,
            ) for prompt_type in PROMPTS}
            replicate = {"replicate": index, "image": str(image), "image_sha256": image_sha256,
                         "source": generator.source, "target": generator.target, "ground_truth": truth,
                         "shortest_paths": shortest_paths, "shortest_length": len(shortest_paths[0]) - 1,
                         "prompts": prompts, "responses": []}
            report["replicates"].append(replicate)
            print(f"Replicate {index}/{n_replicates}: {generator.source} -> {generator.target}", flush=True)
            print(f"Image SHA-256: {image_sha256}", flush=True)
            for provider, model in models:
                name = f"{provider}/{model}"
                prediction = exported[name]
                prompt_responses = {}
                responses[name][str(index)] = prompt_responses
                result = {"provider": provider, "model": model, "effort": efforts[provider],
                          "predictions": prediction, "prompt_responses": prompt_responses, "mock": mock,
                          "response": None, "paths": None, "path": None, "path_correctness": None,
                          "correct": False, "error": None, "error_type": None, "image_sha256": image_sha256}
                replicate["responses"].append(result)
                history = []
                started = time.perf_counter()
                for prompt_type, prompt in prompts.items():
                    # A fresh history per model/replicate prevents cross-model answer leakage.
                    entry = {"prompt": prompt, "response": None, "prediction": None,
                             "usage": None, "error": None, "error_type": None, "image_sha256": image_sha256, "mock": mock}
                    prompt_responses[prompt_type] = entry
                    append_prompt_transcript(output_dir / prompt_files[provider, model], replicate, prompt_type, prompt)
                    turn_started = time.perf_counter()
                    try:
                        if mock:
                            # Synthetic answers still pass through the real parser, scorer, and exports.
                            answer, usage = mock_response(truth, prompt_type, n_paths)
                        else:
                            answer, usage = query_model(provider, model, keys[provider], png, prompt,
                                                        timeout, max_tokens, effort=efforts[provider], history=list(history))
                        entry.update(response=answer, usage=usage)
                    except (OSError, RuntimeError, ValueError) as exc:
                        # Redact provider messages before console output or any JSON checkpoint.
                        message = str(exc)
                        for key in keys.values():
                            message = message.replace(key, "[REDACTED]")
                        entry.update(error=message, error_type="api")
                    else:
                        # Preserve visible answers as context even if their output format is invalid.
                        history.append({"prompt": prompt, "response": answer})
                        try:
                            parsed = parse_prompt_prediction(answer, prompt_type)
                            entry["prediction"] = parsed
                            prediction.update(parsed)
                        except ValueError as exc:
                            entry.update(error=str(exc), error_type="format")
                    entry["elapsed_seconds"] = time.perf_counter() - turn_started
                    print(f"  {name} {prompt_type} output:\n{entry['response'] or '<no response>'}", flush=True)
                    if entry["error"]:
                        print(f"  {name} {prompt_type}: {entry['error_type']}: {entry['error']}", flush=True)
                    # Persist every task independently so interruption never loses earlier answers.
                    write_json(output_dir / filenames[provider, model], responses[name])
                    write_json(graph_export_file, graph_replicates)
                    write_json(output_dir / "run_metadata.json", report)
                result["elapsed_seconds"] = time.perf_counter() - started
                shortest = prompt_responses["shortest_paths"]
                paths = prediction["pred_all_shortest_paths"]
                path_scores = score_graph_world.score_shortest_paths(shortest_paths, paths, n_paths)
                result.update(response=shortest["response"], paths=paths,
                              path=paths[0] if isinstance(paths, list) and len(paths) == 1 else None,
                              path_correctness=path_scores["path_correctness"], correct=bool(path_scores["correct"]))
                errors = [entry for entry in prompt_responses.values() if entry["error"]]
                if errors:
                    result.update(error=errors[0]["error"], error_type="api" if any(
                        entry["error_type"] == "api" for entry in errors) else "format")
                # All evaluation metrics and aggregations remain in score_graph_world.py.
                scores = score_graph_world.write_scores(output_dir, graph_replicates, n_paths)
                result["scores"] = {task: value["scores"][str(index)][name] for task, value in scores.items()}
                for scored_name, aggregate in score_graph_world.summarize_prompt_scores(scores).items():
                    summaries[scored_name]["scores"] = aggregate
                score_graph_world.update_model_summary(summaries[name], result)
                print_model_progress(name, summaries[name]["attempted"], n_replicates)
                write_json(output_dir / "run_metadata.json", report)
        print_summary(summaries)
        print(f"\nSaved images and results: {output_dir.resolve()}")
        return report


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return value


def model_spec(text):
    provider, separator, model = text.partition(":")
    if provider not in KEY_NAMES or not separator or not model.strip():
        raise argparse.ArgumentTypeError("Use openai:MODEL_ID, anthropic:MODEL_ID, or gemini:MODEL_ID")
    return provider, model.strip()


def integer_or_range(values, name):
    """Parse an integer, a quoted (lower, upper) tuple, or separate bounds."""
    message = f'{name} needs a nonnegative integer, a quoted "(LOWER,UPPER)" tuple, or LOWER UPPER'
    try:
        value = ast.literal_eval(" ".join(str(part) for part in values))
    except (ValueError, SyntaxError, TypeError, RecursionError):
        # Preserve support for the original two-token range syntax.
        try:
            numbers = tuple(int(part) for part in values)
            value = numbers[0] if len(numbers) == 1 else numbers
        except (ValueError, TypeError):
            raise ValueError(message) from None
    if type(value) is int and value >= 0:
        return value
    if (isinstance(value, tuple) and len(value) == 2
            and all(type(bound) is int for bound in value)
            and 0 <= value[0] <= value[1]):
        return value
    raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", action="extend", nargs="+", type=model_spec, default=None,
        help="List of PROVIDER:MODEL_ID values; replaces defaults and may be repeated. Defaults: "
             + " ".join(f"{provider}:{model}" for provider, model in DEFAULT_MODELS),
    )
    for provider, levels in EFFORT_LEVELS.items():
        parser.add_argument(
            f"--{provider}-effort", choices=levels, default=DEFAULT_EFFORTS[provider],
            help=f"Effort for all {provider} models (model-dependent; default: {DEFAULT_EFFORTS[provider]})",
        )
    parser.add_argument("--mock", "--mock-run", action="store_true",
                        help="Use synthetic oracle answers; generate all artifacts without credentials or API calls")
    parser.add_argument("--n-replicates", "--n_replicates", type=positive_int, default=10)
    parser.add_argument("--n_paths", "--n-paths", type=positive_int, default=1,
                        help="Number of distinct equally shortest directed paths to request (default: 1)")
    parser.add_argument("--n-subgraphs", type=positive_int, default=3)
    parser.add_argument("--nodes-per-subgraph", nargs="+", default=[5], metavar="N",
                        help='Integer, quoted tuple "(LOWER,UPPER)", or two inclusive bounds (default: 5)')
    parser.add_argument("--out-degree", nargs="+", default=[2], metavar="D",
                        help='Integer, quoted tuple "(LOWER,UPPER)", or two inclusive bounds (default: 2)')
    parser.add_argument("--themes", nargs="+", default=["physics", "spatial", "quantitative"])
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent / ".env")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "api_results",
        help="Parent directory for run_DATE_HH-MM-SS folders (default: api_results beside this script)",
    )
    parser.add_argument("--timeout", type=positive_int, default=600, help="Request timeout in seconds")
    parser.add_argument("--max-output-tokens", type=positive_int, default=32768)
    parser.add_argument("--export-run", type=Path,
                        help="Export an existing run's run_metadata.json without model requests")
    parser.add_argument("--score-run", type=Path,
                        help="Score an existing graph_replicates.json in a run directory without API calls")
    args = parser.parse_args()
    try:
        if args.export_run is not None and args.score_run is not None:
            raise ValueError("Choose either --export-run or --score-run")
        if args.score_run is not None:
            print(f"Saved scores: {score_graph_world.score_saved_run(args.score_run)}")
            return
        if args.export_run is not None:
            print(f"Saved graph export: {export_saved_run(args.export_run)}")
            return
        nodes = integer_or_range(args.nodes_per_subgraph, "--nodes-per-subgraph")
        degree = integer_or_range(args.out_degree, "--out-degree")
        minimum_nodes = nodes if isinstance(nodes, int) else nodes[0]
        maximum_degree = degree if isinstance(degree, int) else degree[1]
        if minimum_nodes < 1 or maximum_degree > (minimum_nodes - 1) // 2:
            raise ValueError("All sampled graphs require n >= 1 and d <= (n - 1) // 2")
        # A mock run does not need to read credential files.
        if not args.mock:
            load_env(args.env_file)
        run_evaluation(
            list(dict.fromkeys(args.model if args.model is not None else DEFAULT_MODELS)),
            args.n_replicates, args.output_dir,
            n_subgraphs=args.n_subgraphs, n_nodes_per_subgraph=nodes, out_degree=degree,
            themes=args.themes, timeout=args.timeout, max_tokens=args.max_output_tokens,
            openai_effort=args.openai_effort, anthropic_effort=args.anthropic_effort,
            gemini_effort=args.gemini_effort, n_paths=args.n_paths, mock=args.mock,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
