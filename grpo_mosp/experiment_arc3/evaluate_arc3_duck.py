"""Run Tufa's ARC-AGI-3 benchmark with frozen Qwen and selectable policies."""

import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import sys
import time
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean
from threading import Lock
from types import SimpleNamespace

if __name__ == "__main__":
    sys.dont_write_bytecode = True

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "jm_test.graph_world.grpo_mosp.experiment_arc3"

from ..experiment_downstream_tasks.checkpoints import MODEL_CONFIG, load_selection, read_json, serve_checkpoint
from ..experiment_downstream_tasks.serve_qwen import DEADLINE_HEADER, REASONING_EFFORTS
from ..run_artifacts import artifact_environment, console_log
from .duck.adapter import load_duck
from .duck.install import REVISION, UPSTREAM, runner_installed
from .duck.vllm_backend import serve_vllm, vllm_post
from .evaluate_arc3 import (
    DIFFICULTIES, REPO, load_difficulties, new_run, print_game, select_games,
    show_summary, summarize_games, write_json,
)

PROMPT = (
    "Play the interactive game, discover its rules, and complete all levels using as few "
    "actions as possible. Look at the latest grid and recent actions. Reply with one JSON "
    'object, for example {"action":"UP"}. For MOUSE, include integer row and col from 0 to '
    '63 with origin at the top left, for example {"action":"MOUSE","row":12,"col":34}.'
)
GENERATION_TIMEOUT = 230
REQUEST_TIMEOUT = 240
GAME_TIMEOUT = 2 * 60 * 60
PREFLIGHT_TIMEOUT = 600
DEFAULT_REASONING_EFFORT = "medium"


@dataclass(frozen=True)
class Condition:
    model: str
    harness: str
    reasoning_effort: str
    backend: str = "transformers"


class InferenceFailure(RuntimeError):
    """A decision could not be completed within the inference policy."""

    def __init__(self, message, *, budget_limited=False):
        super().__init__(message)
        self.budget_limited = budget_limited


def decode_move(content, available):
    if not isinstance(content, str):
        raise ValueError("Expected a JSON object containing one action")
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    action = json.loads(text)
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise ValueError("Expected an action name")
    name = action["action"].strip().upper()
    if name not in available:
        raise ValueError("Action is unavailable")
    move = {"action": name}
    if name == "MOUSE":
        for axis in ("row", "col"):
            value = action.get(axis)
            if type(value) is not int or not 0 <= value <= 63:
                raise ValueError("MOUSE row and col must be integers in [0, 63]")
            move[axis] = value
    return move


class ModelClient:
    """Build and account for requests shared by both policies."""

    def __init__(self, endpoint, config, *, deadline=False, max_output_tokens=None,
                 reasoning_effort=DEFAULT_REASONING_EFFORT, generation_timeout=GENERATION_TIMEOUT,
                 request_timeout=REQUEST_TIMEOUT, model="base", post=None):
        self.url = endpoint + "/chat/completions"
        self.model = model
        self.post = post
        self.config = config
        self.deadline = deadline
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.generation_timeout = generation_timeout
        self.request_timeout = request_timeout
        self.model_calls = 0
        self.generated_tokens = 0
        self.inference_seconds = 0.0

    def chat_payload(self, messages, *, tools=None, tool_choice=None):
        from inference.utils.openai_compat import build_chat_payload

        settings = self.config["analyzer"]
        payload = build_chat_payload(
            provider=self.config["shared"]["provider"], model=self.model, messages=messages,
            max_tokens=self.max_output_tokens or settings["max_output"] or None, temperature=settings["temperature"],
            top_p=settings["top_p"], top_k=settings["top_k"], thinking=settings["thinking"],
            tools=tools, tool_choice=tool_choice,
        )
        payload["chat_template_kwargs"].update(self.config["server"]["default_chat_template_kwargs"])
        payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def deadline_completion(self, payload, record, request_timeout_seconds=None, *, extract=None):
        import requests

        decision_end = None if request_timeout_seconds is None else time.monotonic() + request_timeout_seconds
        response_margin = self.request_timeout - self.generation_timeout
        for attempt in (1, 2):
            started = time.monotonic()
            timeout = self.request_timeout
            if decision_end is not None:
                timeout = min(timeout, max(0.0, decision_end - started))
            if timeout <= 0:
                raise InferenceFailure("decision_budget_expired: no time left for a transport retry",
                                       budget_limited=True)
            generation_seconds = min(self.generation_timeout, timeout - min(response_margin, timeout / 2))
            deadline = started + generation_seconds
            tokens = 0
            reason = "transport_error"
            transport_error = None
            self.model_calls += 1
            try:
                post = self.post or requests.post
                response = post(self.url, json=payload, timeout=timeout,
                                headers={DEADLINE_HEADER: str(deadline)})
                reason = "invalid_response"
                reply = response.json()
                tokens = reply.get("usage", {}).get("completion_tokens", 0)
                self.generated_tokens += tokens
                if not response.ok:
                    error = reply.get("error", {})
                    reason = error.get("code") or f"http_{response.status_code}"
                    raise InferenceFailure(
                        f"{reason}: {error.get('message', response.reason)}",
                        budget_limited=reason == "generation_deadline" and generation_seconds < self.generation_timeout,
                    )
                choice = reply["choices"][0]
                reason = choice["finish_reason"]
                if extract is None:
                    return reply
                try:
                    return extract(choice["message"].get("content"))
                except ValueError as error:
                    reason = f"invalid_action ({reason})"
                    raise InferenceFailure(f"{reason}: {error}") from error
            except (requests.ConnectionError, requests.Timeout) as error:
                transport_error = str(error)
                if attempt == 2:
                    raise InferenceFailure(f"{reason}: two attempts failed: {error}") from error
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise InferenceFailure(f"{reason}: malformed inference response: {error}") from error
            finally:
                elapsed = time.monotonic() - started
                self.inference_seconds += elapsed
                event = {"call": self.model_calls, "attempt": attempt, "elapsed_seconds": elapsed,
                         "generated_tokens": tokens, "stop_reason": reason}
                if transport_error is not None:
                    event["detail"] = transport_error
                record(event)

    def completion(self, messages, tools, timeout, *, record):
        from inference.agent.tool_agent import _request_tool_choice

        payload = self.chat_payload(messages, tools=tools, tool_choice=_request_tool_choice(tools))
        if self.deadline:
            return self.deadline_completion(payload, record, timeout)
        return self.legacy_completion(payload, timeout)

    def legacy_completion(self, payload, request_timeout_seconds):
        import requests

        timeout = self.request_timeout
        if request_timeout_seconds is not None:
            timeout = min(timeout, request_timeout_seconds)
        self.model_calls += 1
        started = time.monotonic()
        try:
            post = self.post or requests.post
            response = post(self.url, json=payload, timeout=timeout)
            if response.status_code in {429, 502, 503, 504}:
                response.raise_for_status()
        finally:
            self.inference_seconds += time.monotonic() - started
        if not response.ok:
            raise InferenceFailure(f"http_{response.status_code}: {response.text}")
        reply = response.json()
        self.generated_tokens += reply["usage"]["completion_tokens"]
        return reply


class DirectAnalyzer(ModelClient):
    """Adapt one text reply to the native runner's action callback."""

    model_actions = 0
    previous_error = ""

    def make_request(self, frame, history, action_num, valid_actions):
        from inference.agent.action_names import to_engine_action, to_model_action
        from inference.agent.vision_context import frame_to_png_data_url

        available = [to_model_action(name) for name in valid_actions if to_engine_action(name)]
        recent = ", ".join(entry.action for entry in history[-16:] if entry.action) or "None"
        instructions = (
            f"Level {frame.level}, action count {action_num}. Available actions: {', '.join(available)}."
            f"\nRecent actions: {recent}."
        )
        if self.previous_error:
            instructions += f"\nPrevious reply was rejected: {self.previous_error}."
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": instructions},
                {"type": "image_url", "image_url": {
                    "url": frame_to_png_data_url(frame, upscale=self.config["multimodal"]["upscale"]),
                }},
            ]},
        ]
        return self.chat_payload(messages), available

    def deadline_move(self, payload, available, record, request_timeout_seconds=None):
        return self.deadline_completion(
            payload, record, request_timeout_seconds,
            extract=lambda content: (decode_move(content, available), content),
        )

    def analyze(self, state_path, action_num, *, valid_actions, step_env, transcript_path,
                analysis_step, request_timeout_seconds, should_stop):
        import requests
        from inference.agent.runtime_state import load_runtime_state
        from inference.agent.tool_agent import AnalyzerTurnResult

        if should_stop():
            return AnalyzerTurnResult(step_executed=False)
        frame, history = load_runtime_state(state_path)
        payload, available = self.make_request(frame, history, action_num, valid_actions)
        if self.deadline:
            def record(event):
                with transcript_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"Decision {analysis_step} inference: {json.dumps(event)}\n")

            move, content = self.deadline_move(payload, available, record, request_timeout_seconds)
            if should_stop():
                return AnalyzerTurnResult(step_executed=False)
            self.model_actions += 1
            result = step_env(move)
            with transcript_path.open("a", encoding="utf-8") as stream:
                stream.write(f"Decision {analysis_step}: {content!r}\n")
            if not result["executed"] and not should_stop():
                raise InferenceFailure(f"action_rejected: {result.get('error', 'Action was rejected')}")
            return AnalyzerTurnResult(step_executed=result["executed"])

        try:
            reply = self.legacy_completion(payload, request_timeout_seconds)
        except requests.RequestException as error:
            with transcript_path.open("a", encoding="utf-8") as stream:
                stream.write(f"Decision {analysis_step}: {error}\n")
            return AnalyzerTurnResult(step_executed=False, retryable_failure=True)
        message = reply["choices"][0]["message"]
        executed = False
        content = message.get("content")
        try:
            move = decode_move(content, available)
            if not should_stop():
                self.model_actions += 1
                result = step_env(move)
                executed = result["executed"]
                self.previous_error = "" if executed else result.get("error", "Action was rejected")
        except ValueError as error:
            self.previous_error = str(error)
        with transcript_path.open("a", encoding="utf-8") as stream:
            stream.write(f"Decision {analysis_step}: {content!r}\n")
            if self.previous_error:
                stream.write(f"Rejected: {self.previous_error}\n")
        return AnalyzerTurnResult(step_executed=executed)


def game_result(run, pass_number, title, analyzer, condition, level_retries=None):
    resets = sum(entry.action.id.name == "RESET" for entry in run.history)
    moves = len(run.history) - resets
    status = {"won": "WIN", "crashed": "error"}.get(run.state, run.state)
    score = run.final_score
    if not run.base_actions_per_level or status in {"error", "cancelled"}:
        score = None
    return {
        "model": condition.model, "harness": condition.harness,
        "backend": condition.backend,
        "reasoning_effort": condition.reasoning_effort, "pass": pass_number,
        "game_id": run.game_id, "game_name": title or run.game_id,
        "rhae_percent": score, "status": status,
        "levels_completed": run.levels_completed, "levels_total": run.number_of_levels,
        "recovery_resets": resets, "accepted_moves": moves,
        "level_retries": sum((level_retries or {}).values()),
        "model_actions": analyzer.model_actions if analyzer else 0,
        "model_calls": analyzer.model_calls if analyzer else 0,
        "completion_tokens": analyzer.generated_tokens if analyzer else 0,
        "inference_seconds": analyzer.inference_seconds if analyzer else 0.0,
        "seconds": run.final_wallclock_seconds if run.final_score is not None else None,
        "error": run.solver_note if status == "error" else None,
    }


def export_summaries(run_dir, console, summaries):
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    console.print("Scored counts game passes. Mean RHAE uses Tufa scores for scored passes. "
                  "Gameplay timing begins when the live game opens.")
    show_summary(console, summaries, score_key="tufa_mean_rhae_percent", score_label="Tufa RHAE")


def report_summary(run_dir, console, results, total_passes, condition, run_error=None):
    summary = summarize_games(results, condition.model, condition.harness, total_passes, run_error)
    summary["reasoning_effort"] = condition.reasoning_effort
    summary["backend"] = condition.backend
    summary["level_retries"] = sum(row["level_retries"] for row in results)
    scores = [row["rhae_percent"] for row in results if row["rhae_percent"] is not None]
    summary["tufa_mean_rhae_percent"] = mean(scores) if scores else None
    if run_error:
        console.print(f"{condition.model} [{condition.harness}, {condition.reasoning_effort}] · {run_error}",
                      markup=False, highlight=False)
    export_summaries(run_dir, console, [summary])
    return summary


def make_arcade_spec(args, run_dir):
    from arc_agi import OperationMode
    from taaf.game_api import ArcadeSpec

    if args.online:
        return ArcadeSpec(
            operation_mode=OperationMode.ONLINE, arc_base_url=args.arc_base_url,
            environments_dir=str(run_dir / "environments"),
        )
    return ArcadeSpec(environments_dir=str(args.environments_dir))


def check_game_availability(args, selected, eligible, run_dir):
    from arc_agi import Arcade
    from requests import HTTPError

    spec = make_arcade_spec(args, run_dir)
    arcade = Arcade(operation_mode=spec.operation_mode, arc_base_url=spec.arc_base_url,
                    environments_dir=spec.environments_dir, recordings_dir=str(run_dir / "recordings"))
    reserve = sorted(set(eligible) - set(selected))
    random.Random(args.seed).shuffle(reserve)
    games = {}
    checks = []
    for original in selected:
        candidate = original
        while candidate is not None:
            card_id = arcade.create_scorecard()
            if not card_id:
                raise RuntimeError("ARC did not open a scorecard for the game availability check")
            try:
                try:
                    game = arcade.make(candidate, scorecard_id=card_id)
                except HTTPError as error:
                    if error.response is None or error.response.status_code != 404:
                        raise
                    game = None
                frame = game.observation_space if game is not None else None
                info = game.environment_info if game is not None else None
            finally:
                try:
                    arcade.close_scorecard(card_id)
                except HTTPError as error:
                    if error.response is None or error.response.status_code != 404:
                        raise
                    print(f"Warning: ARC could not close the availability scorecard for {candidate}", flush=True)
            found = bool(frame is not None and frame.frame and frame.win_levels > 0
                         and info is not None and info.game_id == candidate)
            checks.append({"requested": original, "candidate": candidate, "available": found})
            if found:
                games[candidate] = (frame, list(info.baseline_actions) if info.baseline_actions else None,
                                    info.title)
                if candidate != original:
                    print(f"Replaced unavailable game {original} with {candidate}", flush=True)
                break
            print(f"Warning: ARC could not open {candidate}", flush=True)
            candidate = reserve.pop() if reserve else None
        if candidate is None:
            print(f"Warning: omitting {original}; no unselected eligible game is available", flush=True)
    write_json(run_dir / "game_availability.json", checks)
    return games


def policy_analyzer(args, config, endpoint, condition, *, record=None):
    client_type = DirectAnalyzer if condition.harness == "none" else ModelClient
    client = client_type(endpoint, config, deadline=args.deadline, max_output_tokens=args.max_output_tokens,
                         reasoning_effort=condition.reasoning_effort, generation_timeout=args.generation_timeout,
                         request_timeout=args.request_timeout, model=condition.model,
                         post=vllm_post if condition.backend == "vllm" else None)
    if condition.harness == "none":
        return client
    from .duck.native_policy import NativeDuckAnalyzer

    return NativeDuckAnalyzer(client, endpoint, record=record)


def preflight_duck(analyzer, frame, valid_actions, run_dir, ends_at):
    from inference.agent.action_names import to_engine_action, to_model_action
    from inference.agent.runtime_state import HistoryEntry, RUNTIME_STATE_FILENAME, write_runtime_state

    state_path = run_dir / "preflight_artifacts" / RUNTIME_STATE_FILENAME
    write_runtime_state(state_path, current_frame=frame, history=[HistoryEntry("", frame)])
    available = [to_model_action(name) for name in valid_actions if to_engine_action(name)]
    move = None

    def check_action(payload):
        nonlocal move
        if time.monotonic() >= ends_at:
            return {"executed": False, "error": "Preflight time expired", "valid_actions": []}
        try:
            move = decode_move(json.dumps(payload["actions"][0]), available)
        except (IndexError, ValueError) as error:
            return {"executed": False, "error": str(error), "valid_actions": available}
        return {"executed": True, "executed_count": 1, "done": True, "run_complete": True,
                "action_num": 1, "level": frame.level, "valid_actions": []}

    analyzer.analyze(
        state_path, 0, valid_actions=valid_actions, step_env=check_action,
        transcript_path=run_dir / "preflight_artifacts" / "transcript.txt", analysis_step=1,
        request_timeout_seconds=max(0.0, ends_at - time.monotonic()),
        should_stop=lambda: time.monotonic() >= ends_at,
    )
    if move is None:
        raise InferenceFailure("preflight_no_action: Duck tool policy proposed no playable action")
    return move


def run_preflight(args, config, game, run_dir, endpoint, condition):
    from arcengine import GameAction
    from inference.agent.runtime_state import Frame

    game_id = game.game_id
    report = {"game_id": game_id, "model": condition.model, "harness": condition.harness,
              "backend": condition.backend,
              "reasoning_effort": condition.reasoning_effort, "attempts": [], "status": "failed"}
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ends_at = started + args.preflight_timeout
    try:
        analyzer = policy_analyzer(args, config, endpoint, condition, record=report["attempts"].append)
        observation = game.current_state.raw
        grid = tuple(tuple(int(cell) for cell in row) for row in observation.frame[-1])
        frame = Frame(grid, 0, observation.levels_completed + 1)
        actions = [GameAction.from_id(action).name for action in observation.available_actions
                   if action != GameAction.RESET.value]
        if condition.harness == "duck":
            move = preflight_duck(analyzer, frame, actions, run_dir, ends_at)
        else:
            payload, available = analyzer.make_request(frame, [], 0, actions)
            move, _ = analyzer.deadline_move(payload, available, report["attempts"].append,
                                             max(0.0, ends_at - time.monotonic()))
        if time.monotonic() >= ends_at:
            raise InferenceFailure("preflight_time_limit: did not finish within the total preflight budget")
        report.update(status="passed", action=move)
    except Exception as error:
        report["error"] = str(error)
        raise InferenceFailure(f"Preflight failed for {game_id}: {error}") from error
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(run_dir / "preflight.json", report)
        for event in report["attempts"]:
            print(f"Preflight {game_id} call {event['call']} attempt {event['attempt']} · "
                  f"{event['elapsed_seconds']:.2f}s · "
                  f"{event['generated_tokens']} generated tokens · {event['stop_reason']}", flush=True)
        print(f"Preflight {game_id} {report['status']} · {report['elapsed_seconds']:.2f}s", flush=True)


def run_benchmark(config, args, games, run_dir, endpoint, condition):
    from rich.console import Console
    from taaf.benchmark import Benchmark
    from taaf.deploy_inline import InlineTarget

    from .duck.level_budget import LevelBudgetSolver
    from .duck.reporting import ReportingGame

    console = Console(file=sys.stdout, width=168, force_terminal=False)
    results = {}
    result_lock = Lock()
    preflight_locks = {game_id: Lock() for game_id in games}
    preflight_errors = {}

    def record_result(index, run, title=None, analyzer=None, retries=None):
        result = game_result(run, index // len(games) + 1, title, analyzer, condition, retries)
        with result_lock:
            if index not in results:
                results[index] = result
                print_game(console, result)

    def make_analyzer(game, index):
        analyzer = None
        game.on_finished = lambda run: record_result(index, run, game.title, analyzer, game.level_retries)
        game.begin_timing()
        if args.deadline:
            game_id = game.game_id
            with preflight_locks[game_id]:
                if game_id not in preflight_errors:
                    try:
                        run_preflight(args, config, game, run_dir / "preflight" / game_id, endpoint, condition)
                    except InferenceFailure as error:
                        preflight_errors[game_id] = str(error)
                    else:
                        preflight_errors[game_id] = None
            if error := preflight_errors[game_id]:
                raise InferenceFailure(error)
        game.open_for_play()
        analyzer = policy_analyzer(args, config, endpoint, condition)
        return analyzer

    arcade = make_arcade_spec(args, run_dir)
    environment = config["environment"]
    policy = "direct" if condition.harness == "none" else "duck"
    label = f"qwen-{condition.model}-{policy}-{condition.reasoning_effort}"
    if condition.backend != "transformers":
        label += f"-{condition.backend}"
    solver = LevelBudgetSolver(
        label=label, model=condition.model,
        analyzer_timeout=args.request_timeout,
        max_actions_per_game=environment["max_steps"],
        level_timeout=args.game_timeout, bounded_recovery=args.bounded_recovery,
        concurrency=args.concurrent_jobs,
        save_request_logs=config["analyzer"]["save_request_logs"],
        kaggle_enable_vllm=False,
        analyzer_factory=make_analyzer,
    )
    benchmark = Benchmark(
        label=label, games=[
            ReportingGame(env_name=game_id, arcade_spec=arcade, opening_frame=frame,
                          human_actions=baselines, title=title)
            for game_id, (frame, baselines, title) in games.items()
        ],
        solver=solver, n_passes=args.n_passes, job_dir=run_dir,
    )
    # Match the official inline CLI's per-wave budget and teardown buffer.
    waves = math.ceil(len(games) * args.n_passes / args.concurrent_jobs)
    max_levels = max(frame.win_levels for frame, _, _ in games.values())
    pass_budget = args.game_timeout * max_levels * (1 + args.bounded_recovery)
    if args.deadline:
        pass_budget += args.preflight_timeout
    target = InlineTarget(max_runtime_s=pass_budget * waves + 10 * 60)
    run_error = None
    try:
        asyncio.run(benchmark.deploy(target))
    except KeyboardInterrupt:
        run_error = "KeyboardInterrupt"
        raise
    except Exception as error:
        run_error = str(error) or type(error).__name__
    finally:
        for index, game in enumerate(benchmark.game_runs):
            if index not in results:
                record_result(index, game)
        summary = report_summary(run_dir, console, list(results.values()), len(games) * args.n_passes,
                                 condition, run_error)
    return summary


def run_policy(args, config, games, policy_dir, endpoint, condition):
    from rich.console import Console

    reset_levels = os.environ.get("ONLY_RESET_LEVELS")
    try:
        with load_duck() if condition.harness == "duck" else nullcontext():
            return run_benchmark(config, args, games, policy_dir, endpoint, condition)
    except Exception as error:
        console = Console(file=sys.stdout, width=168, force_terminal=False)
        return report_summary(policy_dir, console, [], len(games) * args.n_passes, condition, str(error))
    finally:
        if reset_levels is None:
            os.environ.pop("ONLY_RESET_LEVELS", None)
        else:
            os.environ["ONLY_RESET_LEVELS"] = reset_levels


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer")
    return number


def print_cli_options(args, *, file):
    lines = ["CLI options"]
    for name, value in vars(args).items():
        flag = "--" + name.replace("_", "-")
        lines.append(f"  {flag} = {json.dumps(value, default=str)}")
    print("\n".join(lines), file=file, flush=True)


def resolve_base_models(specs):
    configured = read_json(MODEL_CONFIG).get("base_models", {})
    models = {}
    for spec in specs:
        alias, separator, location = spec.partition("=")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", alias):
            raise ValueError(f"Invalid base model alias {alias!r}")
        if alias == "base":
            if separator:
                raise ValueError("'base' is reserved for the existing frozen Qwen checkpoint")
            config, snapshot, _ = load_selection(SimpleNamespace(model=["base"]))
            details = {"model": config["base_model"], "model_revision": config["base_revision"]}
        else:
            if not separator:
                location = configured.get(alias)
            if not location:
                raise ValueError(f"Unknown base model {alias!r}. Use {alias}=/path/to/local/snapshot "
                                 f"or configure the alias in {MODEL_CONFIG}")
            base_dir = Path.cwd() if separator else MODEL_CONFIG.parent
            snapshot = (base_dir / Path(location).expanduser()).resolve()
            model_config = read_json(snapshot / "config.json")
            quantization = model_config.get("quantization_config") or {}
            if (not model_config.get("vision_config") or quantization.get("bnb_4bit_quant_type") != "nf4"
                    or not quantization.get("bnb_4bit_use_double_quant")
                    or not (quantization.get("load_in_4bit") or quantization.get("_load_in_4bit"))):
                raise ValueError(f"{alias!r} must point to a multimodal checkpoint with nested NF4 weights")
            if not any((snapshot / name).is_file() for name in (
                "model.safetensors", "model.safetensors.index.json",
            )):
                raise ValueError(f"No local safetensors weights found in {snapshot}")
            details = {"model": alias, "model_revision": snapshot.name if snapshot.parent.name == "snapshots" else None}
        details["snapshot"] = str(snapshot)
        if alias in models and models[alias] != details:
            raise ValueError(f"Conflicting checkpoints for base model alias {alias!r}")
        models[alias] = details
    return models


def main():
    grpo_python = Path.home() / "miniconda/envs/grpo/bin/python"
    vllm_python = Path.home() / "miniconda/envs/vllm/bin/python"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", nargs="+", choices=("transformers", "vllm"),
                        help="Serving backends to compare (default: transformers).")
    parser.add_argument("--harness", action="append", nargs="+", choices=("none", "duck"),
                        help="Policies to run (default: none). Pass both as --harness none duck.")
    parser.add_argument("--game", action="extend", nargs="+", metavar="GAME",
                        help="Select games by short or versioned ID.")
    parser.add_argument("--n-tasks", type=positive_int, metavar="N",
                        help="Sample N distinct games, stratified across eligible difficulty labels.")
    parser.add_argument("--difficulty", action="append", nargs="+", choices=DIFFICULTIES,
                        help="Limit evaluation to games with any of these difficulty labels.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for task sampling (default: 0).")
    parser.add_argument("--list-games", action="store_true", help="Print the selected versioned game IDs.")
    parser.add_argument("--n-passes", type=positive_int, help="Override Tufa's configured passes per game.")
    parser.add_argument("--concurrent-jobs", type=positive_int, help="Override Tufa's configured worker count.")
    parser.add_argument("--reasoning-effort", action="append", nargs="+", choices=(*REASONING_EFFORTS, "easy"),
                        help=f"Qwen reasoning levels (default: {DEFAULT_REASONING_EFFORT}). 'easy' is an alias for 'low'.")
    parser.add_argument("--deadline", action="store_true",
                        help="Require a preflight action and enforce the local generation deadline.")
    parser.add_argument("--generation-timeout", type=positive_int, default=GENERATION_TIMEOUT, metavar="SECONDS",
                        help="Local server deadline for queueing and generation with --deadline (default: %(default)s).")
    parser.add_argument("--request-timeout", type=positive_int, default=REQUEST_TIMEOUT, metavar="SECONDS",
                        help="Client timeout per inference request (default: %(default)s).")
    parser.add_argument("--game-timeout", type=positive_int, default=GAME_TIMEOUT, metavar="SECONDS",
                        help="Wall-clock budget per level attempt (default: %(default)s).")
    parser.add_argument("--bounded-recovery", type=nonnegative_int, default=2, metavar="N",
                        help="Maximum retries per level after timeout or game over (default: %(default)s).")
    parser.add_argument("--preflight-timeout", type=positive_int, default=PREFLIGHT_TIMEOUT, metavar="SECONDS",
                        help="Total preflight budget per game and condition with --deadline (default: %(default)s).")
    parser.add_argument("--max-output-tokens", type=positive_int, metavar="N",
                        help="Optional local output cap for --deadline mode.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--environments-dir", type=Path, help="Use a local copy of Tufa's ARC environments.")
    source.add_argument("--online", action="store_true", help="Use the hosted ARC API instead of the offline bundle.")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env", help="ARC_API_KEY fallback for hosted games.")
    parser.add_argument("--arc-base-url", default=os.environ.get("ARC_BASE_URL") or "https://arcprize.org")
    parser.add_argument("--model", action="append", nargs="+", metavar="ALIAS[=SNAPSHOT]",
                        help="Local frozen base models to compare (default: base). Use configured aliases or ALIAS=PATH.")
    parser.add_argument("--model-python", default=str(grpo_python) if grpo_python.is_file() else sys.executable)
    parser.add_argument("--vllm-python", default=str(vllm_python),
                        help="Python in the vllm Conda environment.")
    args = parser.parse_args()
    for name, default in (("backend", ["transformers"]), ("harness", ["none"]),
                          ("reasoning_effort", [DEFAULT_REASONING_EFFORT]),
                          ("difficulty", None), ("model", ["base"])):
        groups = getattr(args, name)
        if groups and len(groups) != 1:
            parser.error(f"Pass --{name.replace('_', '-')} once followed by its list of values")
        setattr(args, name, list(dict.fromkeys(groups[0])) if groups else default)
    args.reasoning_effort = list(dict.fromkeys("low" if level == "easy" else level for level in args.reasoning_effort))
    if args.deadline and args.generation_timeout >= args.request_timeout:
        parser.error("--generation-timeout must be shorter than --request-timeout when using --deadline")
    if args.max_output_tokens is not None and not args.deadline:
        parser.error("--max-output-tokens requires --deadline")
    if not runner_installed():
        parser.error("Install the runner with python -m jm_test.graph_world.grpo_mosp.experiment_arc3.duck.install --runner")
    config = json.loads((UPSTREAM / "configs/inference.json").read_text(encoding="utf-8"))
    environment = config["environment"]
    args.n_passes = args.n_passes or environment["n_passes"]
    args.concurrent_jobs = args.concurrent_jobs or environment["concurrent_jobs"]
    sys.path[:0] = [str(UPSTREAM), str(UPSTREAM / "dependencies")]
    from inference.framework.kaggle import DUCK_HARNESS_PUBLIC_GAME_IDS

    games = {game.split("-")[0]: game for game in DUCK_HARNESS_PUBLIC_GAME_IDS}
    game_ids = list(DUCK_HARNESS_PUBLIC_GAME_IDS)
    if args.game:
        game_ids = list(dict.fromkeys(games.get(game.lower(), game.lower()) for game in args.game))
        unknown = set(game_ids) - set(DUCK_HARNESS_PUBLIC_GAME_IDS)
        if unknown:
            parser.error(f"Unknown game IDs: {', '.join(sorted(unknown))}. See --list-games.")
    # Match the other ARC evaluator's sampling order while preserving Tufa's
    # catalog order for runs that do not sample.
    candidates = sorted(game_ids) if args.n_tasks is not None else game_ids
    catalog = [SimpleNamespace(game_id=game_id) for game_id in candidates]
    labels = load_difficulties([SimpleNamespace(game_id=game) for game in DUCK_HARNESS_PUBLIC_GAME_IDS])
    eligible = [game for game in DUCK_HARNESS_PUBLIC_GAME_IDS
                if not args.difficulty or labels[game] in args.difficulty]
    try:
        game_ids = [game.game_id for game in select_games(catalog, args, labels)]
    except ValueError as error:
        parser.error(str(error))
    if args.environments_dir:
        args.environments_dir = args.environments_dir.expanduser().resolve()
    if args.list_games:
        print_cli_options(args, file=sys.stderr)
        print("\n".join(game_ids))
        return
    if args.environments_dir:
        if not args.environments_dir.is_dir():
            parser.error(f"Missing offline environment directory: {args.environments_dir}")
    elif not args.online:
        parser.error("Provide Tufa's offline bundle with --environments-dir, or use --online for the hosted ARC API.")
    elif not os.environ.get("ARC_API_KEY"):
        from dotenv import dotenv_values

        key = dotenv_values(args.env_file).get("ARC_API_KEY")
        if not key:
            parser.error("Set ARC_API_KEY, pass --env-file, or provide an offline --environments-dir.")
        os.environ["ARC_API_KEY"] = key
    if os.environ.get("OPERATION_MODE"):
        parser.error("Unset OPERATION_MODE so this evaluator can select the requested ARC environment mode.")

    try:
        models = resolve_base_models(args.model)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    context = config["shared"]["context_window"]
    if args.max_output_tokens is not None and args.max_output_tokens > context:
        parser.error(f"--max-output-tokens must fit the {context}-token context")
    run_dir = new_run()
    model_dirs = {alias: run_dir / "models" / alias if len(models) > 1 else run_dir for alias in models}
    backend_dirs = {
        (alias, backend): path / backend if len(args.backend) > 1 else path
        for alias, path in model_dirs.items() for backend in args.backend
    }
    condition_dirs = {}
    for alias, backend, harness, effort in product(models, args.backend, args.harness, args.reasoning_effort):
        path = backend_dirs[alias, backend]
        if len(args.harness) > 1:
            path /= harness
        if len(args.reasoning_effort) > 1:
            path /= effort
        path.mkdir(parents=True, exist_ok=True)
        condition_dirs[Condition(alias, harness, effort, backend)] = path
    multiple_conditions = len(condition_dirs) > 1
    with artifact_environment(run_dir):
        print_cli_options(args, file=sys.stdout)
        games = check_game_availability(args, game_ids, eligible, run_dir)
        if not games:
            raise RuntimeError(f"No selected or replacement games could be opened. See {run_dir / 'log.txt'}")
        write_json(run_dir / "manifest.json", {
            "runner_revision": REVISION, "runner": "taaf.deploy_inline.InlineTarget",
            "policy": "comparison" if multiple_conditions else {"none": "direct", "duck": "duck"}[args.harness[0]],
            "harnesses": args.harness, "models": models,
            **(next(iter(models.values())) if len(models) == 1 else {}),
            "backend": (args.backend[0] + "-bitsandbytes") if len(args.backend) == 1 else "comparison",
            "backends": {backend: {"python": args.vllm_python if backend == "vllm" else args.model_python}
                         for backend in args.backend},
            "conditions": [
                {"model": condition.model, "harness": condition.harness, "reasoning_effort": condition.reasoning_effort,
                 "backend": condition.backend,
                 "directory": str(path.relative_to(run_dir))}
                for condition, path in condition_dirs.items()
            ],
            "arcade": str(args.environments_dir) if args.environments_dir else args.arc_base_url,
            "games": list(games), "n_passes": args.n_passes, "concurrent_jobs": args.concurrent_jobs,
            "selection": {
                "n_tasks": args.n_tasks, "difficulty": args.difficulty, "seed": args.seed,
                "requested_games": game_ids,
                "difficulty_labels": {game_id: labels[game_id] for game_id in games},
            },
            "upstream_config": config,
            "reasoning_effort": args.reasoning_effort[0] if len(args.reasoning_effort) == 1 else None,
            "reasoning_efforts": args.reasoning_effort,
            "request_timeout_seconds": args.request_timeout,
            "level_timeout_seconds": args.game_timeout,
            "bounded_recovery_per_level": args.bounded_recovery,
            "deadline": {
                "enabled": args.deadline,
                "server_timeout_seconds": args.generation_timeout if args.deadline else None,
                "preflight_timeout_seconds": args.preflight_timeout if args.deadline else None,
                "transport_retries": 1 if args.deadline else None,
                "local_output_cap_tokens": args.max_output_tokens,
            },
        })
        print(f"Tufa runner · {len(games)} games × {args.n_passes} passes "
              f"× {len(condition_dirs)} conditions · {run_dir}", flush=True)
        if args.concurrent_jobs > 1 and "transformers" in args.backend:
            print("The local model server processes one generation at a time. Queued requests count toward timeout.")
        server_args = [
            "--context-limit", str(context), "--completion-limit", str(context),
            "--default-completion-to-context",
        ]
        if args.deadline:
            server_args.extend(["--generation-timeout", str(args.generation_timeout)])
        if args.max_output_tokens is not None:
            server_args.append("--fit-completion-to-context")
        from rich.console import Console

        summaries = []
        for (alias, backend), model_dir in backend_dirs.items():
            snapshot = Path(models[alias]["snapshot"])
            log = model_dir / "model_server.log"
            with ExitStack() as stack:
                startup_error = None
                try:
                    if backend == "vllm":
                        server = serve_vllm(args.vllm_python, snapshot, alias, log, context_limit=context,
                                            concurrent_jobs=args.concurrent_jobs)
                    else:
                        server = serve_checkpoint(args.model_python, snapshot, None, alias, log,
                                                  model_dir / "server.ready", server_args=server_args)
                    endpoint = stack.enter_context(server)
                except Exception as error:
                    startup_error = str(error) or type(error).__name__
                for condition, policy_dir in condition_dirs.items():
                    if (condition.model, condition.backend) != (alias, backend):
                        continue
                    with console_log(policy_dir / "log.txt") if multiple_conditions else nullcontext():
                        print(f"Policy · {alias} [{condition.harness}, {condition.reasoning_effort}, {backend}] "
                              f"· {policy_dir}", flush=True)
                        if startup_error is None:
                            summary = run_policy(args, config, games, policy_dir, endpoint, condition)
                        else:
                            console = Console(file=sys.stdout, width=168, force_terminal=False)
                            summary = report_summary(policy_dir, console, [], len(games) * args.n_passes,
                                                     condition, startup_error)
                        summaries.append(summary)
        if multiple_conditions:
            console = Console(file=sys.stdout, width=168, force_terminal=False)
            export_summaries(run_dir, console, summaries)
        if any(summary["error"] is not None or summary["errors"] for summary in summaries):
            raise RuntimeError(f"Some policy evaluations failed. See {run_dir / 'log.txt'}")


if __name__ == "__main__":
    main()
