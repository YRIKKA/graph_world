"""Evaluate the shared Qwen checkpoints on the ARC-AGI-3 game catalog."""

import argparse
import base64
import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict, deque
from contextlib import ExitStack, nullcontext, redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean, pvariance
from urllib.request import Request, urlopen

if __name__ == "__main__":
    sys.dont_write_bytecode = True

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "grpo_mosp.experiment_arc3"

from ..experiment_downstream_tasks.checkpoints import (
    add_model_options,
    load_selection,
    model_paths,
    serve_checkpoint,
)
from ..run_artifacts import artifact_environment

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DIFFICULTIES = ("easy", "medium", "hard", "extra hard")
CONTEXT_TOKENS = 262_144
COMPLETION_TOKENS = 131_072
DEFAULT_RECOVERY_RESETS = 2
HUMAN_ACTION_MULTIPLIER = 5
MAX_REPLY_ATTEMPTS = 4
DUCK_GAME_TIMEOUT = 2700
PROMPT = (
    "Play the interactive game to discover its rules and complete all levels. "
    "Use as few actions as you can. Inspect the latest image and learn from recent outcomes. "
    "Reply with exactly one JSON object with an action from the available list and an optional "
    'short note to remember your observation or plan. Example {"action":"ACTION1","note":"..."}. '
    "ACTION1–4 often mean up/down/left/right. ACTION5 interacts, and ACTION7 undoes, when "
    "available. For ACTION6 include integer x and y in the original 0–63 grid, with origin "
    'at the top left, e.g. {"action":"ACTION6","x":12,"y":34}. '
    "The image scales each grid cell to a block of pixels."
)
TASK_FIELDS = (
    "model", "harness", "game_id", "game_name", "status", "final_state", "rhae_percent", "levels_completed",
    "levels_total", "official_actions", "accepted_moves", "model_actions", "invalid_responses",
    "recovery_resets", "attempt_actions", "level_actions", "level_action_limit",
    "model_calls", "completion_tokens", "inference_seconds", "seconds", "error",
)
ACTION_FIELDS = (
    "model", "harness", "game_id", "game_name", "scorecard_id", "event", "source", "action_index",
    "model_action", "model_actions", "attempt", "level", "action", "x", "y", "accepted", "status",
    "attempt_actions", "attempt_action_limit", "level_actions", "level_action_limit",
    "attempt_actions_after", "level_actions_after", "refunded_level_actions", "reset_reason",
    "human_actions", "accepted_moves", "recovery_resets", "state_before", "state_after",
    "levels_before", "levels_after", "levels_completed", "levels_total", "level_cleared",
    "attempt_outcome", "game_guid_before", "game_guid_after",
    "available_actions", "model_calls", "model_replies", "invalid_responses", "finish_reasons", "reply_errors",
    "completion_tokens", "inference_seconds", "action_seconds", "elapsed_seconds", "official_actions",
    "rhae_percent", "error",
)


def new_run():
    root = HERE / "runs"
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        run = root / datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        try:
            run.mkdir()
            return run
        except FileExistsError:
            time.sleep(1)
    raise RuntimeError("Could not reserve an ARC-AGI-3 timestamp")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def make_arcade(args, root, logger):
    from arc_agi import Arcade, OperationMode
    from dotenv import dotenv_values

    mode = OperationMode.OFFLINE if args.environments_dir else OperationMode.ONLINE
    api_key = ""
    if mode is OperationMode.ONLINE:
        api_key = os.environ.get("ARC_API_KEY") or dotenv_values(args.env_file).get("ARC_API_KEY")
        if not api_key:
            raise ValueError("Set ARC_API_KEY or point --env-file at an authenticated ARC configuration")
    arcade = Arcade(
        operation_mode=mode, arc_api_key=api_key or "",
        environments_dir=str(args.environments_dir or root / "environments"),
        recordings_dir=str(root / "recordings"), logger=logger,
    )
    if arcade.operation_mode is not mode:
        raise ValueError("Unset OPERATION_MODE to use this evaluator's selected mode")
    games = sorted(arcade.get_environments(), key=lambda item: item.game_id)
    if not games:
        raise RuntimeError("The ARC toolkit returned an empty game catalog")
    if len({game.game_id for game in games}) != len(games):
        raise RuntimeError("The ARC catalog contains duplicate versioned game IDs")
    if mode is OperationMode.OFFLINE and any(not game.baseline_actions for game in games):
        raise ValueError("The offline bundle is missing official human baseline actions")
    return arcade, games


def load_difficulties(games):
    with (HERE / "arc3_human_data_processed.csv").open(newline="", encoding="utf-8") as stream:
        labels = {row["environment"]: row["difficulty"] for row in csv.DictReader(stream)}
    return {game.game_id: labels.get(game.game_id.split("-", 1)[0]) for game in games}


def stratified_sample(games, n_tasks, rng, labels):
    groups = {}
    for game in games:
        groups.setdefault(labels[game.game_id], []).append(game)

    strata = list(groups)
    rng.shuffle(strata)  # Break quota ties reproducibly.
    cover_all = n_tasks >= len(strata)
    quotas = dict.fromkeys(strata, int(cover_all))
    limits = {label: len(groups[label]) if cover_all else 1 for label in strata}
    for _ in range(n_tasks - sum(quotas.values())):
        label = max(
            (label for label in strata if quotas[label] < limits[label]),
            key=lambda label: n_tasks * len(groups[label]) - quotas[label] * len(games),
        )
        quotas[label] += 1
    return [game for label in strata for game in rng.sample(groups[label], quotas[label])]


def select_games(games, args, labels):
    n_tasks = args.n_tasks
    difficulties = (args.difficulty,) if isinstance(args.difficulty, str) else args.difficulty
    if difficulties:
        games = [game for game in games if labels[game.game_id] in difficulties]
        if not games:
            raise ValueError(f"No games labeled {args.difficulty!r} are available in this catalog")
    if n_tasks is None:
        return games
    if not 1 <= n_tasks <= len(games):
        raise ValueError(f"--n-tasks must be between 1 and {len(games)} eligible games")
    rng = random.Random(args.seed)
    if not difficulties or len(difficulties) > 1:
        sample = stratified_sample(games, n_tasks, rng, labels)
    else:
        sample = rng.sample(games, n_tasks)
    return sorted(sample, key=lambda game: game.game_id)


def game_catalog(games, labels):
    return [
        {"id": game.game_id, "name": game.title or game.game_id, "difficulty": labels[game.game_id],
         "human_actions": game.baseline_actions}
        for game in games
    ]


def action_limits(game):
    baseline = game.baseline_actions
    if not baseline or any(type(actions) is not int or actions < 1 for actions in baseline):
        raise ValueError(f"Missing or invalid per-level human actions for {game.game_id}")
    return [HUMAN_ACTION_MULTIPLIER * actions for actions in baseline]


def task_directories(games):
    directories = {}
    used = set()
    for game in games:
        name = re.sub(r"[^\w.-]+", "_", game.title or game.game_id).strip(".")[:120] or "task"
        directory = name if name not in used else f"{name}~{len(directories) + 1}"
        used.add(directory)
        directories[game.game_id] = directory
    return directories


def plot_frame(frame, path):
    from arc_agi.rendering import frame_to_rgb_array
    from matplotlib import pyplot as plt

    plt.imsave(path, frame_to_rgb_array(0, frame, scale=6))


def save_observation(observation, index, frame_dir, image_dir):
    if (frame_dir is None and image_dir is None) or not observation.frame:
        return

    import numpy as np

    frames = np.asarray(observation.frame)
    if frame_dir is not None:
        np.savez_compressed(frame_dir / f"action_{index:05d}.npz", frames=frames)
    if image_dir is not None:
        for frame_index, frame in enumerate(frames):
            plot_frame(frame, image_dir / f"action_{index:05d}_frame_{frame_index:03d}.png")


def frame_url(observation, coordinates=False):
    from arc_agi.rendering import frame_to_rgb_array
    from PIL import Image, ImageDraw

    if not observation.frame:
        raise RuntimeError("ARC returned no image for an active game")
    frame = frame_to_rgb_array(0, observation.frame[-1], scale=6)
    image = Image.fromarray(frame)
    if coordinates:
        margin = 24
        grid = Image.new("RGB", (image.width + margin, image.height + margin), "white")
        grid.paste(image, (margin, margin))
        draw = ImageDraw.Draw(grid)
        for x in range(0, image.width // 6, 8):
            draw.text((margin + 6 * x, 1), str(x), fill="black")
            draw.line((margin + 6 * x, margin - 5, margin + 6 * x, margin - 1), fill="black")
        for y in range(0, image.height // 6, 8):
            draw.text((1, margin + 6 * y), str(y), fill="black")
            draw.line((margin - 5, margin + 6 * y, margin - 1, margin + 6 * y), fill="black")
        image = grid
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def describe_changes(previous, observation):
    import numpy as np

    before, after = np.asarray(previous.frame[-1]), np.asarray(observation.frame[-1])
    if before.shape != after.shape:
        return f"Grid shape changed from {before.shape} to {after.shape} (rows, columns)."
    changes = np.argwhere(before != after)
    examples = ", ".join(f"({x},{y}):{before[y, x]}→{after[y, x]}" for y, x in changes[:12])
    return f"{len(changes)} cells changed." + (f" Examples (x,y): {examples}." if examples else "")


def request_action(endpoint, model, observation, available, history, args, *, previous, notes, repeats, timeout=None):
    context = f"Levels completed: {observation.levels_completed}/{observation.win_levels}. "
    context += f"Available actions: {', '.join(available)}."
    if history:
        context += "\nRecent actions:\n" + "\n".join(history)
    prompt = PROMPT
    images = []
    if args.keep_notes:
        prompt += (
            ' Use the "note" field to maintain one compact scratchpad of observed rules, failed '
            "experiments, and your plan. When updating it, return the whole revised scratchpad "
            "in up to 1200 characters."
        )
        context += f"\nYour scratchpad: {notes or '(empty)'}"
        if previous is not None and previous.frame:
            context += "\nSince the previous frame: " + describe_changes(previous, observation)
            images.extend([
                {"type": "text", "text": "Previous frame"},
                {"type": "image_url", "image_url": {"url": frame_url(previous)}},
            ])
        context += "\nCurrent frame follows."
        if "ACTION6" in available:
            context += " The top and left margins mark game x/y coordinates at 8-cell intervals."
    if args.allow_thinking:
        prompt += " Choose a move that tests a specific hypothesis or advances your plan."
        if repeats:
            attempted = ", ".join(f"{action} ({count}×)" for action, count in repeats)
            context += (f"\nActions tried on this visual state: {attempted}. "
                        "If an action failed to help, try a different one.")
    images.append({"type": "image_url", "image_url": {
        "url": frame_url(observation, coordinates=args.keep_notes and "ACTION6" in available),
    }})
    request = {
        "model": model, "temperature": 0, "stream": False,
        "enable_thinking": args.allow_thinking,
        "max_tokens": COMPLETION_TOKENS,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "text", "text": context}, *images]},
        ],
    }
    payload = Request(endpoint + "/chat/completions", data=json.dumps(request).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(payload, timeout=args.timeout if timeout is None else timeout) as response:
        reply = json.load(response)
    choice = reply["choices"][0]
    return choice["message"]["content"], reply["usage"]["completion_tokens"], choice["finish_reason"]


def decode_action(content, available):
    if not isinstance(content, str):
        raise ValueError("Expected a JSON action")
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    choice = json.loads(content)
    if not isinstance(choice, dict) or not isinstance(choice.get("action"), str):
        raise ValueError("Expected an action name")
    action = choice["action"].upper()
    if action not in available:
        raise ValueError("Action is unavailable")
    data = {}
    if available[action].is_complex():
        for axis in ("x", "y"):
            value = choice.get(axis)
            if type(value) is not int or not 0 <= value <= 63:
                raise ValueError("Click coordinates must be integers in [0, 63]")
            data[axis] = value
    note = choice.get("note")
    return available[action], data, note.strip() if isinstance(note, str) else ""


def choose_action(endpoint, model, observation, available, history, args, event, result,
                  *, previous, notes, repeats, deadline=None):
    event["replies"] = []
    started = time.perf_counter()
    try:
        for _ in range(MAX_REPLY_ATTEMPTS):
            timeout = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ARC game wall-clock limit reached")
                timeout = min(args.timeout, remaining)
            result["model_calls"] += 1
            content, tokens, finish_reason = request_action(
                endpoint, model, observation, available, history, args,
                previous=previous, notes=notes, repeats=repeats, timeout=timeout,
            )
            reply = {"response": content, "completion_tokens": tokens, "finish_reason": finish_reason}
            event["replies"].append(reply)
            result["completion_tokens"] += tokens
            try:
                return decode_action(content, available)
            except ValueError as error:
                result["invalid_responses"] += 1
                reply["error"] = str(error)
                history.append(f"Invalid reply ({error}). Use one available action in JSON.")
        raise RuntimeError(f"No valid action after {MAX_REPLY_ATTEMPTS} model replies")
    finally:
        event["inference_seconds"] = time.perf_counter() - started
        result["inference_seconds"] += event["inference_seconds"]


def play_game(arcade, game, card_id, endpoint, model, args, run, task_directory, record_action,
              *, harness="none", upstream=None, split_outputs=False):
    from arcengine import GameAction, GameState

    started = time.perf_counter()
    game_timeout = getattr(args, "game_timeout", None)
    deadline = time.monotonic() + game_timeout if game_timeout is not None else None
    duck = None
    history = deque(maxlen=16)
    tried_actions = defaultdict(Counter)
    previous = None
    notes = ""
    observation = None
    result = {
        "model": model, "harness": harness, "game_id": game.game_id, "game_name": game.title or game.game_id,
        "accepted_moves": 0, "model_actions": 0, "invalid_responses": 0, "recovery_resets": 0,
        "attempt_actions": 0, "level_actions": 0,
        "model_calls": 0, "completion_tokens": 0, "inference_seconds": 0.0, "status": "action_limit",
    }
    identifiers = {key: result[key] for key in ("model", "harness", "game_id", "game_name")}
    identifiers["scorecard_id"] = card_id
    trace_path = Path(task_directory) / model
    if split_outputs or harness == "duck":
        trace_path /= harness
    action_dir = run / "actions" / trace_path if args.save_actions else None
    frame_dir = run / "frames" / trace_path if args.save_frames else None
    image_dir = run / "frame_images" / trace_path if args.plot_frames else None

    def time_expired():
        return deadline is not None and time.monotonic() >= deadline

    try:
        limits = action_limits(game)
        game_limit = sum(limits)
        if args.max_actions is not None:
            game_limit = min(game_limit, args.max_actions)
        for directory in (action_dir, frame_dir, image_dir):
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
        # In the pinned toolkit make() already performs the first RESET.
        env = arcade.make(game.game_id, seed=args.seed, scorecard_id=card_id)
        if env is None or env.observation_space is None:
            raise RuntimeError("ARC failed to initialize this game")
        observation = env.observation_space
        if observation.win_levels != len(limits):
            raise RuntimeError(f"Expected {len(limits)} human baselines for {game.game_id}, "
                               f"but ARC reported {observation.win_levels} levels")
        level = observation.levels_completed
        record_action({
            **identifiers, "event": "initial", "source": "initialization", "action_index": 0,
            "attempt": 1, "level": level + 1, "action": "RESET", "accepted": True,
            "status": "applied", "state_after": observation.state.name,
            "game_guid_after": observation.guid,
            "levels_after": level, "levels_total": observation.win_levels,
            "attempt_actions": 0, "attempt_action_limit": game_limit,
            "level_actions": 0, "level_action_limit": limits[level] if level < len(limits) else None,
            "human_actions": game.baseline_actions[level] if level < len(limits) else None,
            "model_actions": 0, "accepted_moves": 0, "recovery_resets": 0,
            "elapsed_seconds": time.perf_counter() - started,
        })
        save_observation(observation, 0, frame_dir, image_dir)
        if harness == "duck":
            from .duck.adapter import DuckSession

            duck = DuckSession(upstream, endpoint, model, run / "duck" / model / task_directory,
                               args, CONTEXT_TOKENS, COMPLETION_TOKENS)
            duck.deadline = deadline
            duck.update(observation, 0)

        def available_actions():
            actions = ([GameAction.from_id(action) for action in observation.available_actions]
                       if observation.available_actions else env.action_space)
            return {action.name: action for action in actions if action is not GameAction.RESET}

        def recovery_reason():
            if observation.state is GameState.GAME_OVER:
                return "game_over"
            remaining = game_limit - result["attempt_actions"]
            if result["level_actions"] >= min(limits[observation.levels_completed], remaining):
                return "level_action_limit"
            return None

        def step_game(reset_reason=None, selected=None):
            nonlocal observation, previous, notes

            level = observation.levels_completed
            recovering = reset_reason is not None
            if not recovering:
                result["model_actions"] += 1
                result["level_actions"] += 1
            event = {
                "action_index": result["model_actions"] + result["recovery_resets"] + int(recovering),
                "model_action": None if recovering else result["model_actions"],
                "attempt": result["recovery_resets"] + 1,
                "attempt_action": result["attempt_actions"],
                "level_action": result["level_actions"], "level_action_limit": limits[level],
                "source": "recovery" if recovering else "model",
                "status": "applied", "action": None, "data": None,
                "state_before": observation.state.name,
                "levels_before": observation.levels_completed,
            }
            accepted = False
            available = {}
            guid_before = observation.guid
            action_started = time.perf_counter()
            try:
                if recovering:
                    event.update(action="RESET", data={}, reset_reason=reset_reason)
                    # The SDK treats RESET after gameplay as a level reset. Use
                    # the existing session so a missing remote game cannot start a new one.
                    if not guid_before or result["level_actions"] == 0:
                        raise RuntimeError("Cannot safely reset this level without an active played session")
                    next_observation = env.step(GameAction.RESET, data={})
                else:
                    available = available_actions()
                    if not available:
                        raise RuntimeError("ARC provided no playable action")
                    state_key = None
                    if duck is None and args.allow_thinking and observation.frame:
                        frame_key = hashlib.blake2b(observation.frame[-1].tobytes(), digest_size=16).digest()
                        state_key = (observation.levels_completed, frame_key)
                    if selected is None:
                        action, data, note = choose_action(
                            endpoint, model, observation, available, history, args, event, result,
                            previous=previous, notes=notes, deadline=deadline,
                            repeats=tried_actions[state_key].most_common(5) if state_key is not None else (),
                        )
                    else:
                        action, data, note = selected
                    event.update(action=action.name, data=data)
                    action_text = f"{action.name} {data}"
                    if state_key is not None:
                        tried_actions[state_key][action_text] += 1
                    if args.keep_notes and note:
                        notes = note[:1200]
                        event["scratchpad"] = notes
                    if time_expired():
                        raise TimeoutError("ARC game wall-clock limit reached")
                    next_observation = env.step(action, data=data)
                if next_observation is None:
                    raise RuntimeError("ARC returned no observation for an action")
                if recovering:
                    if (next_observation.guid != guid_before
                            or getattr(next_observation, "full_reset", False)
                            or next_observation.levels_completed != level
                            or next_observation.state is not GameState.NOT_FINISHED):
                        raise RuntimeError("ARC did not restart the current level in the same game")
                    event["refunded_level_actions"] = result["level_actions"]
                    result["recovery_resets"] += 1
                    result["level_actions"] = 0
                    history.append(f"Current level restarted after {reset_reason}. Reuse what you learned.")
                else:
                    result["accepted_moves"] += 1
                    if next_observation.levels_completed > level:
                        result["attempt_actions"] += result["level_actions"]
                        result["level_actions"] = 0
                accepted = True
                previous, observation = observation, next_observation
                if not recovering:
                    history.append(f"{action_text} → {observation.state.name}; "
                                   f"{observation.levels_completed} levels cleared. "
                                   f"{'' if args.keep_notes else note[:240]}".strip())
            except Exception as error:
                event.update(status="error", error=str(error))
                raise
            finally:
                event.update(
                    state_after=observation.state.name, levels_after=observation.levels_completed,
                    attempt_actions_after=result["attempt_actions"], level_actions_after=result["level_actions"],
                )
                replies = event.get("replies", ())
                data = event["data"] or {}
                outcome = {"WIN": "won", "GAME_OVER": "lost"}.get(event["state_after"])
                record_action({
                    **identifiers, "event": "action", "source": event["source"],
                    "action_index": event["action_index"], "model_action": event["model_action"],
                    "model_actions": result["model_actions"],
                    "attempt": event["attempt"], "level": level + 1, "action": event["action"],
                    "x": data.get("x"), "y": data.get("y"), "accepted": accepted,
                    "status": event["status"], "attempt_actions": event["attempt_action"],
                    "attempt_action_limit": game_limit, "level_actions": event["level_action"],
                    "level_action_limit": limits[level], "human_actions": game.baseline_actions[level],
                    "attempt_actions_after": event["attempt_actions_after"],
                    "level_actions_after": event["level_actions_after"],
                    "refunded_level_actions": event.get("refunded_level_actions"),
                    "reset_reason": event.get("reset_reason"),
                    "accepted_moves": result["accepted_moves"], "recovery_resets": result["recovery_resets"],
                    "state_before": event["state_before"], "state_after": event["state_after"],
                    "levels_before": level, "levels_after": event["levels_after"],
                    "levels_total": observation.win_levels, "level_cleared": event["levels_after"] > level,
                    "attempt_outcome": outcome if accepted and not recovering else None,
                    "game_guid_before": guid_before, "game_guid_after": observation.guid,
                    "available_actions": " ".join(available),
                    "model_replies": len(replies),
                    "invalid_responses": sum("error" in reply for reply in replies),
                    "finish_reasons": json.dumps([reply["finish_reason"] for reply in replies]),
                    "reply_errors": json.dumps([reply["error"] for reply in replies if "error" in reply]),
                    "completion_tokens": sum(reply["completion_tokens"] for reply in replies),
                    "inference_seconds": event.get("inference_seconds"),
                    "action_seconds": time.perf_counter() - action_started,
                    "elapsed_seconds": time.perf_counter() - started, "error": event.get("error"),
                })
                if action_dir is not None:
                    with (action_dir / "actions.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({**identifiers, **event}, ensure_ascii=False) + "\n")
            save_observation(observation, event["action_index"], frame_dir, image_dir)
            if duck is not None:
                duck.update(observation, event["action_index"], event["action"])

        while observation.state is not GameState.WIN:
            if time_expired():
                result["status"] = "time_limit"
                break
            level = observation.levels_completed
            result["level_action_limit"] = limits[level]
            if result["attempt_actions"] >= game_limit:
                break
            reset_reason = recovery_reason()
            if reset_reason is not None:
                if result["recovery_resets"] >= args.bounded_recovery:
                    break
                step_game(reset_reason)
            elif duck is None:
                step_game()
            else:
                duck_error = None

                def stop_batch():
                    return (duck_error is not None or time_expired()
                            or observation.state is GameState.WIN
                            or observation.levels_completed != level
                            or recovery_reason() is not None)

                def play_batch(payload):
                    nonlocal duck_error

                    try:
                        return duck.play_batch(
                            payload, lambda: observation, available_actions, decode_action,
                            lambda action: step_game(selected=action), stop_batch, result,
                        )
                    except Exception as error:
                        duck_error = error
                        raise

                moves = result["accepted_moves"]
                duck.analyze(result["model_actions"] + 1, list(available_actions()), play_batch, stop_batch)
                if duck_error is not None:
                    raise duck_error
                if result["accepted_moves"] == moves and not time_expired():
                    raise RuntimeError("Duck returned no accepted action. See its game transcript.")
        if result["status"] != "time_limit" and observation.state in (GameState.WIN, GameState.GAME_OVER):
            result["status"] = observation.state.name
    except Exception as error:
        if isinstance(error, TimeoutError) and time_expired():
            result["status"] = "time_limit"
        else:
            result.update(status="error", error=str(error))
    finally:
        if duck is not None:
            result.update(model_calls=duck.response_count, completion_tokens=duck.completion_tokens,
                          inference_seconds=duck.inference_seconds)
            duck.close()
        result["seconds"] = time.perf_counter() - started
    if observation is not None:
        result.update(final_state=observation.state.name, levels_completed=observation.levels_completed,
                      levels_total=observation.win_levels)
    return result, getattr(observation, "guid", None)


def read_game_score(card, game_id, guid):
    if card is None:
        return None
    return next((game for game in card.environments
                 if game.id == game_id or (guid and any(entry.guid == guid for entry in game.runs))), None)


def format_duration(seconds):
    if seconds is None:
        return "—"
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_game(console, result):
    score = result.get("rhae_percent")
    score_text = "unavailable" if score is None else f"{score:.2f}%"
    levels = f"{result.get('levels_completed', '—')}/{result.get('levels_total', '—')}"
    game_pass = f" · pass {result['pass']}" if "pass" in result else ""
    condition = result["harness"]
    if effort := result.get("reasoning_effort"):
        condition += f", {effort}"
    retries = f"level retries {result['level_retries']}  │  " if "level_retries" in result else ""
    console.print(f"{result['model']} [{condition}]  │  {result['game_name']} ({result['game_id']}){game_pass}  │  "
                  f"RHAE {score_text}  │  levels {levels}  │  resets {result['recovery_resets']}  │  "
                  f"{retries}"
                  f"accepted moves {result['accepted_moves']}  │  "
                  f"{result['status']}  │  "
                  f"{format_duration(result['seconds'])}", markup=False, highlight=False)
    if result.get("error"):
        console.print(f"  {result['error']}", markup=False, highlight=False)


def evaluate_model(args, arcade, games, run, endpoint, model, harness, writer, stream, console,
                   directories, record_action, upstream, startup_error):
    results = []
    card = None
    model_error = None
    split_outputs = len(args.harness) > 1
    try:
        if startup_error is not None:
            raise RuntimeError(startup_error)
        card_id = arcade.create_scorecard(tags=["grpo-mosp-arc3", run.name, model, harness])
        if not card_id:
            raise RuntimeError("ARC did not return a scorecard ID")
        try:
            for game in games:
                result, guid = play_game(arcade, game, card_id, endpoint, model, args,
                                         run, directories[game.game_id], record_action,
                                         harness=harness, upstream=upstream, split_outputs=split_outputs)
                final_levels = result.get("levels_completed")
                try:
                    score = read_game_score(arcade.get_scorecard(card_id), game.game_id, guid)
                    if score is None or not score.runs:
                        raise RuntimeError("The official scorecard has no entry for this game")
                    message = next((entry.message for entry in score.runs if entry.message), None)
                    if message:
                        raise RuntimeError(message)
                    result.update(rhae_percent=score.score, official_actions=score.actions,
                                  levels_completed=max(entry.levels_completed for entry in score.runs))
                except Exception as error:
                    detail = result.get("error", "")
                    result.update(status="error", error=f"{detail} Scorecard: {error}".strip())
                record_action({
                    **result, "scorecard_id": card_id, "event": "game_result",
                    "state_after": result.get("final_state"),
                    "levels_after": final_levels, "game_guid_after": guid,
                    "attempt": result["recovery_resets"] + 1 if final_levels is not None else None,
                    "attempt_outcome": {"WIN": "won", "GAME_OVER": "lost"}.get(result.get("final_state")),
                    "elapsed_seconds": result["seconds"],
                })
                results.append(result)
                writer.writerow(result)
                stream.flush()
                print_game(console, result)
        finally:
            card = arcade.close_scorecard(card_id)
            if card is None:
                raise RuntimeError(f"ARC did not finalize scorecard {card_id}")
            suffix = f".{harness}" if split_outputs or harness == "duck" else ""
            write_json(run / f"{model}{suffix}.scorecard.json", card.model_dump(mode="json", exclude={"api_key"}))
    except Exception as error:
        model_error = str(error)
        console.print(f"{model} [{harness}] · {model_error}", markup=False, highlight=False)

    summary = summarize_games(results, model, harness, len(games), model_error)
    summary.update(
        official_rhae_percent=card.score if card is not None and summary["status"] == "complete" else None,
        scorecard_id=card.card_id if card is not None else None,
    )
    return summary


def summarize_games(results, model, harness, games_total, model_error=None):
    times = [result["seconds"] for result in results if result["seconds"] is not None]
    resets = [result["recovery_resets"] for result in results if "levels_total" in result]
    scored = sum(result.get("rhae_percent") is not None for result in results)
    errors = sum(result["status"] == "error" for result in results)
    complete = scored == games_total and errors == 0 and model_error is None
    return {
        "model": model, "harness": harness, "games_total": games_total, "games_timed": len(times),
        "games_scored": scored, "games_won": sum(row["status"] == "WIN" for row in results),
        "games_with_level_clear": sum(row.get("levels_completed", 0) > 0 for row in results),
        "levels_completed": sum(row.get("levels_completed", 0) for row in results),
        "accepted_moves": sum(row["accepted_moves"] for row in results),
        "model_actions": sum(row["model_actions"] for row in results),
        "model_calls": sum(row["model_calls"] for row in results),
        "recovery_resets": sum(resets),
        "games_with_reset_data": len(resets),
        "mean_recovery_resets": mean(resets) if resets else None,
        "variance_recovery_resets": pvariance(resets) if resets else None,
        "completion_tokens": sum(row["completion_tokens"] for row in results),
        "inference_seconds": sum(row["inference_seconds"] for row in results),
        "errors": errors,
        "mean_seconds": mean(times) if times else None,
        "variance_seconds_squared": pvariance(times) if times else None,
        "total_seconds": sum(times), "status": "complete" if complete else "incomplete",
        "error": model_error,
    }


def show_summary(console, summaries, *, score_key="official_rhae_percent", score_label="RHAE"):
    from rich.table import Table

    def number(value, suffix=""):
        return "—" if value is None else f"{value:.2f}{suffix}"

    table = Table(title="ARC-AGI-3 model comparison")
    show_effort = any("reasoning_effort" in row for row in summaries)
    show_backend = any("backend" in row for row in summaries)
    show_retries = any("level_retries" in row for row in summaries)
    table.add_column("Model", justify="left")
    table.add_column("Harness", justify="left")
    if show_backend:
        table.add_column("Backend", justify="left")
    if show_effort:
        table.add_column("Effort", justify="left")
    for name in ("Scored", "Wins", "Games cleared", "Levels", score_label, "Moves", "Calls", "Gen. tokens",
                 "Mean resets", "Reset variance", *(("Retries",) if show_retries else ()),
                 "Mean time", "Variance (s²)", "Total time", "Status"):
        table.add_column(name, justify="left" if name == "Status" else "right",
                         min_width=len("incomplete") if name == "Status" else None)
    for row in summaries:
        effort = (row.get("reasoning_effort", "—"),) if show_effort else ()
        backend = (row.get("backend", "—"),) if show_backend else ()
        retries = (str(row.get("level_retries", "—")),) if show_retries else ()
        table.add_row(
            row["model"], row["harness"], *backend, *effort, f"{row['games_scored']}/{row['games_total']}", str(row["games_won"]),
            str(row["games_with_level_clear"]), str(row["levels_completed"]),
            number(row[score_key], "%"),
            str(row["accepted_moves"]), str(row["model_calls"]), str(row["completion_tokens"]),
            number(row["mean_recovery_resets"]), number(row["variance_recovery_resets"]),
            *retries,
            format_duration(row["mean_seconds"]), number(row["variance_seconds_squared"]),
            format_duration(row["total_seconds"]), row["status"],
        )
    console.print(table)


def run_experiment(args, model_config, snapshot, models):
    from rich.console import Console

    run = new_run()
    logger = logging.getLogger("mosp.arc3")
    logger.setLevel(logging.WARNING)
    handler = logging.FileHandler(run / "toolkit.log")
    logger.addHandler(handler)
    logger.propagate = False
    summaries = []
    duck_settings = None
    duck_runtime = nullcontext(None)
    if "duck" in args.harness:
        from .duck.adapter import duck_config, load_duck

        duck_settings = duck_config()
        duck_runtime = load_duck()
    try:
        with artifact_environment(run), duck_runtime as upstream:
            console = Console(file=sys.stdout, width=168, force_terminal=False)
            arcade, catalog = make_arcade(args, run, logger)
            labels = load_difficulties(catalog)
            games = select_games(catalog, args, labels)
            limits = {game.game_id: action_limits(game) for game in games}
            directories = task_directories(games)
            manifest = {
                "mode": arcade.operation_mode.value, "games": game_catalog(games, labels),
                "arc_agi": version("arc-agi"), "arcengine": version("arcengine"),
                "catalog_size": len(catalog),
                "n_tasks": len(games),
                "difficulty": args.difficulty,
                "task_sampling_seed": args.seed if args.n_tasks is not None else None,
                "game_seed": args.seed if args.environments_dir else None,
                "human_action_multiplier": HUMAN_ACTION_MULTIPLIER,
                "level_action_limits": limits, "max_actions_per_game": args.max_actions,
                "max_replies_per_action": MAX_REPLY_ATTEMPTS, "model_config": model_config,
                "keep_notes": args.keep_notes, "allow_thinking": args.allow_thinking,
                "harnesses": args.harness, "duck": duck_settings, "game_timeout_seconds": args.game_timeout,
                "bounded_recovery": args.bounded_recovery > 0,
                "context_limit_tokens": CONTEXT_TOKENS,
                "max_completion_tokens_per_reply": COMPLETION_TOKENS,
                "max_recovery_resets_per_game": args.bounded_recovery,
                "recovery_reset_scope": "level",
                "base_snapshot": str(snapshot), "model_python": args.model_python,
                "models": model_paths(models), "time": "game initialization and play, seconds",
                "save_actions": args.save_actions, "save_frames": args.save_frames,
                "plot_frames": args.plot_frames, "task_directories": directories,
                "variance": "population variance across timed games",
                "offline_game_directory": str(args.environments_dir) if args.environments_dir else None,
            }
            write_json(run / "manifest.json", manifest)
            console.print(f"ARC-AGI-3 · {len(games)} games × {len(models)} models × {len(args.harness)} harness conditions · {run}",
                          markup=False)
            with ((run / "tasks.csv").open("w", newline="", encoding="utf-8") as stream,
                  (run / "action_data.csv").open("w", newline="", encoding="utf-8") as action_stream):
                writer = csv.DictWriter(stream, fieldnames=TASK_FIELDS)
                writer.writeheader()
                action_writer = csv.DictWriter(action_stream, fieldnames=ACTION_FIELDS, extrasaction="ignore")
                action_writer.writeheader()

                def record_action(row):
                    action_writer.writerow(row)
                    action_stream.flush()

                for model, adapter in models.items():
                    with ExitStack() as server:
                        endpoint, startup_error = None, None
                        try:
                            endpoint = server.enter_context(serve_checkpoint(
                                args.model_python, snapshot, adapter, model,
                                run / f"{model}.server.log", run / ".server-port",
                                server_args=("--context-limit", str(CONTEXT_TOKENS),
                                             "--completion-limit", str(COMPLETION_TOKENS),
                                             "--fit-completion-to-context"),
                            ))
                        except Exception as error:
                            startup_error = str(error)
                        for harness in args.harness:
                            summaries.append(evaluate_model(
                                args, arcade, games, run, endpoint, model, harness, writer, stream,
                                console, directories, record_action, upstream, startup_error,
                            ))
            with (run / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
            show_summary(console, summaries)
    finally:
        logger.removeHandler(handler)
        handler.close()
    if any(result["status"] != "complete" for result in summaries):
        raise RuntimeError(f"Some model evaluations were incomplete. See {run / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_options(parser)
    parser.add_argument("--harness", action="append", choices=("none", "duck"),
                        help="Repeat to compare harness conditions (default: none).")
    parser.add_argument("--game-timeout", type=int,
                        help="Per-game wall-clock limit in seconds (default: 2700 for runs with Duck).")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env",
                        help="File containing ARC_API_KEY if it is absent from the environment.")
    parser.add_argument("--environments-dir", type=Path,
                        help="Use an offline ARC game bundle with official human baselines.")
    parser.add_argument("--n-tasks", type=int, metavar="N",
                        help="Sample N distinct games, stratified unless --difficulty is set.")
    parser.add_argument("--difficulty", choices=DIFFICULTIES,
                        help="Limit evaluation to games with this label in the human data CSV.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for task sampling and offline games (default: 0).")
    parser.add_argument("--max-actions", type=int, metavar="N",
                        help="Optional additional cap on the game action budget for short runs.")
    parser.add_argument("--timeout", type=int, default=180, help="Local model request timeout in seconds.")
    parser.add_argument("--keep-notes", action="store_true",
                        help="Keep a per-game scratchpad and show visual changes (default: off).")
    parser.add_argument("--allow-thinking", action="store_true",
                        help="Enable Qwen thinking and show repeated state-action pairs (default: off).")
    parser.add_argument("--bounded-recovery", type=int, default=DEFAULT_RECOVERY_RESETS, metavar="N",
                        help=f"Allow up to N current-level resets per game after game over or budget exhaustion "
                             f"(default: {DEFAULT_RECOVERY_RESETS}, 0 disables recovery).")
    parser.add_argument("--save-frames", action=argparse.BooleanOptionalAction, default=False,
                        help="Save raw frame arrays for each action (default: off).")
    parser.add_argument("--save-actions", action=argparse.BooleanOptionalAction, default=True,
                        help="Save a JSONL record for each action (default: on).")
    parser.add_argument("--plot-frames", action=argparse.BooleanOptionalAction, default=False,
                        help="Export a PNG for every frame using PyPlot (default: off).")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--list-games", action="store_true",
                           help="List the games that would be evaluated.")
    operation.add_argument("--inventory", action="store_true", help="Resolve the selected local checkpoints.")
    args = parser.parse_args()
    args.harness = list(dict.fromkeys(args.harness or ["none"]))
    if args.game_timeout is not None and args.game_timeout < 1:
        parser.error("--game-timeout must be positive")
    if args.game_timeout is None and "duck" in args.harness:
        args.game_timeout = DUCK_GAME_TIMEOUT
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if args.max_actions is not None and args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if args.n_tasks is not None and args.n_tasks < 1:
        parser.error("--n-tasks must be positive")
    if args.bounded_recovery < 0:
        parser.error("--bounded-recovery must be non-negative")
    if args.environments_dir:
        args.environments_dir = args.environments_dir.expanduser().resolve()
        if not args.environments_dir.is_dir():
            parser.error("--environments-dir must name an installed ARC game bundle")
    if args.list_games:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix="arc3-catalog-") as root, redirect_stdout(sys.stderr):
            _, games = make_arcade(args, Path(root), logging.getLogger("mosp.arc3"))
        labels = load_difficulties(games)
        print(json.dumps(game_catalog(select_games(games, args, labels), labels), indent=2))
        return
    model_config, snapshot, models = load_selection(args)
    if args.inventory:
        print(json.dumps({"snapshot": str(snapshot), "models": model_paths(models)}, indent=2))
        return
    run_experiment(args, model_config, snapshot, models)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
