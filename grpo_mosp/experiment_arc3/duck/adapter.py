"""Connect Tufa's Duck tool policy to the shared ARC evaluator and Qwen server."""

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent

from .install import REVISION, UPSTREAM, installed

CONFIG = Path(__file__).with_name("config.json")


def duck_config():
    return {"revision": REVISION, "preserve_thinking": True,
            **json.loads(CONFIG.read_text(encoding="utf-8"))}


@contextmanager
def load_duck():
    if not installed():
        raise RuntimeError("Run experiment_arc3/duck/install.py to install the pinned Duck inference source")
    if "inference" in sys.modules:
        package = sys.modules["inference"]
        if not Path(package.__file__).resolve().is_relative_to(UPSTREAM):
            raise RuntimeError("The Duck inference package conflicts with an already loaded package")
    sys.path.insert(0, str(UPSTREAM))
    try:
        from inference.agent import python_tool_sandbox
        from inference.agent import tool_agent
        from inference.agent import runtime_state
    finally:
        sys.path.pop(0)

    original = python_tool_sandbox._SANDBOX_BOOTSTRAP
    anchor = "    _set_limits(timeout_seconds)\n"
    if original.count(anchor) != 1:
        raise RuntimeError("The pinned Duck sandbox startup hook has changed")
    limits = Path(__file__).with_name("worker_limits.py").read_text(encoding="utf-8")
    python_tool_sandbox._SANDBOX_BOOTSTRAP = (
        dedent(limits) + "\n" + original.replace(anchor, anchor + "    restrict_worker(SAFE_MODULES)\n")
    )
    variables = {"MULTIMODAL_CONTEXT": "current_grid", "MULTIMODAL_UPSCALE": str(duck_config()["image_upscale"])}
    previous = {key: os.environ.get(key) for key in variables}
    os.environ.update(variables)
    try:
        yield tool_agent, runtime_state
    finally:
        python_tool_sandbox._SANDBOX_BOOTSTRAP = original
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class DuckSession:
    def __init__(self, upstream, endpoint, model, output_dir, args, context_limit, completion_limit):
        import requests

        tool_agent, self.runtime = upstream
        config = duck_config()
        self.response_count = 0
        self.completion_tokens = 0
        self.inference_seconds = 0.0
        self.deadline = None
        self.url = endpoint + "/chat/completions"
        self.args = args
        self.model = model
        self.completion_limit = completion_limit
        self.directory = output_dir
        self.state_path = output_dir / self.runtime.RUNTIME_STATE_FILENAME
        self.history = []
        self.client = requests.Session()
        self.client.trust_env = False
        self.client.headers["Content-Type"] = "application/json"
        session = self

        class LocalToolAgent(tool_agent.ToolAgent):
            def _chat_completion(self, messages, *, tools, request_timeout_seconds=None):
                payload = {
                    "model": session.model, "messages": messages, "tools": tools,
                    "tool_choice": "required" if tools else "none",
                    "max_tokens": session.completion_limit,
                    "chat_template_kwargs": {
                        "enable_thinking": args.allow_thinking,
                        "preserve_thinking": config["preserve_thinking"],
                    },
                    "temperature": 0, "stream": False,
                }
                timeout = args.timeout if request_timeout_seconds is None else request_timeout_seconds
                if session.deadline is not None:
                    timeout = min(timeout, max(0.001, session.deadline - time.monotonic()))
                session.response_count += 1
                start = time.perf_counter()
                try:
                    reply = session.client.post(session.url, json=payload, timeout=timeout)
                    reply.raise_for_status()
                    value = reply.json()
                    choice = value["choices"][0]
                    usage = value["usage"]
                    session.completion_tokens += usage["completion_tokens"]
                    return tool_agent._ChatCompletionResult(choice["message"], choice["finish_reason"], usage)
                finally:
                    session.inference_seconds += time.perf_counter() - start

        self.agent = LocalToolAgent(model=model, base_url=endpoint, provider="vllm",
                                    api_key="local", save_request_logs=args.save_actions)
        self.agent._max_output_tokens = completion_limit
        self.agent._reply_reserve_tokens = completion_limit
        self.agent._context_budget_tokens = context_limit - completion_limit - self.agent._request_safety_margin_tokens
        self.agent._tool_steps = config["max_model_calls_per_decision"]

    def close(self):
        self.client.close()

    def update(self, observation, index, action=""):
        grid = tuple(tuple(int(cell) for cell in row) for row in observation.frame[-1]) if observation.frame else ()
        level = min(observation.levels_completed + 1, observation.win_levels)
        frame = self.runtime.Frame(grid, index, level)
        self.history.append(self.runtime.HistoryEntry(action, frame))
        self.runtime.write_runtime_state(self.state_path, current_frame=frame, history=self.history)

    def analyze(self, action_count, valid_actions, callback, should_stop):
        return self.agent.analyze(
            self.state_path, action_count, valid_actions=valid_actions, step_env=callback,
            should_stop=should_stop,
        )

    def play_batch(self, payload, observe, available_actions, decode_action, step, should_stop, result):
        from inference.agent.action_names import to_engine_action, to_model_action

        import numpy as np

        requested = payload["actions"]
        before = observe()
        executed = []
        error = None
        for item in requested:
            if should_stop():
                break
            name = item["action"].upper()
            name = "ACTION7" if name == "UNDO" else (to_engine_action(name) or name)
            data = {"action": name, "x": item.get("col"), "y": item.get("row")}
            try:
                action = decode_action(json.dumps(data), available_actions())
            except ValueError as invalid:
                result["invalid_responses"] += 1
                error = str(invalid)
                break
            step(action)
            executed.append(to_model_action(name))

        after = observe()
        boundary = should_stop()
        response = {
            "executed": bool(executed), "executed_actions": executed,
            "action_num": result["model_actions"], "level": min(after.levels_completed + 1, after.win_levels),
            "score": after.levels_completed, "reward": after.levels_completed - before.levels_completed,
            "state": after.state.name, "valid_actions": [] if boundary else list(available_actions()),
            "board_changed": bool(before.frame and after.frame and not np.array_equal(before.frame[-1], after.frame[-1])),
            "done": boundary, "level_completed": after.levels_completed > before.levels_completed,
            "game_over": after.state.name == "GAME_OVER", "run_complete": after.state.name == "WIN",
            "requested_count": len(requested), "executed_count": len(executed),
            "stopped_early": len(executed) != len(requested),
        }
        if error is not None:
            response.update(error=error, stop_reason="invalid_action")
        elif response["stopped_early"]:
            response["stop_reason"] = "evaluator_boundary"
        return response
