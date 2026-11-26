"""Regression tests for bounded Duck gameplay with the installed Tufa runner."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from ..experiment_arc3 import evaluate_arc3_duck as evaluation
from ..experiment_arc3.duck import move_budget
from ..experiment_arc3.duck.install import UPSTREAM, runner_installed


@pytest.fixture
def duck(monkeypatch):
    if not runner_installed():
        pytest.skip("The pinned Duck runner is required")
    monkeypatch.syspath_prepend(str(UPSTREAM / "dependencies"))
    monkeypatch.syspath_prepend(str(UPSTREAM))
    from ..experiment_arc3.duck import level_budget, native_policy
    from inference.agent import tool_agent
    import arcengine

    config = json.loads((UPSTREAM / "configs/inference.json").read_text())
    return SimpleNamespace(
        config=config, policy=native_policy.NativeDuckAnalyzer, solver=level_budget.LevelBudgetSolver,
        tool_agent=tool_agent, engine=arcengine,
    )


@pytest.fixture
def tool_clock(duck, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(duck.tool_agent, "time", SimpleNamespace(
        monotonic=lambda: clock.now, strftime=time.strftime,
    ))
    return clock


def tool_exchange(call_id, content, code="print(current_frame.step)"):
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": "python", "arguments": json.dumps({"code": code})},
    }
    return (
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call_id, "content": content},
    )


def tool_reply(code):
    assistant, _ = tool_exchange("call-python", "", code)
    body = {"choices": [{"message": assistant, "finish_reason": "tool_calls"}],
            "usage": {"completion_tokens": 1}}
    return SimpleNamespace(ok=True, status_code=200, json=lambda: body)


class OfflineGame:
    def __init__(self, engine, game_id, *, lose_first_move=False):
        self.engine = engine
        self.game_id = game_id
        self.lose_first_move = lose_first_move
        self.number_of_levels = 1
        self.level_retries = {}
        self.game_run = SimpleNamespace(
            game_id=game_id, state="playing", final_score=None, solver_note=None,
            history=[], base_actions_per_level=None, actions_per_level=[0], levels_completed=0, number_of_levels=1,
        )
        self.set_state(engine.GameState.NOT_FINISHED)

    def set_state(self, state, color=0):
        won = state == self.engine.GameState.WIN
        self.current_state = SimpleNamespace(
            raw=SimpleNamespace(state=state, guid="game-guid", levels_completed=int(won), full_reset=False),
            frame=SimpleNamespace(data=[[color]]), won=won, levels_completed=int(won), just_won_level=won,
            available_actions=[self.engine.GameAction.ACTION1.value],
        )

    def execute_action(self, action, **kwargs):
        self.game_run.history.append(action.id)
        if action.id == self.engine.GameAction.RESET:
            self.set_state(self.engine.GameState.NOT_FINISHED)
        elif self.lose_first_move and len(self.game_run.history) == 1:
            self.set_state(self.engine.GameState.GAME_OVER, color=1)
        else:
            self.game_run.state = "won"
            self.set_state(self.engine.GameState.WIN, color=2)
            self.game_run.levels_completed = 1
        return self.current_state

    def finish_game(self):
        self.game_run.final_score = float(self.game_run.levels_completed)


def sandbox(*, code, timeout_seconds, initial_state, action_handler):
    if code == "act":
        payload = action_handler(["UP"])
        return {"action_results": [payload["action_result"]]}
    return {"stdout": str(initial_state["last_action_result"])}


def run_games(duck, tmp_path, games, analyzers):
    solver = duck.solver(label="test-duck", concurrency=1, analyzer_factory=lambda game, index: analyzers[index])
    solver.job_dir = tmp_path
    asyncio.run(solver._run_games(games))


@pytest.mark.parametrize(("tool_steps", "cap"), [(0, None), (1, 3)])
def test_watchdog_finishes_stalled_pass_and_runs_next_game(duck, monkeypatch, tmp_path, tool_steps, cap):
    monkeypatch.setattr(duck.tool_agent, "run_sandboxed_python", sandbox)
    duck.config["analyzer"]["tool_steps"] = tool_steps
    stalls = Mock(return_value=tool_reply("inspect"))
    kwargs = {} if cap is None else {"accepted_move_cap": cap}
    client = evaluation.ModelClient("http://unused", duck.config, post=stalls, **kwargs)
    next_client = evaluation.ModelClient(
        "http://unused", duck.config, post=Mock(return_value=tool_reply("act")),
        accepted_move_cap=1,
    )
    analyzers = [duck.policy(each, "http://unused") for each in (client, next_client)]
    first, second = (OfflineGame(duck.engine, game_id) for game_id in ("first", "second"))

    run_games(duck, tmp_path, [first, second], analyzers)

    expected = 16 if cap is None else cap
    assert stalls.call_count == expected, first.game_run.solver_note
    assert f"no_game_progress: {expected} model calls without an accepted move" in first.game_run.solver_note
    assert first.game_run.state == "crashed"
    assert first.game_run.history == []
    assert first.level_retries == {}
    assert "TOOL RESULT" in (tmp_path / "transcripts/first_p0.txt").read_text()
    assert second.game_run.state == "won"
    assert next_client.move_budget.calls == 0


@pytest.mark.parametrize("yield_seconds", [None, evaluation.DUCK_GAMEPLAY_YIELD_SECONDS])
def test_gameplay_yield_keeps_the_same_step_and_watchdog(duck, tool_clock, monkeypatch, tmp_path, yield_seconds):
    monkeypatch.setattr(duck.tool_agent, "run_sandboxed_python", sandbox)
    duck.config["analyzer"]["tool_steps"] = 0
    reply = tool_reply("inspect")

    def post(*args, **kwargs):
        tool_clock.now += 61.0
        return reply

    client = evaluation.ModelClient("http://unused", duck.config, post=post, accepted_move_cap=3)
    analyzer = duck.policy(client, "http://unused", yield_seconds=yield_seconds)
    game = OfflineGame(duck.engine, "yield-test")

    run_games(duck, tmp_path, [game], [analyzer])

    transcript = (tmp_path / "transcripts/yield-test_p0.txt").read_text()
    assert transcript.count("--- analysis_step=1 | action=1 |") == (3 if yield_seconds else 1)
    assert transcript.count("Yielded control to solver: turn_time_budget.") == (3 if yield_seconds else 0)
    assert "no_game_progress: 3 model calls" in game.game_run.solver_note
    assert client.model_calls == 3


def test_preflight_has_no_cooperative_yield(duck, tool_clock, monkeypatch, tmp_path):
    from inference.agent.runtime_state import Frame

    monkeypatch.setattr(duck.tool_agent, "_LOCAL_ANALYZER_YIELD_SECONDS", 60)
    monkeypatch.setattr(duck.tool_agent, "run_sandboxed_python", sandbox)
    duck.config["analyzer"]["tool_steps"] = 0
    args = SimpleNamespace(
        deadline=True, max_output_tokens=None, generation_timeout=230, request_timeout=240, accepted_move_cap=16,
        duck_yield_seconds=evaluation.DUCK_GAMEPLAY_YIELD_SECONDS,
    )
    analyzer = evaluation.policy_analyzer(
        args, duck.config, "http://unused", evaluation.Condition("qwen-frozen", "duck", "medium"),
    )
    replies = iter([tool_reply("inspect"), tool_reply("act")])

    def post(*args, **kwargs):
        tool_clock.now += 61.0
        return next(replies)

    analyzer.client.post = post
    move = evaluation.preflight_duck(
        analyzer, Frame(((0,),), 0, 1), ["ACTION1"], tmp_path, time.monotonic() + 300,
    )

    assert move == {"action": "UP"}
    assert analyzer.model_calls == 2
    transcript = (tmp_path / "preflight_artifacts/transcript.txt").read_text()
    assert "yield_seconds: disabled" in transcript


def test_transport_retry_counts_toward_cap():
    post = Mock(side_effect=requests.ConnectionError("test transport failure"))
    client = evaluation.ModelClient("http://unused", {}, deadline=True, post=post, accepted_move_cap=2)
    events = []

    with pytest.raises(evaluation.InferenceFailure, match="two attempts failed"):
        client.deadline_completion({}, events.append)
    with pytest.raises(move_budget.NoGameProgress, match="2 model calls"):
        client.deadline_completion({}, events.append)

    assert post.call_count == client.model_calls == 2
    assert [event["stop_reason"] for event in events] == ["transport_error", "transport_error"]


@pytest.mark.parametrize("deadline", [False, True])
def test_request_stops_at_five_minute_watchdog(monkeypatch, deadline):
    clock = SimpleNamespace(now=0.0)
    fake_time = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr(move_budget, "time", fake_time)
    monkeypatch.setattr(evaluation, "time", fake_time)

    def post(_url, *, json, timeout, headers=None):
        assert timeout == 3.0
        if deadline:
            assert float(headers[evaluation.DEADLINE_HEADER]) == 300.0
        clock.now += timeout
        raise requests.Timeout("test inference timed out")

    client = evaluation.ModelClient("http://unused", {}, deadline=deadline, post=post)
    clock.now = 297.0
    with pytest.raises(move_budget.NoGameProgress, match="no accepted move for 300 seconds"):
        if deadline:
            client.deadline_completion({}, lambda event: None)
        else:
            client.legacy_completion({}, None)
    assert client.model_calls == 1


def test_automatic_reset_refreshes_tool_state_without_renewing_watchdog(duck, monkeypatch, tmp_path):
    replies = iter([tool_reply("act"), tool_reply("inspect"), tool_reply("act")])
    client = evaluation.ModelClient(
        "http://unused", duck.config, post=lambda *args, **kwargs: next(replies), accepted_move_cap=2,
    )
    analyzer = duck.policy(client, "http://unused")
    game = OfflineGame(duck.engine, "reset", lose_first_move=True)
    refresh = Mock(wraps=analyzer.refresh_after_reset)
    renew = Mock(wraps=client.move_budget.accepted_move)
    monkeypatch.setattr(analyzer, "refresh_after_reset", refresh)
    monkeypatch.setattr(client.move_budget, "accepted_move", renew)

    def inspect_reset(**kwargs):
        if kwargs["code"] == "inspect":
            state = kwargs["initial_state"]
            assert state["history"][-1]["action"] == "RESET"
            assert state["last_action_result"]["action_display"] == "RESET"
            assert state["last_action_result"]["state"] == "NOT_FINISHED"
            assert not state["last_action_result"]["game_over"]
            assert not analyzer._last_step_summary["game_over"]
            assert renew.call_count == 1
        return sandbox(**kwargs)

    inspection = Mock(side_effect=inspect_reset)
    monkeypatch.setattr(duck.tool_agent, "run_sandboxed_python", inspection)
    run_games(duck, tmp_path, [game], [analyzer])

    assert game.game_run.state == "won", game.game_run.solver_note
    assert [call.kwargs["code"] for call in inspection.call_args_list] == ["act", "inspect", "act"]
    assert game.game_run.history == [duck.engine.GameAction.ACTION1, duck.engine.GameAction.RESET,
                                     duck.engine.GameAction.ACTION1]
    assert refresh.call_count == 1
    assert renew.call_count == 2


def test_trim_preserves_current_task_and_latest_tool_exchange(duck):
    analyzer = duck.policy(evaluation.ModelClient("http://unused", duck.config), "http://unused")
    system = {"role": "system", "content": "Solve the puzzle."}
    task = analyzer._build_user_message("Current task and board.", None)
    prior_task = {"role": "user", "content": "Old task."}
    old_exchange = tool_exchange("old", "Old output." * 500)
    assistant, latest = tool_exchange("latest", "New output." * 500)
    extra_assistant, extra = tool_exchange("extra", "Other output." * 500)
    assistant["tool_calls"].extend(extra_assistant["tool_calls"])
    analyzer._context_budget_tokens = 250
    messages = [system, prior_task, *old_exchange, task, *old_exchange, assistant, latest, extra]

    trimmed = analyzer._trim_messages_for_context(messages)

    assert trimmed[:3] == [system, task, assistant]
    assert [message["tool_call_id"] for message in trimmed[3:]] == ["latest", "extra"]
    assert trimmed[3]["content"].startswith("New output.")
    assert len(trimmed[3]["content"]) < len(latest["content"])
    assert analyzer._estimate_request_input_tokens(trimmed) <= analyzer._context_budget_tokens


def test_oversized_task_fails_before_any_model_call(duck, tmp_path):
    from inference.agent.runtime_state import Frame, write_runtime_state

    model = Mock()
    analyzer = duck.policy(evaluation.ModelClient("http://unused", duck.config, post=model), "http://unused")
    analyzer._context_budget_tokens = 1
    state_path = tmp_path / "state.json"
    write_runtime_state(state_path, current_frame=Frame(((0,),), 0, 1), history=[])

    with pytest.raises(RuntimeError, match="context_budget_exhausted"):
        analyzer.analyze(
            state_path, 0, valid_actions=["ACTION1"], step_env=Mock(), transcript_path=tmp_path / "transcript.txt",
            analysis_step=1, request_timeout_seconds=100, should_stop=lambda: False,
        )

    model.assert_not_called()
