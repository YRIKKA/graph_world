"""Add level budgets to the pinned Tufa session without changing its play loop."""

import time
from dataclasses import dataclass

from arcengine import GameAction, GameState
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
from inference.agent.tool_agent import AnalyzerTurnResult
from inference.framework.solver import HarnessSolver, _HarnessGameSession


class TimedAnalyzer:
    def __init__(self, session, analyzer):
        self.session = session
        self.analyzer = analyzer

    @property
    def generated_tokens(self):
        return self.analyzer.generated_tokens

    def analyze(self, *args, **kwargs):
        session = self.session
        if session.should_stop():
            return AnalyzerTurnResult(step_executed=False)
        if session.level_remaining() <= 0:
            session.restart_level("level_time_limit")
            return AnalyzerTurnResult(step_executed=False)
        session.analyzing = True
        budget_expired = False
        try:
            result = self.analyzer.analyze(*args, **kwargs)
        except Exception as error:
            budget_expired = getattr(error, "budget_limited", False)
            if not budget_expired and session.level_remaining() > 0:
                raise
            result = AnalyzerTurnResult(step_executed=False)
        finally:
            session.analyzing = False
        if not session.should_stop() and (budget_expired or session.level_remaining() <= 0):
            session.restart_level("level_time_limit")
            return AnalyzerTurnResult(step_executed=False)
        return result


class LevelSession(_HarnessGameSession):
    analyzing = False
    level_has_moves = False

    def play(self):
        self.level_started_at = time.monotonic()
        self.analyzer = TimedAnalyzer(self, self.analyzer)
        super().play()

    def level_remaining(self):
        return self.solver.level_timeout - (time.monotonic() - self.level_started_at)

    def should_stop(self):
        return super().should_stop() or (self.analyzing and self.level_remaining() <= 0)

    def request_timeout_seconds(self):
        job_remaining = super().request_timeout_seconds()
        remaining = max(0.001, self.level_remaining())
        return remaining if job_remaining is None else min(remaining, job_remaining)

    def timing_payload(self):
        return {
            **super().timing_payload(),
            "time_remaining_seconds": max(0.0, self.level_remaining()),
        }

    def _execute_action(self, action, **kwargs):
        if self.analyzing and action.id == GameAction.RESET:
            raise ValueError("RESET is managed by the level recovery budget")
        completed = self.game.current_state.levels_completed
        payload = super()._execute_action(action, **kwargs)
        if self.game.current_state.levels_completed > completed:
            self.level_started_at = time.monotonic()
            self.level_has_moves = False
            run = self.game.game_run
            baselines = run.base_actions_per_level
            human_actions = baselines[completed] if baselines else "unavailable"
            partial_rhae = f"{run._compute_final_score():.2f}%" if baselines else "unavailable"
            message = (
                f"{self.solver.label} · {run.game_id} · pass {self.pass_index + 1} · "
                f"cleared level {completed + 1}/{run.number_of_levels} · "
                f"Tufa actions {run.actions_per_level[completed]} (human {human_actions}) · "
                f"partial RHAE {partial_rhae}"
            )
            print(message, flush=True)
        else:
            self.level_has_moves = action.id != GameAction.RESET
        return payload

    def _execute_auto_reset(self):
        self.restart_level("game_over")

    def restart_level(self, reason):
        before = self.game.current_state.raw
        level = before.levels_completed + 1
        used = self.game.level_retries.get(level, 0)
        limit = self.solver.bounded_recovery
        if used >= limit:
            raise RuntimeError(f"{reason}: level {level} exhausted after {used}/{limit} retries")
        elapsed = time.monotonic() - self.level_started_at
        # With no moves the level is already at its starting board. A wire
        # RESET at that point can mean a full-game reset on the hosted API.
        if self.level_has_moves or reason == "game_over":
            super()._execute_auto_reset()
            after = self.game.current_state.raw
            if (after.guid != before.guid or after.levels_completed != before.levels_completed
                    or after.full_reset or after.state is not GameState.NOT_FINISHED):
                raise RuntimeError(f"{reason}: ARC did not restart level {level} in the same game")
        self.level_started_at = time.monotonic()
        self.game.level_retries[level] = used + 1
        message = (f"{self.game.game_id} · pass {self.pass_index + 1} · level {level} · "
                   f"retry {used + 1}/{limit} · {reason} · previous attempt {elapsed:.2f}s")
        print(message, flush=True)
        with self.transcript_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


@dataclass
class LevelBudgetSolver(HarnessSolver):
    level_timeout: int = 7200
    bounded_recovery: int = 2

    def _play_one(self, game, index, pass_index, local_server=None):
        if self._stop_event.is_set():
            return
        try:
            analyzer = self._make_analyzer(game, index, local_server)
            stem = self._run_stem(game.game_run.game_id, pass_index)
            LevelSession(
                solver=self, game=game, analyzer=analyzer, game_index=index, pass_index=pass_index,
                state_path=self._artifacts_dir() / f"{stem}_{RUNTIME_STATE_FILENAME}",
                transcript_path=self._transcripts_dir() / f"{stem}.txt",
                analysis_html_relpath=f"solver_analysis/{stem}.html", stop_event=self._stop_event,
                viewer_data_path=self._artifacts_dir() / f"{stem}_viewer_data.json",
            ).play()
        except Exception as error:
            self._finish_after_error(game, error)
