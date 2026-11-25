"""Prepare native Tufa game records, then open live sessions at play time."""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

from arcengine import FrameDataRaw, GameAction
from taaf.game import GameState
from taaf.game_api import GameAPI


_reset_lock = Lock()


@dataclass
class ReportingGame(GameAPI):
    opening_frame: FrameDataRaw = field(kw_only=True)
    human_actions: list[int] | None = field(kw_only=True)
    title: str = field(kw_only=True)
    level_retries: dict[int, int] = field(default_factory=dict, init=False)
    on_finished = None

    def _start_game(self, session):
        self._scheduled_session = session
        self.game_id = self.env_name
        self.number_of_levels = self.opening_frame.win_levels
        self.base_actions_per_level = self.human_actions
        self.grid_size = (64, 64)
        return GameState(raw=self.opening_frame)

    def open_for_play(self):
        with _reset_lock:
            # The SDK uses this variable when opening local games as well
            # as on retries. A fresh game must register as a new play.
            previous = os.environ.pop("ONLY_RESET_LEVELS", None)
            try:
                initial = super()._start_game(self._scheduled_session)
            except Exception:
                if previous is not None:
                    os.environ["ONLY_RESET_LEVELS"] = previous
                raise
        run = self.game_run
        if (self.game_id, self.number_of_levels) != (run.game_id, run.number_of_levels):
            raise RuntimeError(f"{self.env_name} changed since its availability check")
        run.base_actions_per_level = self.base_actions_per_level
        self._current_state = initial
        if run.record_intermediate_states:
            run.intermediate_states[0] = initial
        self.begin_timing()

    def begin_timing(self):
        self.game_run.started_at = datetime.now()
        self.game_run.started_at_monotonic = time.monotonic()

    def _execute_action(self, action):
        if action.id == GameAction.RESET:
            with _reset_lock:
                return super()._execute_action(action)
        return super()._execute_action(action)

    def finish_game(self, generated_tokens=0, uncached_input_tokens=0):
        super().finish_game(generated_tokens, uncached_input_tokens)
        callback = self.on_finished
        if callback is not None:
            self.on_finished = None
            callback(self.game_run)
