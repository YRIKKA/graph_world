"""Bound inference between accepted environment moves."""

import time

DEFAULT_ACCEPTED_MOVE_CAP = 16
ACCEPTED_MOVE_TIMEOUT = 300


class NoGameProgress(RuntimeError):
    """The policy used its inference budget without an accepted move."""


class AcceptedMoveBudget:
    def __init__(self, call_limit):
        self.call_limit = call_limit
        self.accepted_move()

    def accepted_move(self):
        self.calls = 0
        self.deadline = time.monotonic() + ACCEPTED_MOVE_TIMEOUT

    def time_expired(self):
        return time.monotonic() >= self.deadline

    def check_time(self):
        if self.time_expired():
            raise NoGameProgress(f"no_game_progress: no accepted move for {ACCEPTED_MOVE_TIMEOUT} seconds")

    def check(self):
        self.check_time()
        if self.calls >= self.call_limit:
            raise NoGameProgress(f"no_game_progress: {self.call_limit} model calls without an accepted move")

    def begin_call(self):
        self.check()
        self.calls += 1
