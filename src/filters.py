from __future__ import annotations

from enum import Enum
from typing import Sequence


class TerminationReason(str, Enum):
    HALT = "halt"
    STATE_UNIFORM = "state_uniform"


class HaltDetector:
    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._last_snapshot: tuple[tuple[int, int, int, int], ...] | None = None
        self._unchanged_count = 0

    def observe(self, snapshot: tuple[tuple[int, int, int, int], ...]) -> bool:
        if self._last_snapshot is None:
            self._last_snapshot = snapshot
            return False

        if snapshot == self._last_snapshot:
            self._unchanged_count += 1
        else:
            self._unchanged_count = 0
            self._last_snapshot = snapshot

        return self._unchanged_count >= self.window


class StateUniformDetector:
    def observe(self, states: Sequence[int]) -> bool:
        if not states:
            return False
        return len(set(states)) == 1
