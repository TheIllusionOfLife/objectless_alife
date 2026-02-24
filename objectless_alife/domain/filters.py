"""Termination and dynamic filter detectors for simulation runs.

Each detector is a stateful observer that returns True once its trigger
condition is met.  Detectors are consumed by the simulation engine to decide
whether a run should terminate early.
"""

from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Sequence

from objectless_alife.config.constants import ACTION_SPACE_SIZE


class TerminationReason(str, Enum):
    """Termination reason labels persisted in run metadata."""

    HALT = "halt"
    STATE_UNIFORM = "state_uniform"
    SHORT_PERIOD = "short_period"
    LOW_ACTIVITY = "low_activity"


class HaltDetector:
    """Detect N consecutive unchanged snapshots."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._last_snapshot: tuple[tuple[int, int, int, int], ...] | None = None
        self._unchanged_count = 0

    def observe(self, snapshot: tuple[tuple[int, int, int, int], ...]) -> bool:
        """Return True once snapshot has remained unchanged for `window` checks."""
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
    """Detect whether all agents currently share the same internal state."""

    def observe(self, states: Sequence[int]) -> bool:
        """Return True only when all states are equal and input is non-empty."""
        if not states:
            return False
        return len(set(states)) == 1


class ShortPeriodDetector:
    """Detect short periodic loops in recent snapshots."""

    def __init__(self, max_period: int, history_size: int) -> None:
        if max_period < 1:
            raise ValueError("max_period must be >= 1")
        if history_size < max_period * 2:
            raise ValueError("history_size must be at least 2 * max_period")
        self.max_period = max_period
        self.history_size = history_size
        self._history: list[tuple[tuple[int, int, int, int], ...]] = []

    def observe(self, snapshot: tuple[tuple[int, int, int, int], ...]) -> bool:
        """Return True when current history matches a recent short cycle."""
        self._history.append(snapshot)
        if len(self._history) > self.history_size:
            self._history.pop(0)

        for period in range(1, self.max_period + 1):
            if len(self._history) < period * 2:
                continue
            recent = self._history[-period:]
            prev = self._history[-2 * period : -period]
            if recent == prev:
                return True
        return False


class LowActivityDetector:
    """Detect low activity from intended-action diversity over a rolling window."""

    def __init__(self, window: int, min_unique_ratio: float) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if not isfinite(min_unique_ratio) or not 0.0 <= min_unique_ratio <= 1.0:
            raise ValueError("min_unique_ratio must be in [0.0, 1.0]")
        self.window = window
        self.min_unique_ratio = min_unique_ratio
        self._recent: list[Sequence[int]] = []

    def observe(self, actions: Sequence[int]) -> bool:
        """Return True when unique-action ratio stays below threshold across window."""
        self._recent.append(tuple(actions))
        if len(self._recent) < self.window:
            return False
        if len(self._recent) > self.window:
            self._recent.pop(0)

        flattened = [action for step_actions in self._recent for action in step_actions]
        if not flattened:
            return False
        unique_ratio = len(set(flattened)) / ACTION_SPACE_SIZE
        return unique_ratio < self.min_unique_ratio
