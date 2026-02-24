"""Temporal metrics: phase transitions, periodicity, and action entropy."""

from __future__ import annotations

import statistics
from typing import Sequence

from objectless_alife.metrics.information import state_entropy


def quasi_periodicity_peak_count(series: Sequence[float]) -> int:
    """Count local maxima in positive-lag autocorrelation."""
    n = len(series)
    if n < 4:
        return 0
    mean = sum(series) / n
    centered = [v - mean for v in series]
    denom = sum(v * v for v in centered)
    if denom == 0.0:
        return 0

    ac = []
    for lag in range(1, (n // 2) + 1):
        num = sum(centered[t] * centered[t - lag] for t in range(lag, n))
        ac.append(num / denom)

    if len(ac) < 3:
        return 0

    threshold = 0.1
    peaks = 0
    for i in range(1, len(ac) - 1):
        if ac[i] > threshold and ac[i] > ac[i - 1] and ac[i] > ac[i + 1]:
            peaks += 1
    return peaks


def phase_transition_max_delta(series: Sequence[float]) -> float:
    """Compute max absolute first-difference in a time series."""
    if len(series) < 2:
        return 0.0
    return max(abs(curr - prev) for prev, curr in zip(series, series[1:], strict=False))


def action_entropy(actions: Sequence[int]) -> float:
    """Compute Shannon entropy for an action sequence."""
    return state_entropy(actions)


def action_entropy_variance(per_agent_actions: Sequence[Sequence[int]]) -> float:
    """Compute variance of per-agent action entropy values."""
    if not per_agent_actions:
        return 0.0
    entropies = [action_entropy(actions) for actions in per_agent_actions]
    if len(entropies) < 2:
        return 0.0
    return statistics.pvariance(entropies)
