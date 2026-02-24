"""Per-step metric computation helpers for the simulation engine."""

from __future__ import annotations

import math

from objectless_alife.metrics.information import compression_ratio, neighbor_mutual_information
from objectless_alife.metrics.spatial import (
    cluster_count_by_state,
    morans_i_occupied,
    same_state_adjacency_fraction,
)


def entropy_from_action_counts(action_counts: list[int], total_actions: int) -> float:
    """Compute Shannon entropy from pre-aggregated action counts."""
    if total_actions < 1:
        return 0.0
    entropy = 0.0
    for count in action_counts:
        if count == 0:
            continue
        p = count / total_actions
        entropy -= p * math.log2(p)
    return entropy


def mean_and_pvariance(values: list[float]) -> tuple[float, float]:
    """Return mean and population variance for non-empty values."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance


def compute_step_metrics(
    *,
    snapshot: tuple[tuple[int, int, int, int], ...],
    snapshot_bytes: bytes,
    step_entropy: float,
    predictability: float | None,
    running_phase_transition_delta: float,
    action_entropy_mean: float,
    action_entropy_var: float,
    block_ncd_value: float | None,
    grid_width: int,
    grid_height: int,
) -> dict[str, float | int | None]:
    """Compute per-step metric values for a single simulation step."""
    return {
        "state_entropy": step_entropy,
        "compression_ratio": compression_ratio(snapshot_bytes),
        "predictability_hamming": predictability,
        "morans_i": morans_i_occupied(snapshot, grid_width=grid_width, grid_height=grid_height),
        "cluster_count": cluster_count_by_state(
            snapshot, grid_width=grid_width, grid_height=grid_height
        ),
        "same_state_adjacency_fraction": same_state_adjacency_fraction(
            snapshot, grid_width=grid_width, grid_height=grid_height
        ),
        "phase_transition_max_delta": running_phase_transition_delta,
        "neighbor_mutual_information": neighbor_mutual_information(
            snapshot, grid_width=grid_width, grid_height=grid_height
        ),
        "action_entropy_mean": action_entropy_mean,
        "action_entropy_variance": action_entropy_var,
        "block_ncd": block_ncd_value,
    }


# Private aliases for backward compatibility
_entropy_from_action_counts = entropy_from_action_counts
_mean_and_pvariance = mean_and_pvariance
_compute_step_metrics = compute_step_metrics
