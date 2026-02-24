"""Simulation engine: batch search, per-step metrics, and Parquet persistence."""

from objectless_alife.simulation.engine import run_batch_search
from objectless_alife.simulation.persistence import flush_sim_columns
from objectless_alife.simulation.step import (
    _compute_step_metrics as _compute_step_metrics,
)
from objectless_alife.simulation.step import (
    _entropy_from_action_counts as _entropy_from_action_counts,
)
from objectless_alife.simulation.step import (
    _mean_and_pvariance as _mean_and_pvariance,
)
from objectless_alife.simulation.step import (
    compute_step_metrics,
    entropy_from_action_counts,
    mean_and_pvariance,
)

__all__ = [
    "compute_step_metrics",
    "entropy_from_action_counts",
    "flush_sim_columns",
    "mean_and_pvariance",
    "run_batch_search",
]
