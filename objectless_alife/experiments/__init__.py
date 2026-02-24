"""Experiments layer: search, orchestration, density sweeps, and robustness."""

from objectless_alife.experiments.density_sweep import run_density_sweep
from objectless_alife.experiments.experiment import run_experiment
from objectless_alife.experiments.robustness import run_halt_window_sweep, run_multi_seed_robustness
from objectless_alife.experiments.selection import select_top_rules_by_delta_mi
from objectless_alife.experiments.summaries import (
    collect_final_metric_rows,
)

__all__ = [
    "collect_final_metric_rows",
    "run_density_sweep",
    "run_experiment",
    "run_halt_window_sweep",
    "run_multi_seed_robustness",
    "select_top_rules_by_delta_mi",
]
