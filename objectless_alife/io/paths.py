"""Path construction helpers for simulation output directories.

Centralises the directory/file naming conventions used throughout the
simulation engine and experiment orchestration layers.
"""

from __future__ import annotations

from pathlib import Path


def resolve_within_base(path: Path, base_dir: Path) -> Path:
    """Resolve *path* and ensure it stays within the trusted *base_dir*.

    Raises :exc:`ValueError` if the resolved path escapes the base directory.
    """
    candidate = path if path.is_absolute() else base_dir / path
    resolved = candidate.resolve()
    base_resolved = base_dir.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise ValueError(f"Path escapes base_dir: {path}")
    return resolved


def rules_dir(out_dir: Path) -> Path:
    """Return path to the rules subdirectory within an output directory."""
    return out_dir / "rules"


def logs_dir(out_dir: Path) -> Path:
    """Return path to the logs subdirectory within an output directory."""
    return out_dir / "logs"


def simulation_log_path(out_dir: Path) -> Path:
    """Return path to the simulation log Parquet file."""
    return logs_dir(out_dir) / "simulation_log.parquet"


def metrics_summary_path(out_dir: Path) -> Path:
    """Return path to the metrics summary Parquet file."""
    return logs_dir(out_dir) / "metrics_summary.parquet"


def experiment_runs_path(out_dir: Path) -> Path:
    """Return path to the experiment runs Parquet file."""
    return logs_dir(out_dir) / "experiment_runs.parquet"


def phase_summary_path(out_dir: Path) -> Path:
    """Return path to the phase summary Parquet file."""
    return logs_dir(out_dir) / "phase_summary.parquet"


def phase_comparison_path(out_dir: Path) -> Path:
    """Return path to the phase comparison JSON file."""
    return logs_dir(out_dir) / "phase_comparison.json"


def density_sweep_runs_path(out_dir: Path) -> Path:
    """Return path to the density sweep runs Parquet file."""
    return logs_dir(out_dir) / "density_sweep_runs.parquet"


def density_phase_summary_path(out_dir: Path) -> Path:
    """Return path to the density phase summary Parquet file."""
    return logs_dir(out_dir) / "density_phase_summary.parquet"


def density_phase_comparison_path(out_dir: Path) -> Path:
    """Return path to the density phase comparison Parquet file."""
    return logs_dir(out_dir) / "density_phase_comparison.parquet"


def multi_seed_results_path(out_dir: Path) -> Path:
    """Return path to the multi-seed results Parquet file."""
    return logs_dir(out_dir) / "multi_seed_results.parquet"


def halt_window_sweep_path(out_dir: Path) -> Path:
    """Return path to the halt-window sweep Parquet file."""
    return logs_dir(out_dir) / "halt_window_sweep.parquet"


def phase_out_dir(out_dir: Path, phase_value: int) -> Path:
    """Return path to the per-phase output subdirectory."""
    return out_dir / f"phase_{phase_value}"


def density_out_dir(out_dir: Path, grid_width: int, grid_height: int, num_agents: int) -> Path:
    """Return path to the per-density-point output subdirectory."""
    return out_dir / f"density_w{grid_width}_h{grid_height}_a{num_agents}"
