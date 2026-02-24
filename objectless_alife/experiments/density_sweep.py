"""Density sweep orchestration across grid/agent configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from objectless_alife.config.constants import MAX_EXPERIMENT_WORK_UNITS
from objectless_alife.config.types import DensitySweepConfig, SearchConfig, SimulationResult
from objectless_alife.domain.rules import ObservationPhase
from objectless_alife.domain.world import WorldConfig
from objectless_alife.experiments.summaries import (
    _build_phase_comparison,
    _build_phase_summary,
    collect_final_metric_rows,
)
from objectless_alife.io.schemas import (
    DENSITY_PHASE_COMPARISON_SCHEMA,
    DENSITY_PHASE_SUMMARY_SCHEMA,
    DENSITY_SWEEP_RUNS_SCHEMA,
    DENSITY_SWEEP_SCHEMA_VERSION,
)
from objectless_alife.simulation.engine import run_batch_search

_DENSITY_METRIC_COLUMNS = [
    "rule_id",
    "step",
    "state_entropy",
    "compression_ratio",
    "predictability_hamming",
    "morans_i",
    "cluster_count",
    "same_state_adjacency_fraction",
    "neighbor_mutual_information",
    "quasi_periodicity_peaks",
    "phase_transition_max_delta",
    "action_entropy_mean",
    "action_entropy_variance",
    "block_ncd",
    "mi_shuffle_null",
]


def _validate_density_sweep_config(config: DensitySweepConfig) -> None:
    """Fail fast when density sweep configuration is structurally invalid."""
    if config.n_rules < 1:
        raise ValueError("n_rules must be >= 1")
    if config.n_seed_batches < 1:
        raise ValueError("n_seed_batches must be >= 1")
    if config.steps < 1:
        raise ValueError("steps must be >= 1")
    if not config.grid_sizes:
        raise ValueError("grid_sizes must not be empty")
    if not config.agent_counts:
        raise ValueError("agent_counts must not be empty")

    density_points = len(config.grid_sizes) * len(config.agent_counts)
    total_work_units = density_points * 2 * config.n_rules * config.n_seed_batches * config.steps
    if total_work_units > MAX_EXPERIMENT_WORK_UNITS:
        raise ValueError(
            "density sweep workload exceeds safety threshold; reduce grid-sizes/agent-counts/"
            "n-rules/seed-batches/steps"
        )


def _density_search_config(config: DensitySweepConfig) -> SearchConfig:
    """Convert shared density sweep options into SearchConfig."""
    return config.resolved_search_config()


def _make_density_phase_summary_rows_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build density phase summary table using static schema."""
    return pa.Table.from_pylist(rows, schema=DENSITY_PHASE_SUMMARY_SCHEMA)


def _run_density_phase(
    *,
    config: DensitySweepConfig,
    phase: ObservationPhase,
    phase_out_dir: Path,
    grid_width: int,
    grid_height: int,
    num_agents: int,
    density_ratio: float,
    total_rules_per_phase: int,
) -> tuple[list[SimulationResult], list[dict[str, Any]], dict[str, Any]]:
    """Run one phase for a single density point and return aggregates."""
    phase_search_config = _density_search_config(config)
    phase_world_config = WorldConfig(
        grid_width=grid_width,
        grid_height=grid_height,
        num_agents=num_agents,
        steps=config.steps,
    )
    phase_results = run_batch_search(
        n_rules=total_rules_per_phase,
        phase=phase,
        out_dir=phase_out_dir,
        base_rule_seed=config.rule_seed_start,
        base_sim_seed=config.sim_seed_start,
        world_config=phase_world_config,
        config=phase_search_config,
    )

    current_phase_run_rows: list[dict[str, Any]] = []
    for i, result in enumerate(phase_results):
        seed_batch = i // config.n_rules
        rule_seed = config.rule_seed_start + i
        sim_seed = config.sim_seed_start + i
        current_phase_run_rows.append(
            {
                "schema_version": DENSITY_SWEEP_SCHEMA_VERSION,
                "rule_id": result.rule_id,
                "phase": phase.value,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "num_agents": num_agents,
                "density_ratio": density_ratio,
                "seed_batch": seed_batch,
                "rule_seed": rule_seed,
                "sim_seed": sim_seed,
                "survived": result.survived,
                "termination_reason": result.termination_reason,
                "terminated_at": result.terminated_at,
            }
        )

    metrics_path = phase_out_dir / "logs" / "metrics_summary.parquet"
    final_metric_rows = collect_final_metric_rows(
        metrics_path=metrics_path,
        metric_columns=_DENSITY_METRIC_COLUMNS,
        phase_results=phase_results,
        default_final_step=config.steps - 1,
    )
    base_summary = _build_phase_summary(
        phase=phase,
        run_rows=current_phase_run_rows,
        final_metric_rows=final_metric_rows,
    )
    summary_row = {
        **base_summary,
        "schema_version": DENSITY_SWEEP_SCHEMA_VERSION,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "num_agents": num_agents,
        "density_ratio": density_ratio,
    }
    return phase_results, current_phase_run_rows, summary_row


def _append_density_phase_comparison_rows(
    comparison_rows: list[dict[str, Any]],
    per_density_phase_summaries: list[dict[str, Any]],
    *,
    grid_width: int,
    grid_height: int,
    num_agents: int,
    density_ratio: float,
) -> None:
    """Append comparison rows for one density point."""
    comparison_payload = _build_phase_comparison(per_density_phase_summaries)
    base_phase = comparison_payload["phases"][0]
    target_phase = comparison_payload["phases"][1]
    for metric, deltas in comparison_payload["deltas"].items():
        comparison_rows.append(
            {
                "schema_version": DENSITY_SWEEP_SCHEMA_VERSION,
                "base_phase": base_phase,
                "target_phase": target_phase,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "num_agents": num_agents,
                "density_ratio": density_ratio,
                "metric": metric,
                "delta_absolute": deltas["absolute"],
                "delta_relative": deltas["relative"],
            }
        )


def run_density_sweep(config: DensitySweepConfig) -> list[SimulationResult]:
    """Run explicit grid/agent sweeps across both observation phases."""
    _validate_density_sweep_config(config)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root_logs_dir = out_dir / "logs"
    root_logs_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[SimulationResult] = []
    sweep_rows: list[dict[str, Any]] = []
    density_phase_summary_rows: list[dict[str, Any]] = []
    density_phase_comparison_rows: list[dict[str, Any]] = []

    total_rules_per_phase = config.n_rules * config.n_seed_batches
    phases = (ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE)

    for grid_width, grid_height in config.grid_sizes:
        if grid_width < 1 or grid_height < 1:
            raise ValueError("grid dimensions must be >= 1")
        max_cells = grid_width * grid_height
        for num_agents in config.agent_counts:
            if num_agents > max_cells:
                raise ValueError(
                    f"num_agents ({num_agents}) cannot exceed grid cells ({max_cells}) for "
                    f"{grid_width}x{grid_height}"
                )
            density_ratio = num_agents / max_cells
            per_density_phase_summaries: list[dict[str, Any]] = []

            for phase in phases:
                phase_out_dir = (
                    out_dir
                    / f"density_w{grid_width}_h{grid_height}_a{num_agents}"
                    / f"phase_{phase.value}"
                )
                phase_out_dir.mkdir(parents=True, exist_ok=True)

                phase_results, current_phase_run_rows, summary_row = _run_density_phase(
                    config=config,
                    phase=phase,
                    phase_out_dir=phase_out_dir,
                    grid_width=grid_width,
                    grid_height=grid_height,
                    num_agents=num_agents,
                    density_ratio=density_ratio,
                    total_rules_per_phase=total_rules_per_phase,
                )
                all_results.extend(phase_results)
                sweep_rows.extend(current_phase_run_rows)
                density_phase_summary_rows.append(summary_row)
                per_density_phase_summaries.append(summary_row)

            _append_density_phase_comparison_rows(
                density_phase_comparison_rows,
                per_density_phase_summaries,
                grid_width=grid_width,
                grid_height=grid_height,
                num_agents=num_agents,
                density_ratio=density_ratio,
            )

    pq.write_table(
        pa.Table.from_pylist(sweep_rows, schema=DENSITY_SWEEP_RUNS_SCHEMA),
        root_logs_dir / "density_sweep_runs.parquet",
    )
    pq.write_table(
        _make_density_phase_summary_rows_table(density_phase_summary_rows),
        root_logs_dir / "density_phase_summary.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            density_phase_comparison_rows,
            schema=DENSITY_PHASE_COMPARISON_SCHEMA,
        ),
        root_logs_dir / "density_phase_comparison.parquet",
    )

    return all_results
