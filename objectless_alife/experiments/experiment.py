"""Multi-phase experiment orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from objectless_alife.config.constants import MAX_EXPERIMENT_WORK_UNITS
from objectless_alife.config.types import ExperimentConfig, SimulationResult
from objectless_alife.experiments.summaries import (
    _build_phase_comparison,
    _build_phase_summary,
    collect_final_metric_rows,
)
from objectless_alife.io.schemas import AGGREGATE_SCHEMA_VERSION
from objectless_alife.simulation.engine import run_batch_search

_EXPERIMENT_METRIC_COLUMNS = [
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


def run_experiment(config: ExperimentConfig) -> list[SimulationResult]:
    """Run multi-phase, multi-seed experiments and persist aggregate artifacts."""
    if config.n_rules < 1:
        raise ValueError("n_rules must be >= 1")
    if config.n_seed_batches < 1:
        raise ValueError("n_seed_batches must be >= 1")
    if not config.phases:
        raise ValueError("phases must not be empty")
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root_logs_dir = out_dir / "logs"
    root_logs_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[SimulationResult] = []
    experiment_rows: list[dict[str, int | str | bool | None]] = []
    phase_summaries: list[dict[str, int | float | None]] = []

    total_work_units = len(config.phases) * config.n_rules * config.n_seed_batches * config.steps
    if total_work_units > MAX_EXPERIMENT_WORK_UNITS:
        raise ValueError(
            "experiment workload exceeds safety threshold; reduce phases/n_rules/seed-batches/steps"
        )

    total_rules_per_phase = config.n_rules * config.n_seed_batches

    for phase in config.phases:
        phase_out_dir = out_dir / f"phase_{phase.value}"
        phase_out_dir.mkdir(parents=True, exist_ok=True)
        phase_search_config = config.resolved_search_config()
        phase_results = run_batch_search(
            n_rules=total_rules_per_phase,
            phase=phase,
            out_dir=phase_out_dir,
            base_rule_seed=config.rule_seed_start,
            base_sim_seed=config.sim_seed_start,
            config=phase_search_config,
        )
        all_results.extend(phase_results)

        current_phase_run_rows: list[dict[str, int | str | bool | None]] = []
        for i, result in enumerate(phase_results):
            seed_batch = i // config.n_rules
            rule_seed = config.rule_seed_start + i
            sim_seed = config.sim_seed_start + i
            current_phase_run_rows.append(
                {
                    "schema_version": AGGREGATE_SCHEMA_VERSION,
                    "rule_id": result.rule_id,
                    "phase": phase.value,
                    "seed_batch": seed_batch,
                    "rule_seed": rule_seed,
                    "sim_seed": sim_seed,
                    "survived": result.survived,
                    "termination_reason": result.termination_reason,
                    "terminated_at": result.terminated_at,
                }
            )
        experiment_rows.extend(current_phase_run_rows)

        metrics_path = phase_out_dir / "logs" / "metrics_summary.parquet"
        final_metric_rows = collect_final_metric_rows(
            metrics_path=metrics_path,
            metric_columns=_EXPERIMENT_METRIC_COLUMNS,
            phase_results=phase_results,
            default_final_step=config.steps - 1,
        )

        phase_summaries.append(
            _build_phase_summary(
                phase=phase,
                run_rows=current_phase_run_rows,
                final_metric_rows=final_metric_rows,
            )
        )

    pq.write_table(pa.Table.from_pylist(experiment_rows), root_logs_dir / "experiment_runs.parquet")
    pq.write_table(pa.Table.from_pylist(phase_summaries), root_logs_dir / "phase_summary.parquet")
    phase_comparison = _build_phase_comparison(phase_summaries)
    (root_logs_dir / "phase_comparison.json").write_text(
        json.dumps(phase_comparison, ensure_ascii=False, indent=2)
    )

    return all_results
