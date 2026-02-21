"""Experiment orchestration and metric aggregation helpers.

Contains higher-level experiment drivers (``run_experiment``,
``run_density_sweep``, ``run_multi_seed_robustness``,
``run_halt_window_sweep``, ``select_top_rules_by_delta_mi``) and the
internal aggregation helpers they share.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from objectless_alife.config import (
    MAX_EXPERIMENT_WORK_UNITS,
    DensitySweepConfig,
    ExperimentConfig,
    HaltWindowSweepConfig,
    MultiSeedConfig,
    SearchConfig,
    SimulationResult,
    StateUniformMode,
    UpdateMode,
)
from objectless_alife.filters import HaltDetector, StateUniformDetector, TerminationReason
from objectless_alife.metrics import (
    neighbor_mutual_information,
    neighbor_pair_count,
    same_state_adjacency_fraction,
    shuffle_null_mi,
)
from objectless_alife.rules import ObservationPhase, generate_rule_table
from objectless_alife.schemas import (
    AGGREGATE_SCHEMA_VERSION,
    DENSITY_PHASE_COMPARISON_SCHEMA,
    DENSITY_PHASE_SUMMARY_SCHEMA,
    DENSITY_SWEEP_RUNS_SCHEMA,
    DENSITY_SWEEP_SCHEMA_VERSION,
    HALT_WINDOW_SWEEP_SCHEMA,
    MULTI_SEED_SCHEMA,
    PHASE_SUMMARY_METRIC_NAMES,
)
from objectless_alife.simulation import run_batch_search
from objectless_alife.world import World, WorldConfig

# ---------------------------------------------------------------------------
# Internal helpers — percentile / float extraction / aggregation
# ---------------------------------------------------------------------------

_SIM_SEED_MULTIPLIER = 10_000
"""Multiplier to derive sim_seed from rule_seed.  Centralised here to
prevent silent seed-collision bugs across sweep functions."""


def _percentile_pre_sorted(sorted_values: list[float], q: float) -> float | None:
    """Compute percentile in [0, 1] with linear interpolation on pre-sorted values."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    fraction = pos - lo
    return sorted_values[lo] * (1.0 - fraction) + sorted_values[hi] * fraction


def _to_float_list(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        numeric = float(value)
        if numeric != numeric:
            continue
        values.append(numeric)
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Phase summary / comparison builders
# ---------------------------------------------------------------------------


def _build_phase_summary(
    phase: ObservationPhase,
    run_rows: list[dict[str, Any]],
    final_metric_rows: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    rules_evaluated = len(run_rows)
    survived_count = sum(1 for row in run_rows if bool(row["survived"]))
    terminated_at_values = [
        int(row["terminated_at"]) for row in run_rows if row.get("terminated_at") is not None
    ]

    summary: dict[str, int | float | None] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "phase": phase.value,
        "rules_evaluated": rules_evaluated,
        "survival_rate": (survived_count / rules_evaluated) if rules_evaluated else 0.0,
        "termination_rate": ((rules_evaluated - survived_count) / rules_evaluated)
        if rules_evaluated
        else 0.0,
        "mean_terminated_at": _mean([float(v) for v in terminated_at_values]),
    }

    # Derive delta_mi per rule before summarizing (avoid mutating caller's list)
    enriched_rows = []
    for row in final_metric_rows:
        new_row = row.copy()
        mi = new_row.get("neighbor_mutual_information")
        null = new_row.get("mi_shuffle_null")
        new_row["delta_mi"] = (
            float(mi) - float(null)
            if mi is not None and null is not None and mi == mi and null == null
            else None
        )
        enriched_rows.append(new_row)

    for metric_name in PHASE_SUMMARY_METRIC_NAMES:
        values = sorted(_to_float_list(enriched_rows, metric_name))
        summary[f"{metric_name}_mean"] = _mean(values)
        summary[f"{metric_name}_p25"] = _percentile_pre_sorted(values, 0.25)
        summary[f"{metric_name}_p50"] = _percentile_pre_sorted(values, 0.50)
        summary[f"{metric_name}_p75"] = _percentile_pre_sorted(values, 0.75)

    return summary


def _build_phase_comparison(phase_summaries: list[dict[str, int | float | None]]) -> dict[str, Any]:
    def _phase_value(row: dict[str, int | float | None]) -> int:
        phase_value = row.get("phase")
        if not isinstance(phase_value, int):
            raise ValueError("phase summary row missing integer 'phase'")
        return phase_value

    sorted_rows = sorted(phase_summaries, key=_phase_value)
    payload: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "phases": [_phase_value(row) for row in sorted_rows],
        "deltas": {},
        "deltas_base_phase": None,
        "deltas_target_phase": None,
        "pairwise_deltas": [],
    }
    if len(sorted_rows) < 2:
        return payload

    def _row_delta(
        base: dict[str, int | float | None],
        target: dict[str, int | float | None],
    ) -> dict[str, dict[str, float | None]]:
        deltas: dict[str, dict[str, float | None]] = {}
        for key, target_value in target.items():
            if key in {"phase", "schema_version"}:
                continue
            base_value = base.get(key)
            if not isinstance(base_value, (int, float)) or not isinstance(
                target_value, (int, float)
            ):
                continue
            delta_abs = float(target_value) - float(base_value)
            delta_rel = None if float(base_value) == 0.0 else delta_abs / float(base_value)
            deltas[key] = {"absolute": delta_abs, "relative": delta_rel}
        return deltas

    for i in range(len(sorted_rows)):
        for j in range(i + 1, len(sorted_rows)):
            base = sorted_rows[i]
            target = sorted_rows[j]
            payload["pairwise_deltas"].append(
                {
                    "base_phase": _phase_value(base),
                    "target_phase": _phase_value(target),
                    "deltas": _row_delta(base, target),
                }
            )

    # Backward-compatible primary delta payload.
    # For N>2 phases, expose the first pair (lowest two phases) explicitly.
    if len(sorted_rows) >= 2:
        payload["deltas_base_phase"] = payload["pairwise_deltas"][0]["base_phase"]
        payload["deltas_target_phase"] = payload["pairwise_deltas"][0]["target_phase"]
        payload["deltas"] = payload["pairwise_deltas"][0]["deltas"]

    return payload


def _collect_final_metric_rows(
    metrics_path: Path,
    metric_columns: list[str],
    phase_results: list[SimulationResult],
    default_final_step: int,
) -> list[dict[str, Any]]:
    """Collect final-step metric rows per rule from parquet in batches."""
    final_steps = {
        result.rule_id: (
            result.terminated_at if result.terminated_at is not None else default_final_step
        )
        for result in phase_results
    }
    final_rows: list[dict[str, Any]] = []
    metrics_file = pq.ParquetFile(metrics_path)
    available_columns = set(metrics_file.schema_arrow.names)
    for required_col in ("rule_id", "step"):
        if required_col not in available_columns:
            raise ValueError(f"metrics parquet missing required column: {required_col}")
    present_columns = [col for col in metric_columns if col in available_columns]

    for batch in metrics_file.iter_batches(columns=present_columns, batch_size=8192):
        batch_dict = batch.to_pydict()
        rule_ids = batch_dict["rule_id"]
        steps = batch_dict["step"]
        for idx, rule_id in enumerate(rule_ids):
            expected_step = final_steps.get(str(rule_id))
            if expected_step is None or int(steps[idx]) != expected_step:
                continue
            row: dict[str, Any] = {}
            for name in metric_columns:
                if name in batch_dict:
                    row[name] = batch_dict[name][idx]
                else:
                    row[name] = None
            final_rows.append(row)
    return final_rows


# ---------------------------------------------------------------------------
# Density sweep helpers
# ---------------------------------------------------------------------------


def _make_density_phase_summary_rows_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build density phase summary table using static schema."""
    return pa.Table.from_pylist(rows, schema=DENSITY_PHASE_SUMMARY_SCHEMA)


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


def _density_metric_columns() -> list[str]:
    """Return metric columns needed to build final-step summaries."""
    return [
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
    final_metric_rows = _collect_final_metric_rows(
        metrics_path=metrics_path,
        metric_columns=_density_metric_columns(),
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


# ---------------------------------------------------------------------------
# Public orchestration functions
# ---------------------------------------------------------------------------


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
        metric_columns = [
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
        final_metric_rows = _collect_final_metric_rows(
            metrics_path=metrics_path,
            metric_columns=metric_columns,
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


def select_top_rules_by_delta_mi(
    metrics_path: Path,
    rules_dir: Path,
    top_k: int = 50,
) -> list[int]:
    """Select top-K rule seeds by delta_mi from existing experiment data.

    Returns a list of rule seeds sorted by descending delta_mi.
    Only includes surviving rules.
    """
    metrics_file = pq.ParquetFile(metrics_path)
    # Collect final-step MI and shuffle null per rule
    max_steps: dict[str, int] = {}
    rule_metrics: dict[str, dict[str, float]] = {}

    for batch in metrics_file.iter_batches(
        columns=["rule_id", "step", "neighbor_mutual_information", "mi_shuffle_null"],
        batch_size=8192,
    ):
        batch_dict = batch.to_pydict()
        for idx, rid in enumerate(batch_dict["rule_id"]):
            step = int(batch_dict["step"][idx])
            if rid not in max_steps or step > max_steps[rid]:
                max_steps[rid] = step
                mi = batch_dict["neighbor_mutual_information"][idx]
                null = batch_dict["mi_shuffle_null"][idx]
                if mi is not None and null is not None and mi == mi and null == null:
                    rule_metrics[rid] = {"mi": float(mi), "null": float(null)}
                else:
                    rule_metrics.pop(rid, None)  # Clear stale entry from earlier step

    # Filter to survived rules only — prefer Parquet if experiment_runs exists
    experiment_parquet = rules_dir.parent / "logs" / "experiment_runs.parquet"
    survived_rule_ids: set[str] = set()
    if experiment_parquet.exists():
        for batch in pq.ParquetFile(experiment_parquet).iter_batches(
            columns=["rule_id", "survived"], batch_size=8192
        ):
            d = batch.to_pydict()
            for idx, rid in enumerate(d["rule_id"]):
                if d["survived"][idx]:
                    survived_rule_ids.add(str(rid))
        survived_seeds: list[tuple[int, float]] = []
        # Build a rule_id → rule_seed lookup from the same Parquet
        rid_to_seed: dict[str, int] = {}
        for batch in pq.ParquetFile(experiment_parquet).iter_batches(
            columns=["rule_id", "rule_seed"], batch_size=8192
        ):
            d = batch.to_pydict()
            for idx, rid in enumerate(d["rule_id"]):
                seed_val = d["rule_seed"][idx]
                if seed_val is not None:
                    rid_to_seed[str(rid)] = int(seed_val)
        for rid_str, m in rule_metrics.items():
            if rid_str not in survived_rule_ids:
                continue
            seed = rid_to_seed.get(rid_str)
            if seed is None:
                continue  # skip if rule_seed missing
            delta = m["mi"] - m["null"]
            survived_seeds.append((seed, delta))
    else:
        survived_seeds = []
        for path in sorted(rules_dir.glob("*.json")):
            data = json.loads(path.read_text())
            if not data.get("survived", False):
                continue
            rid = data["rule_id"]
            if rid not in rule_metrics:
                continue
            delta = rule_metrics[rid]["mi"] - rule_metrics[rid]["null"]
            seed = data["metadata"]["rule_seed"]
            survived_seeds.append((int(seed), delta))

    survived_seeds.sort(key=lambda x: x[1], reverse=True)
    return [seed for seed, _ in survived_seeds[:top_k]]


def _run_simulation_to_completion(
    world: World,
    rule_table: list[int],
    phase: ObservationPhase,
    halt_detector: HaltDetector,
    uniform_detector: StateUniformDetector,
    update_mode: UpdateMode,
    enable_viability_filters: bool = True,
    state_uniform_mode: StateUniformMode = StateUniformMode.TERMINAL,
) -> tuple[str | None, tuple[tuple[int, int, int, int], ...]]:
    """Run a single simulation until termination or completion.

    Returns ``(termination_reason, final_snapshot)``.
    """
    termination_reason: str | None = None
    for step in range(world.config.steps):
        world.step(rule_table, phase, step_number=step, update_mode=update_mode)
        if not enable_viability_filters:
            continue
        snapshot = world.snapshot()
        states = world.state_vector()

        if uniform_detector.observe(states) and state_uniform_mode == StateUniformMode.TERMINAL:
            termination_reason = TerminationReason.STATE_UNIFORM.value
            break
        if halt_detector.observe(snapshot):
            termination_reason = TerminationReason.HALT.value
            break

    return termination_reason, world.snapshot()


def run_multi_seed_robustness(config: MultiSeedConfig) -> Path:
    """Evaluate selected rules across multiple simulation seeds for robustness.

    Returns path to the output parquet file.
    """
    total_work = len(config.rule_seeds) * config.n_sim_seeds * config.steps
    if total_work > MAX_EXPERIMENT_WORK_UNITS:
        raise ValueError(
            "multi-seed robustness workload exceeds safety threshold; "
            "reduce rule_seeds/n_sim_seeds/steps"
        )
    if config.n_sim_seeds >= _SIM_SEED_MULTIPLIER:
        raise ValueError(f"n_sim_seeds must be < {_SIM_SEED_MULTIPLIER} to avoid seed collisions")

    out_dir = Path(config.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for rule_seed in config.rule_seeds:
        for sim_seed_offset in range(config.n_sim_seeds):
            sim_seed = rule_seed * _SIM_SEED_MULTIPLIER + sim_seed_offset

            rule_table = generate_rule_table(phase=config.phase, seed=rule_seed)
            world_cfg = WorldConfig(steps=config.steps)
            world = World(config=world_cfg, sim_seed=sim_seed)
            halt_detector = HaltDetector(window=config.halt_window)
            uniform_detector = StateUniformDetector()

            termination_reason, snapshot = _run_simulation_to_completion(
                world,
                rule_table,
                config.phase,
                halt_detector,
                uniform_detector,
                config.update_mode,
                config.enable_viability_filters,
                config.state_uniform_mode,
            )

            survived = termination_reason is None
            mi = neighbor_mutual_information(snapshot, world_cfg.grid_width, world_cfg.grid_height)
            mi_null = shuffle_null_mi(
                snapshot,
                world_cfg.grid_width,
                world_cfg.grid_height,
                n_shuffles=config.shuffle_null_n_shuffles,
                rng=random.Random(sim_seed),
            )
            mi_delta = mi - mi_null
            n_pairs = neighbor_pair_count(snapshot, world_cfg.grid_width, world_cfg.grid_height)
            adj_frac = same_state_adjacency_fraction(
                snapshot, world_cfg.grid_width, world_cfg.grid_height
            )

            rows.append(
                {
                    "rule_seed": rule_seed,
                    "sim_seed": sim_seed,
                    "survived": survived,
                    "termination_reason": termination_reason,
                    "neighbor_mutual_information": mi,
                    "mi_shuffle_null": mi_null,
                    "delta_mi": mi_delta,
                    "n_pairs": n_pairs,
                    "same_state_adjacency_fraction": adj_frac,
                    "update_mode": config.update_mode.value,
                    "state_uniform_mode": config.state_uniform_mode.value,
                    "enable_viability_filters": config.enable_viability_filters,
                }
            )

    output_path = logs_dir / "multi_seed_results.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=MULTI_SEED_SCHEMA), output_path)
    return output_path


def run_halt_window_sweep(config: HaltWindowSweepConfig) -> Path:
    """Evaluate rules across multiple halt-window values for sensitivity analysis.

    Returns path to the output parquet file.
    """
    total_work = len(config.rule_seeds) * len(config.halt_windows) * config.steps
    if total_work > MAX_EXPERIMENT_WORK_UNITS:
        raise ValueError(
            "halt-window sweep workload exceeds safety threshold; "
            "reduce rule_seeds/halt_windows/steps"
        )

    out_dir = Path(config.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for rule_seed in config.rule_seeds:
        rule_table = generate_rule_table(phase=config.phase, seed=rule_seed)
        for halt_window in config.halt_windows:
            sim_seed = rule_seed * _SIM_SEED_MULTIPLIER
            world_cfg = WorldConfig(steps=config.steps)
            world = World(config=world_cfg, sim_seed=sim_seed)
            halt_detector = HaltDetector(window=halt_window)
            uniform_detector = StateUniformDetector()

            termination_reason, snapshot = _run_simulation_to_completion(
                world,
                rule_table,
                config.phase,
                halt_detector,
                uniform_detector,
                config.update_mode,
                enable_viability_filters=config.enable_viability_filters,
                state_uniform_mode=config.state_uniform_mode,
            )

            survived = termination_reason is None
            mi = neighbor_mutual_information(snapshot, world_cfg.grid_width, world_cfg.grid_height)
            mi_null = shuffle_null_mi(
                snapshot,
                world_cfg.grid_width,
                world_cfg.grid_height,
                n_shuffles=config.shuffle_null_n_shuffles,
                rng=random.Random(sim_seed),
            )
            mi_delta = mi - mi_null
            n_pairs = neighbor_pair_count(snapshot, world_cfg.grid_width, world_cfg.grid_height)

            rows.append(
                {
                    "rule_seed": rule_seed,
                    "halt_window": halt_window,
                    "survived": survived,
                    "termination_reason": termination_reason,
                    "neighbor_mutual_information": mi,
                    "mi_shuffle_null": mi_null,
                    "delta_mi": mi_delta,
                    "n_pairs": n_pairs,
                    "update_mode": config.update_mode.value,
                    "state_uniform_mode": config.state_uniform_mode.value,
                    "enable_viability_filters": config.enable_viability_filters,
                }
            )

    output_path = logs_dir / "halt_window_sweep.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=HALT_WINDOW_SWEEP_SCHEMA), output_path)
    return output_path
