from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.filters import (
    ACTION_SPACE_SIZE,
    HaltDetector,
    LowActivityDetector,
    ShortPeriodDetector,
    StateUniformDetector,
    TerminationReason,
)
from src.metrics import (
    block_ncd,
    cluster_count_by_state,
    compression_ratio,
    morans_i_occupied,
    neighbor_mutual_information,
    normalized_hamming_distance,
    quasi_periodicity_peak_count,
    serialize_snapshot,
    shuffle_null_mi,
    state_entropy,
)
from src.rules import ObservationPhase, generate_rule_table
from src.world import World, WorldConfig

MAX_EXPERIMENT_WORK_UNITS = 100_000_000
AGGREGATE_SCHEMA_VERSION = 1
DENSITY_SWEEP_SCHEMA_VERSION = 1
SIMULATION_SCHEMA = pa.schema(
    [
        ("rule_id", pa.string()),
        ("step", pa.int64()),
        ("agent_id", pa.int64()),
        ("x", pa.int64()),
        ("y", pa.int64()),
        ("state", pa.int64()),
        ("action", pa.int64()),
    ]
)
METRICS_SCHEMA = pa.schema(
    [
        ("rule_id", pa.string()),
        ("step", pa.int64()),
        ("state_entropy", pa.float64()),
        ("compression_ratio", pa.float64()),
        ("predictability_hamming", pa.float64()),
        ("morans_i", pa.float64()),
        ("cluster_count", pa.int64()),
        ("quasi_periodicity_peaks", pa.int64()),
        ("phase_transition_max_delta", pa.float64()),
        ("neighbor_mutual_information", pa.float64()),
        ("action_entropy_mean", pa.float64()),
        ("action_entropy_variance", pa.float64()),
        ("block_ncd", pa.float64()),
        ("mi_shuffle_null", pa.float64()),
    ]
)
PHASE_SUMMARY_METRIC_NAMES = [
    "state_entropy",
    "compression_ratio",
    "predictability_hamming",
    "morans_i",
    "cluster_count",
    "neighbor_mutual_information",
    "quasi_periodicity_peaks",
    "phase_transition_max_delta",
    "action_entropy_mean",
    "action_entropy_variance",
    "block_ncd",
    "mi_shuffle_null",
]
DENSITY_SWEEP_RUNS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int64()),
        ("rule_id", pa.string()),
        ("phase", pa.int64()),
        ("grid_width", pa.int64()),
        ("grid_height", pa.int64()),
        ("num_agents", pa.int64()),
        ("density_ratio", pa.float64()),
        ("seed_batch", pa.int64()),
        ("rule_seed", pa.int64()),
        ("sim_seed", pa.int64()),
        ("survived", pa.bool_()),
        ("termination_reason", pa.string()),
        ("terminated_at", pa.int64()),
    ]
)
DENSITY_PHASE_SUMMARY_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int64()),
        ("phase", pa.int64()),
        ("grid_width", pa.int64()),
        ("grid_height", pa.int64()),
        ("num_agents", pa.int64()),
        ("density_ratio", pa.float64()),
        ("rules_evaluated", pa.int64()),
        ("survival_rate", pa.float64()),
        ("termination_rate", pa.float64()),
        ("mean_terminated_at", pa.float64()),
    ]
    + [
        (f"{metric}_{suffix}", pa.float64())
        for metric in PHASE_SUMMARY_METRIC_NAMES
        for suffix in ("mean", "p25", "p50", "p75")
    ]
)
DENSITY_PHASE_COMPARISON_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int64()),
        ("base_phase", pa.int64()),
        ("target_phase", pa.int64()),
        ("grid_width", pa.int64()),
        ("grid_height", pa.int64()),
        ("num_agents", pa.int64()),
        ("density_ratio", pa.float64()),
        ("metric", pa.string()),
        ("delta_absolute", pa.float64()),
        ("delta_relative", pa.float64()),
    ]
)


@dataclass(frozen=True)
class SimulationResult:
    """Top-level result for one evaluated rule table."""

    rule_id: str
    survived: bool
    terminated_at: int | None
    termination_reason: str | None


@dataclass(frozen=True)
class SearchConfig:
    """Batch-search runtime parameters including optional dynamic filters."""

    steps: int = 200
    halt_window: int = 10
    filter_short_period: bool = False
    short_period_max_period: int = 2
    short_period_history_size: int = 8
    filter_low_activity: bool = False
    low_activity_window: int = 5
    low_activity_min_unique_ratio: float = 0.2
    block_ncd_window: int = 10
    shuffle_null_n_shuffles: int = 200


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment-scale runtime settings for multi-phase, multi-seed evaluation."""

    phases: tuple[ObservationPhase, ...] = (
        ObservationPhase.PHASE1_DENSITY,
        ObservationPhase.PHASE2_PROFILE,
    )
    n_rules: int = 100
    n_seed_batches: int = 1
    out_dir: Path = Path("data")
    steps: int = 200
    halt_window: int = 10
    rule_seed_start: int = 0
    sim_seed_start: int = 0
    filter_short_period: bool = False
    short_period_max_period: int = 2
    short_period_history_size: int = 8
    filter_low_activity: bool = False
    low_activity_window: int = 5
    low_activity_min_unique_ratio: float = 0.2
    block_ncd_window: int = 10


@dataclass(frozen=True)
class DensitySweepConfig:
    """Runtime settings for grid/agent density sweeps across both phases."""

    grid_sizes: tuple[tuple[int, int], ...] = ((20, 20),)
    agent_counts: tuple[int, ...] = (30,)
    n_rules: int = 100
    n_seed_batches: int = 1
    out_dir: Path = Path("data")
    steps: int = 200
    halt_window: int = 10
    rule_seed_start: int = 0
    sim_seed_start: int = 0
    filter_short_period: bool = False
    short_period_max_period: int = 2
    short_period_history_size: int = 8
    filter_low_activity: bool = False
    low_activity_window: int = 5
    low_activity_min_unique_ratio: float = 0.2
    block_ncd_window: int = 10


def _deterministic_rule_id(phase: ObservationPhase, rule_seed: int, sim_seed: int) -> str:
    """Build reproducible rule ID stable across runs for identical seeds."""
    return f"phase{phase.value}_rs{rule_seed}_ss{sim_seed}"


def _entropy_from_action_counts(action_counts: list[int], total_actions: int) -> float:
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


def _mean_and_pvariance(values: list[float]) -> tuple[float, float]:
    """Return mean and population variance for non-empty values."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance


def run_batch_search(
    n_rules: int,
    phase: ObservationPhase,
    out_dir: Path,
    steps: int = 200,
    halt_window: int = 10,
    base_rule_seed: int = 0,
    base_sim_seed: int = 0,
    world_config: WorldConfig | None = None,
    config: SearchConfig | None = None,
) -> list[SimulationResult]:
    """Run seeded batch simulations and persist JSON/Parquet outputs.

    `steps` and `halt_window` are backward-compatible entrypoints. Prefer
    passing `config=SearchConfig(...)` for new settings.
    """
    if n_rules < 1:
        raise ValueError("n_rules must be >= 1")

    search_config = config or SearchConfig(steps=steps, halt_window=halt_window)
    if search_config.steps < 1:
        raise ValueError("steps must be >= 1")
    if config is not None:
        if steps != 200 and steps != config.steps:
            raise ValueError("steps conflicts with config.steps")
        if halt_window != 10 and halt_window != config.halt_window:
            raise ValueError("halt_window conflicts with config.halt_window")

    if world_config is not None:
        if world_config.steps != search_config.steps:
            raise ValueError("steps conflicts with world_config.steps")
        world_cfg = world_config
    else:
        world_cfg = WorldConfig(steps=search_config.steps)

    out_dir = Path(out_dir)
    rules_dir = out_dir / "rules"
    logs_dir = out_dir / "logs"
    rules_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sim_writer: pq.ParquetWriter | None = None
    metric_writer: pq.ParquetWriter | None = None
    simulation_log_path = logs_dir / "simulation_log.parquet"
    metrics_summary_path = logs_dir / "metrics_summary.parquet"
    results: list[SimulationResult] = []

    try:
        for i in range(n_rules):
            rule_seed = base_rule_seed + i
            sim_seed = base_sim_seed + i
            rule_id = _deterministic_rule_id(phase=phase, rule_seed=rule_seed, sim_seed=sim_seed)

            rule_table = generate_rule_table(phase=phase, seed=rule_seed)
            world = World(config=world_cfg, sim_seed=sim_seed)
            halt_detector = HaltDetector(window=search_config.halt_window)
            uniform_detector = StateUniformDetector()
            short_period_detector = (
                ShortPeriodDetector(
                    max_period=search_config.short_period_max_period,
                    history_size=search_config.short_period_history_size,
                )
                if search_config.filter_short_period
                else None
            )
            low_activity_detector = (
                LowActivityDetector(
                    window=search_config.low_activity_window,
                    min_unique_ratio=search_config.low_activity_min_unique_ratio,
                )
                if search_config.filter_low_activity
                else None
            )

            terminated_at: int | None = None
            termination_reason: str | None = None
            prev_states: list[int] | None = None
            entropy_series: list[float] = []
            block_window = search_config.block_ncd_window
            snapshot_bytes_window: deque[bytes] | None = (
                deque(maxlen=block_window * 2) if block_window > 0 else None
            )
            per_agent_action_counts: list[list[int]] = [
                [0] * ACTION_SPACE_SIZE for _ in range(world_cfg.num_agents)
            ]
            per_agent_action_totals: list[int] = [0] * world_cfg.num_agents
            per_agent_entropies: list[float] = [0.0] * world_cfg.num_agents
            sim_columns: dict[str, list[int | str]] = {
                "rule_id": [],
                "step": [],
                "agent_id": [],
                "x": [],
                "y": [],
                "state": [],
                "action": [],
            }
            metric_columns: dict[str, list[int | str | float | None]] = {
                "rule_id": [],
                "step": [],
                "state_entropy": [],
                "compression_ratio": [],
                "predictability_hamming": [],
                "morans_i": [],
                "cluster_count": [],
                "quasi_periodicity_peaks": [],
                "phase_transition_max_delta": [],
                "neighbor_mutual_information": [],
                "action_entropy_mean": [],
                "action_entropy_variance": [],
                "block_ncd": [],
                "mi_shuffle_null": [],
            }
            running_phase_transition_delta = 0.0
            halt_triggered = False
            uniform_triggered = False
            short_period_triggered = False
            low_activity_triggered = False

            for step in range(world_cfg.steps):
                actions = world.step(rule_table, phase, step_number=step)
                snapshot = world.snapshot()
                states = world.state_vector()
                snapshot_bytes = serialize_snapshot(
                    snapshot, world_cfg.grid_width, world_cfg.grid_height
                )
                step_entropy = state_entropy(states)
                if len(entropy_series) > 0:
                    running_phase_transition_delta = max(
                        running_phase_transition_delta, abs(step_entropy - entropy_series[-1])
                    )
                entropy_series.append(step_entropy)
                if snapshot_bytes_window is not None:
                    snapshot_bytes_window.append(snapshot_bytes)
                for agent_id, action in enumerate(actions):
                    per_agent_action_counts[agent_id][action] += 1
                    per_agent_action_totals[agent_id] += 1
                    per_agent_entropies[agent_id] = _entropy_from_action_counts(
                        per_agent_action_counts[agent_id], per_agent_action_totals[agent_id]
                    )

                predictability = (
                    None
                    if prev_states is None
                    else normalized_hamming_distance(prev_states, states)
                )
                block_ncd_value: float | None = None
                if (
                    snapshot_bytes_window is not None
                    and len(snapshot_bytes_window) >= block_window * 2
                ):
                    windowed = iter(snapshot_bytes_window)
                    prev_block = b"".join(itertools.islice(windowed, block_window))
                    curr_block = b"".join(itertools.islice(windowed, block_window))
                    block_ncd_value = block_ncd(prev_block, curr_block)

                action_entropy_mean, action_entropy_var = _mean_and_pvariance(per_agent_entropies)

                metric_columns["rule_id"].append(rule_id)
                metric_columns["step"].append(step)
                metric_columns["state_entropy"].append(step_entropy)
                metric_columns["compression_ratio"].append(compression_ratio(snapshot_bytes))
                metric_columns["predictability_hamming"].append(predictability)
                metric_columns["morans_i"].append(
                    morans_i_occupied(
                        snapshot,
                        grid_width=world_cfg.grid_width,
                        grid_height=world_cfg.grid_height,
                    )
                )
                metric_columns["cluster_count"].append(
                    cluster_count_by_state(
                        snapshot,
                        grid_width=world_cfg.grid_width,
                        grid_height=world_cfg.grid_height,
                    )
                )
                metric_columns["phase_transition_max_delta"].append(running_phase_transition_delta)
                metric_columns["neighbor_mutual_information"].append(
                    neighbor_mutual_information(
                        snapshot,
                        grid_width=world_cfg.grid_width,
                        grid_height=world_cfg.grid_height,
                    )
                )
                metric_columns["action_entropy_mean"].append(action_entropy_mean)
                metric_columns["action_entropy_variance"].append(action_entropy_var)
                metric_columns["block_ncd"].append(block_ncd_value)

                for agent_id, x, y, state in snapshot:
                    sim_columns["rule_id"].append(rule_id)
                    sim_columns["step"].append(step)
                    sim_columns["agent_id"].append(agent_id)
                    sim_columns["x"].append(x)
                    sim_columns["y"].append(y)
                    sim_columns["state"].append(state)
                    sim_columns["action"].append(actions[agent_id])

                halt_triggered = halt_detector.observe(snapshot)
                uniform_triggered = uniform_detector.observe(states)
                short_period_triggered = (
                    short_period_detector.observe(snapshot)
                    if short_period_detector is not None
                    else False
                )
                low_activity_triggered = (
                    low_activity_detector.observe(actions)
                    if low_activity_detector is not None
                    else False
                )
                if uniform_triggered:
                    terminated_at = step
                    termination_reason = TerminationReason.STATE_UNIFORM.value
                    break
                elif halt_triggered:
                    terminated_at = step
                    termination_reason = TerminationReason.HALT.value
                    break
                elif short_period_triggered:
                    terminated_at = step
                    termination_reason = TerminationReason.SHORT_PERIOD.value
                    break
                elif low_activity_triggered:
                    terminated_at = step
                    termination_reason = TerminationReason.LOW_ACTIVITY.value
                    break

                prev_states = states

            quasi_periodicity_peaks = quasi_periodicity_peak_count(entropy_series)
            metric_columns["quasi_periodicity_peaks"] = [quasi_periodicity_peaks] * len(
                metric_columns["step"]
            )
            mi_null = shuffle_null_mi(
                snapshot,
                world_cfg.grid_width,
                world_cfg.grid_height,
                n_shuffles=search_config.shuffle_null_n_shuffles,
                rng=random.Random(sim_seed),
            )
            metric_columns["mi_shuffle_null"] = [mi_null] * len(metric_columns["step"])

            sim_table = pa.Table.from_pydict(sim_columns, schema=SIMULATION_SCHEMA)
            metric_table = pa.Table.from_pydict(metric_columns, schema=METRICS_SCHEMA)
            if sim_writer is None:
                sim_writer = pq.ParquetWriter(simulation_log_path, SIMULATION_SCHEMA)
            if metric_writer is None:
                metric_writer = pq.ParquetWriter(metrics_summary_path, METRICS_SCHEMA)
            sim_writer.write_table(sim_table)
            metric_writer.write_table(metric_table)

            survived = termination_reason is None

            rule_payload = {
                "rule_id": rule_id,
                "table": rule_table,
                "survived": survived,
                "filter_results": {
                    "halt": halt_triggered,
                    "state_uniform": uniform_triggered,
                    "short_period": short_period_triggered,
                    "low_activity": low_activity_triggered,
                },
                "metadata": {
                    "rule_seed": rule_seed,
                    "sim_seed": sim_seed,
                    "steps": world_cfg.steps,
                    "grid_width": world_cfg.grid_width,
                    "grid_height": world_cfg.grid_height,
                    "halt_window": search_config.halt_window,
                    "observation_phase": phase.value,
                    "terminated_at": terminated_at,
                    "termination_reason": termination_reason,
                    "filter_short_period": search_config.filter_short_period,
                    "filter_low_activity": search_config.filter_low_activity,
                    "schema_version": 2,
                },
            }
            (rules_dir / f"{rule_id}.json").write_text(
                json.dumps(rule_payload, ensure_ascii=False, indent=2)
            )

            results.append(
                SimulationResult(
                    rule_id=rule_id,
                    survived=survived,
                    terminated_at=terminated_at,
                    termination_reason=termination_reason,
                )
            )
    finally:
        if sim_writer is not None:
            sim_writer.close()
        if metric_writer is not None:
            metric_writer.close()

    return results


def _parse_phase_list(raw_phases: str) -> tuple[ObservationPhase, ...]:
    """Parse comma-delimited phase list and require exactly two entries."""
    parts = [part.strip() for part in raw_phases.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("phases must contain exactly two values")

    phases: list[ObservationPhase] = []
    for part in parts:
        try:
            phase_raw = int(part)
        except ValueError as exc:
            raise ValueError("phases must be integers 1, 2, 3, or 4") from exc
        phase = _parse_phase(phase_raw)
        phases.append(phase)

    if phases[0] == phases[1]:
        raise ValueError("phases must include two distinct values")

    return tuple(phases)


def _parse_grid_sizes(raw_grid_sizes: str) -> tuple[tuple[int, int], ...]:
    """Parse comma-delimited grid sizes formatted as `WxH`."""
    parts = [part.strip() for part in raw_grid_sizes.split(",") if part.strip()]
    if not parts:
        raise ValueError("grid-sizes must not be empty")

    grid_sizes: list[tuple[int, int]] = []
    for part in parts:
        tokens = part.lower().split("x")
        if len(tokens) != 2:
            raise ValueError("grid-sizes entries must use WxH format")
        width_raw, height_raw = tokens
        try:
            width = int(width_raw)
            height = int(height_raw)
        except ValueError as exc:
            raise ValueError("grid-sizes entries must use integer WxH values") from exc
        if width < 1 or height < 1:
            raise ValueError("grid-sizes entries must be >= 1x1")
        grid_sizes.append((width, height))
    return tuple(grid_sizes)


def _parse_positive_int_csv(raw_values: str, label: str) -> tuple[int, ...]:
    """Parse comma-delimited positive integers."""
    parts = [part.strip() for part in raw_values.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{label} must not be empty")

    values: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"{label} must contain integers") from exc
        if value < 1:
            raise ValueError(f"{label} values must be >= 1")
        values.append(value)
    return tuple(values)


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

    for metric_name in PHASE_SUMMARY_METRIC_NAMES:
        values = sorted(_to_float_list(final_metric_rows, metric_name))
        summary[f"{metric_name}_mean"] = _mean(values)
        summary[f"{metric_name}_p25"] = _percentile_pre_sorted(values, 0.25)
        summary[f"{metric_name}_p50"] = _percentile_pre_sorted(values, 0.50)
        summary[f"{metric_name}_p75"] = _percentile_pre_sorted(values, 0.75)

    return summary


def _build_phase_comparison(phase_summaries: list[dict[str, int | float | None]]) -> dict[str, Any]:
    sorted_rows = sorted(phase_summaries, key=lambda row: int(row["phase"]))
    payload: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "phases": [int(row["phase"]) for row in sorted_rows],
        "deltas": {},
    }
    if len(sorted_rows) < 2:
        return payload

    base = sorted_rows[0]
    target = sorted_rows[1]
    for key, target_value in target.items():
        if key in {"phase", "schema_version"}:
            continue
        base_value = base.get(key)
        if not isinstance(base_value, (int, float)) or not isinstance(target_value, (int, float)):
            continue
        delta_abs = float(target_value) - float(base_value)
        delta_rel = None if float(base_value) == 0.0 else delta_abs / float(base_value)
        payload["deltas"][key] = {"absolute": delta_abs, "relative": delta_rel}
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

    for batch in metrics_file.iter_batches(columns=metric_columns, batch_size=8192):
        batch_dict = batch.to_pydict()
        rule_ids = batch_dict["rule_id"]
        steps = batch_dict["step"]
        for idx, rule_id in enumerate(rule_ids):
            expected_step = final_steps.get(str(rule_id))
            if expected_step is None or int(steps[idx]) != expected_step:
                continue
            final_rows.append({name: batch_dict[name][idx] for name in metric_columns})
    return final_rows


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
    return SearchConfig(
        steps=config.steps,
        halt_window=config.halt_window,
        filter_short_period=config.filter_short_period,
        short_period_max_period=config.short_period_max_period,
        short_period_history_size=config.short_period_history_size,
        filter_low_activity=config.filter_low_activity,
        low_activity_window=config.low_activity_window,
        low_activity_min_unique_ratio=config.low_activity_min_unique_ratio,
        block_ncd_window=config.block_ncd_window,
    )


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
    if len(config.phases) != 2:
        raise ValueError("run_experiment currently supports exactly two phases")

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
        phase_search_config = SearchConfig(
            steps=config.steps,
            halt_window=config.halt_window,
            filter_short_period=config.filter_short_period,
            short_period_max_period=config.short_period_max_period,
            short_period_history_size=config.short_period_history_size,
            filter_low_activity=config.filter_low_activity,
            low_activity_window=config.low_activity_window,
            low_activity_min_unique_ratio=config.low_activity_min_unique_ratio,
            block_ncd_window=config.block_ncd_window,
        )
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


def _parse_phase(raw_phase: int) -> ObservationPhase:
    """Parse CLI phase value into ObservationPhase enum."""
    try:
        return ObservationPhase(raw_phase)
    except ValueError as exc:
        raise ValueError("phase must be 1, 2, 3, or 4") from exc


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for search execution."""
    parser = argparse.ArgumentParser(description="Run objective-free ALife search")
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--n-rules", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--halt-window", type=int, default=10)
    parser.add_argument("--rule-seed", type=int, default=0)
    parser.add_argument("--sim-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--density-sweep", action="store_true")
    mode_group.add_argument("--experiment", action="store_true")
    parser.add_argument("--grid-sizes", type=str, default="20x20")
    parser.add_argument("--agent-counts", type=str, default="30")
    parser.add_argument("--seed-batches", type=int, default=1)
    parser.add_argument("--phases", type=str, default="1,2")
    parser.add_argument("--filter-short-period", action="store_true")
    parser.add_argument("--short-period-max-period", type=int, default=2)
    parser.add_argument("--short-period-history-size", type=int, default=8)
    parser.add_argument("--filter-low-activity", action="store_true")
    parser.add_argument("--low-activity-window", type=int, default=5)
    parser.add_argument("--low-activity-min-unique-ratio", type=float, default=0.2)
    parser.add_argument("--block-ncd-window", type=int, default=10)
    args = parser.parse_args(argv)

    if args.density_sweep:
        density_sweep_config = DensitySweepConfig(
            grid_sizes=_parse_grid_sizes(args.grid_sizes),
            agent_counts=_parse_positive_int_csv(args.agent_counts, "agent-counts"),
            n_rules=args.n_rules,
            n_seed_batches=args.seed_batches,
            out_dir=args.out_dir,
            steps=args.steps,
            halt_window=args.halt_window,
            rule_seed_start=args.rule_seed,
            sim_seed_start=args.sim_seed,
            filter_short_period=args.filter_short_period,
            short_period_max_period=args.short_period_max_period,
            short_period_history_size=args.short_period_history_size,
            filter_low_activity=args.filter_low_activity,
            low_activity_window=args.low_activity_window,
            low_activity_min_unique_ratio=args.low_activity_min_unique_ratio,
            block_ncd_window=args.block_ncd_window,
        )
        results = run_density_sweep(density_sweep_config)
        summary = {
            "mode": "density_sweep",
            "phases": [1, 2],
            "density_points": len(density_sweep_config.grid_sizes)
            * len(density_sweep_config.agent_counts),
            "total_rules": len(results),
            "survived": sum(1 for r in results if r.survived),
            "terminated": sum(1 for r in results if not r.survived),
        }
    elif args.experiment:
        experiment_config = ExperimentConfig(
            phases=_parse_phase_list(args.phases),
            n_rules=args.n_rules,
            n_seed_batches=args.seed_batches,
            out_dir=args.out_dir,
            steps=args.steps,
            halt_window=args.halt_window,
            rule_seed_start=args.rule_seed,
            sim_seed_start=args.sim_seed,
            filter_short_period=args.filter_short_period,
            short_period_max_period=args.short_period_max_period,
            short_period_history_size=args.short_period_history_size,
            filter_low_activity=args.filter_low_activity,
            low_activity_window=args.low_activity_window,
            low_activity_min_unique_ratio=args.low_activity_min_unique_ratio,
            block_ncd_window=args.block_ncd_window,
        )
        results = run_experiment(experiment_config)
        summary = {
            "experiment": True,
            "phases": [phase.value for phase in experiment_config.phases],
            "seed_batches": experiment_config.n_seed_batches,
            "total_rules": len(results),
            "survived": sum(1 for r in results if r.survived),
            "terminated": sum(1 for r in results if not r.survived),
        }
    else:
        phase = _parse_phase(args.phase)
        search_config = SearchConfig(
            steps=args.steps,
            halt_window=args.halt_window,
            filter_short_period=args.filter_short_period,
            short_period_max_period=args.short_period_max_period,
            short_period_history_size=args.short_period_history_size,
            filter_low_activity=args.filter_low_activity,
            low_activity_window=args.low_activity_window,
            low_activity_min_unique_ratio=args.low_activity_min_unique_ratio,
            block_ncd_window=args.block_ncd_window,
        )
        results = run_batch_search(
            n_rules=args.n_rules,
            phase=phase,
            out_dir=args.out_dir,
            base_rule_seed=args.rule_seed,
            base_sim_seed=args.sim_seed,
            config=search_config,
        )

        summary = {
            "experiment": False,
            "phase": phase.value,
            "total_rules": len(results),
            "survived": sum(1 for r in results if r.survived),
            "terminated": sum(1 for r in results if not r.survived),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
