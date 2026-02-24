"""Core simulation engine: seeded batch search with metric computation."""

from __future__ import annotations

import itertools
import json
import random
from collections import deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from objectless_alife.config.constants import ACTION_SPACE_SIZE, FLUSH_THRESHOLD
from objectless_alife.config.types import SearchConfig, SimulationResult, StateUniformMode
from objectless_alife.domain.filters import (
    HaltDetector,
    LowActivityDetector,
    ShortPeriodDetector,
    StateUniformDetector,
    TerminationReason,
)
from objectless_alife.domain.rules import ObservationPhase, generate_rule_table
from objectless_alife.domain.world import World, WorldConfig
from objectless_alife.io.schemas import METRICS_SCHEMA, RULE_PAYLOAD_SCHEMA_VERSION
from objectless_alife.metrics.information import (
    block_ncd,
    normalized_hamming_distance,
    serialize_snapshot,
    shuffle_null_mi,
    state_entropy,
)
from objectless_alife.metrics.temporal import quasi_periodicity_peak_count
from objectless_alife.simulation.persistence import flush_sim_columns
from objectless_alife.simulation.step import (
    compute_step_metrics,
    entropy_from_action_counts,
    mean_and_pvariance,
)

_UNSET: object = object()
"""Sentinel indicating a parameter was not explicitly provided."""


def _deterministic_rule_id(phase: ObservationPhase, rule_seed: int, sim_seed: int) -> str:
    """Build reproducible rule ID stable across runs for identical seeds."""
    return f"phase{phase.value}_rs{rule_seed}_ss{sim_seed}"


def run_batch_search(
    n_rules: int,
    phase: ObservationPhase,
    out_dir: Path,
    steps: object = _UNSET,
    halt_window: object = _UNSET,
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

    # Resolve sentinel defaults
    _steps: int = 200 if steps is _UNSET else int(steps)  # type: ignore[call-overload]
    _halt_window: int = 10 if halt_window is _UNSET else int(halt_window)  # type: ignore[call-overload]

    search_config = config or SearchConfig(steps=_steps, halt_window=_halt_window)
    if search_config.steps < 1:
        raise ValueError("steps must be >= 1")
    if config is not None:
        if steps is not _UNSET and _steps != config.steps:
            raise ValueError("steps conflicts with config.steps")
        if halt_window is not _UNSET and _halt_window != config.halt_window:
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
                "same_state_adjacency_fraction": [],
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
                actions = world.step(
                    rule_table,
                    phase,
                    step_number=step,
                    update_mode=search_config.update_mode,
                )
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
                    per_agent_entropies[agent_id] = entropy_from_action_counts(
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

                action_entropy_mean, action_entropy_var = mean_and_pvariance(per_agent_entropies)

                step_metrics = compute_step_metrics(
                    snapshot=snapshot,
                    snapshot_bytes=snapshot_bytes,
                    step_entropy=step_entropy,
                    predictability=predictability,
                    running_phase_transition_delta=running_phase_transition_delta,
                    action_entropy_mean=action_entropy_mean,
                    action_entropy_var=action_entropy_var,
                    block_ncd_value=block_ncd_value,
                    grid_width=world_cfg.grid_width,
                    grid_height=world_cfg.grid_height,
                )
                metric_columns["rule_id"].append(rule_id)
                metric_columns["step"].append(step)
                for key, value in step_metrics.items():
                    metric_columns[key].append(value)

                for agent_id, x, y, state in snapshot:
                    sim_columns["rule_id"].append(rule_id)
                    sim_columns["step"].append(step)
                    sim_columns["agent_id"].append(agent_id)
                    sim_columns["x"].append(x)
                    sim_columns["y"].append(y)
                    sim_columns["state"].append(state)
                    sim_columns["action"].append(actions[agent_id])
                if len(sim_columns["rule_id"]) >= FLUSH_THRESHOLD:
                    sim_writer = flush_sim_columns(
                        sim_columns=sim_columns,
                        simulation_log_path=simulation_log_path,
                        sim_writer=sim_writer,
                    )

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
                if search_config.enable_viability_filters:
                    if (
                        uniform_triggered
                        and search_config.state_uniform_mode == StateUniformMode.TERMINAL
                    ):
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
            if search_config.skip_null_models:
                mi_null = None
            else:
                mi_null = shuffle_null_mi(
                    snapshot,
                    world_cfg.grid_width,
                    world_cfg.grid_height,
                    n_shuffles=search_config.shuffle_null_n_shuffles,
                    rng=random.Random(sim_seed),
                )
            metric_columns["mi_shuffle_null"] = [mi_null] * len(metric_columns["step"])

            sim_writer = flush_sim_columns(
                sim_columns=sim_columns,
                simulation_log_path=simulation_log_path,
                sim_writer=sim_writer,
            )
            metric_table = pa.Table.from_pydict(metric_columns, schema=METRICS_SCHEMA)
            if metric_writer is None:
                metric_writer = pq.ParquetWriter(metrics_summary_path, METRICS_SCHEMA)
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
                    "enable_viability_filters": search_config.enable_viability_filters,
                    "update_mode": search_config.update_mode.value,
                    "state_uniform_mode": search_config.state_uniform_mode.value,
                    "schema_version": RULE_PAYLOAD_SCHEMA_VERSION,
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
