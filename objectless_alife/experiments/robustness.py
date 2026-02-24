"""Multi-seed robustness and halt-window sensitivity sweep orchestration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from objectless_alife.config.constants import MAX_EXPERIMENT_WORK_UNITS
from objectless_alife.config.types import (
    HaltWindowSweepConfig,
    MultiSeedConfig,
    StateUniformMode,
    UpdateMode,
)
from objectless_alife.domain.filters import HaltDetector, StateUniformDetector, TerminationReason
from objectless_alife.domain.rules import ObservationPhase, generate_rule_table
from objectless_alife.domain.world import World, WorldConfig
from objectless_alife.io.schemas import HALT_WINDOW_SWEEP_SCHEMA, MULTI_SEED_SCHEMA
from objectless_alife.metrics.information import neighbor_mutual_information, shuffle_null_mi
from objectless_alife.metrics.spatial import neighbor_pair_count, same_state_adjacency_fraction

_SIM_SEED_MULTIPLIER = 10_000
"""Multiplier to derive sim_seed from rule_seed to prevent seed-collision bugs."""


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


def _run_simulation_and_metrics(
    rule_table: list[int],
    sim_seed: int,
    steps: int,
    halt_window: int,
    phase: ObservationPhase,
    update_mode: UpdateMode,
    enable_viability_filters: bool,
    state_uniform_mode: StateUniformMode,
    shuffle_null_n_shuffles: int,
) -> dict[str, Any]:
    """Run a single simulation and compute robustness metrics."""
    world_cfg = WorldConfig(steps=steps)
    world = World(config=world_cfg, sim_seed=sim_seed)
    halt_detector = HaltDetector(window=halt_window)
    uniform_detector = StateUniformDetector()

    termination_reason, snapshot = _run_simulation_to_completion(
        world,
        rule_table,
        phase,
        halt_detector,
        uniform_detector,
        update_mode,
        enable_viability_filters,
        state_uniform_mode,
    )

    survived = termination_reason is None
    mi = neighbor_mutual_information(snapshot, world_cfg.grid_width, world_cfg.grid_height)
    mi_null = shuffle_null_mi(
        snapshot,
        world_cfg.grid_width,
        world_cfg.grid_height,
        n_shuffles=shuffle_null_n_shuffles,
        rng=random.Random(sim_seed),
    )
    mi_delta = mi - mi_null
    n_pairs = neighbor_pair_count(snapshot, world_cfg.grid_width, world_cfg.grid_height)
    adj_frac = same_state_adjacency_fraction(snapshot, world_cfg.grid_width, world_cfg.grid_height)

    return {
        "survived": survived,
        "termination_reason": termination_reason,
        "neighbor_mutual_information": mi,
        "mi_shuffle_null": mi_null,
        "delta_mi": mi_delta,
        "n_pairs": n_pairs,
        "same_state_adjacency_fraction": adj_frac,
    }


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
        rule_table = generate_rule_table(phase=config.phase, seed=rule_seed)
        for sim_seed_offset in range(config.n_sim_seeds):
            sim_seed = rule_seed * _SIM_SEED_MULTIPLIER + sim_seed_offset

            result = _run_simulation_and_metrics(
                rule_table=rule_table,
                sim_seed=sim_seed,
                steps=config.steps,
                halt_window=config.halt_window,
                phase=config.phase,
                update_mode=config.update_mode,
                enable_viability_filters=config.enable_viability_filters,
                state_uniform_mode=config.state_uniform_mode,
                shuffle_null_n_shuffles=config.shuffle_null_n_shuffles,
            )

            rows.append(
                {
                    "rule_seed": rule_seed,
                    "sim_seed": sim_seed,
                    "survived": result["survived"],
                    "termination_reason": result["termination_reason"],
                    "neighbor_mutual_information": result["neighbor_mutual_information"],
                    "mi_shuffle_null": result["mi_shuffle_null"],
                    "delta_mi": result["delta_mi"],
                    "n_pairs": result["n_pairs"],
                    "same_state_adjacency_fraction": result["same_state_adjacency_fraction"],
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

            result = _run_simulation_and_metrics(
                rule_table=rule_table,
                sim_seed=sim_seed,
                steps=config.steps,
                halt_window=halt_window,
                phase=config.phase,
                update_mode=config.update_mode,
                enable_viability_filters=config.enable_viability_filters,
                state_uniform_mode=config.state_uniform_mode,
                shuffle_null_n_shuffles=config.shuffle_null_n_shuffles,
            )

            rows.append(
                {
                    "rule_seed": rule_seed,
                    "halt_window": halt_window,
                    "survived": result["survived"],
                    "termination_reason": result["termination_reason"],
                    "neighbor_mutual_information": result["neighbor_mutual_information"],
                    "mi_shuffle_null": result["mi_shuffle_null"],
                    "delta_mi": result["delta_mi"],
                    "n_pairs": result["n_pairs"],
                    "same_state_adjacency_fraction": result["same_state_adjacency_fraction"],
                    "update_mode": config.update_mode.value,
                    "state_uniform_mode": config.state_uniform_mode.value,
                    "enable_viability_filters": config.enable_viability_filters,
                }
            )

    output_path = logs_dir / "halt_window_sweep.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=HALT_WINDOW_SWEEP_SCHEMA), output_path)
    return output_path
