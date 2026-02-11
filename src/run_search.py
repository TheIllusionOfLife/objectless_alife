from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.filters import (
    HaltDetector,
    LowActivityDetector,
    ShortPeriodDetector,
    StateUniformDetector,
    TerminationReason,
)
from src.metrics import (
    action_entropy,
    action_entropy_variance,
    block_ncd,
    cluster_count_by_state,
    compression_ratio,
    morans_i_occupied,
    neighbor_mutual_information,
    normalized_hamming_distance,
    phase_transition_max_delta,
    quasi_periodicity_peak_count,
    serialize_snapshot,
    state_entropy,
)
from src.rules import ObservationPhase, generate_rule_table
from src.world import World, WorldConfig


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


def _deterministic_rule_id(phase: ObservationPhase, rule_seed: int, sim_seed: int) -> str:
    """Build reproducible rule ID stable across runs for identical seeds."""
    return f"phase{phase.value}_rs{rule_seed}_ss{sim_seed}"


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

    simulation_rows: list[dict[str, int | str]] = []
    metric_rows: list[dict[str, int | str | float | None]] = []
    results: list[SimulationResult] = []

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
        snapshot_bytes_history: list[bytes] = []
        per_agent_actions: list[list[int]] = [[] for _ in range(world_cfg.num_agents)]
        halt_triggered = False
        uniform_triggered = False
        short_period_triggered = False
        low_activity_triggered = False

        for step in range(world_cfg.steps):
            actions = world.step(rule_table, phase)
            snapshot = world.snapshot()
            states = world.state_vector()
            snapshot_bytes = serialize_snapshot(
                snapshot, world_cfg.grid_width, world_cfg.grid_height
            )
            step_entropy = state_entropy(states)
            entropy_series.append(step_entropy)
            snapshot_bytes_history.append(snapshot_bytes)
            for agent_id, action in enumerate(actions):
                per_agent_actions[agent_id].append(action)

            predictability = (
                None if prev_states is None else normalized_hamming_distance(prev_states, states)
            )
            block_ncd_value: float | None = None
            window = search_config.block_ncd_window
            if window > 0 and len(snapshot_bytes_history) >= window * 2:
                prev_block = b"".join(snapshot_bytes_history[-2 * window : -window])
                curr_block = b"".join(snapshot_bytes_history[-window:])
                block_ncd_value = block_ncd(prev_block, curr_block)

            per_agent_entropies = [action_entropy(history) for history in per_agent_actions]
            action_entropy_mean = (
                sum(per_agent_entropies) / len(per_agent_entropies) if per_agent_entropies else 0.0
            )

            metric_rows.append(
                {
                    "rule_id": rule_id,
                    "step": step,
                    "state_entropy": step_entropy,
                    "compression_ratio": compression_ratio(snapshot_bytes),
                    "predictability_hamming": predictability,
                    "morans_i": morans_i_occupied(
                        snapshot, grid_width=world_cfg.grid_width, grid_height=world_cfg.grid_height
                    ),
                    "cluster_count": cluster_count_by_state(
                        snapshot, grid_width=world_cfg.grid_width, grid_height=world_cfg.grid_height
                    ),
                    "quasi_periodicity_peaks": quasi_periodicity_peak_count(entropy_series),
                    "phase_transition_max_delta": phase_transition_max_delta(entropy_series),
                    "neighbor_mutual_information": neighbor_mutual_information(
                        snapshot, grid_width=world_cfg.grid_width, grid_height=world_cfg.grid_height
                    ),
                    "action_entropy_mean": action_entropy_mean,
                    "action_entropy_variance": action_entropy_variance(per_agent_actions),
                    "block_ncd": block_ncd_value,
                }
            )

            for agent_id, x, y, state in snapshot:
                simulation_rows.append(
                    {
                        "rule_id": rule_id,
                        "step": step,
                        "agent_id": agent_id,
                        "x": x,
                        "y": y,
                        "state": state,
                        "action": actions[agent_id],
                    }
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
                termination_reason = "short_period"
                break
            elif low_activity_triggered:
                terminated_at = step
                termination_reason = "low_activity"
                break

            prev_states = states

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

    pq.write_table(pa.Table.from_pylist(simulation_rows), logs_dir / "simulation_log.parquet")
    pq.write_table(pa.Table.from_pylist(metric_rows), logs_dir / "metrics_summary.parquet")
    return results


def _parse_phase(raw_phase: int) -> ObservationPhase:
    """Parse CLI phase value into ObservationPhase enum."""
    try:
        return ObservationPhase(raw_phase)
    except ValueError as exc:
        raise ValueError("phase must be 1 or 2") from exc


def main() -> None:
    """CLI entrypoint for search execution."""
    parser = argparse.ArgumentParser(description="Run objective-free ALife search")
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--n-rules", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--halt-window", type=int, default=10)
    parser.add_argument("--rule-seed", type=int, default=0)
    parser.add_argument("--sim-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--filter-short-period", action="store_true")
    parser.add_argument("--short-period-max-period", type=int, default=2)
    parser.add_argument("--short-period-history-size", type=int, default=8)
    parser.add_argument("--filter-low-activity", action="store_true")
    parser.add_argument("--low-activity-window", type=int, default=5)
    parser.add_argument("--low-activity-min-unique-ratio", type=float, default=0.2)
    parser.add_argument("--block-ncd-window", type=int, default=10)
    args = parser.parse_args()

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
        "total_rules": len(results),
        "survived": sum(1 for r in results if r.survived),
        "terminated": sum(1 for r in results if not r.survived),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
