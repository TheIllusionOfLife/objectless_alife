from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from src.filters import HaltDetector, StateUniformDetector, TerminationReason
from src.metrics import (
    compression_ratio,
    normalized_hamming_distance,
    serialize_snapshot,
    state_entropy,
)
from src.rules import ObservationPhase, generate_rule_table
from src.world import World, WorldConfig


@dataclass(frozen=True)
class SimulationResult:
    rule_id: str
    survived: bool
    terminated_at: int | None
    termination_reason: str | None


def run_batch_search(
    n_rules: int,
    phase: ObservationPhase,
    out_dir: Path,
    steps: int = 200,
    halt_window: int = 10,
    base_rule_seed: int = 0,
    base_sim_seed: int = 0,
    world_config: WorldConfig | None = None,
) -> list[SimulationResult]:
    if n_rules < 1:
        raise ValueError("n_rules must be >= 1")

    out_dir = Path(out_dir)
    rules_dir = out_dir / "rules"
    logs_dir = out_dir / "logs"
    rules_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config = world_config or WorldConfig(steps=steps)

    simulation_rows: list[dict[str, int | str]] = []
    metric_rows: list[dict[str, int | str | float]] = []
    results: list[SimulationResult] = []

    for i in range(n_rules):
        rule_id = str(uuid4())
        rule_seed = base_rule_seed + i
        sim_seed = base_sim_seed + i

        rule_table = generate_rule_table(phase=phase, seed=rule_seed)
        world = World(config=config, sim_seed=sim_seed)
        halt_detector = HaltDetector(window=halt_window)
        uniform_detector = StateUniformDetector()

        terminated_at: int | None = None
        termination_reason: str | None = None
        prev_states: list[int] | None = None
        halt_triggered = False
        uniform_triggered = False

        for step in range(config.steps):
            actions = world.step(rule_table, phase)
            snapshot = world.snapshot()
            states = world.state_vector()

            predictability = (
                0.0 if prev_states is None else normalized_hamming_distance(prev_states, states)
            )
            metric_rows.append(
                {
                    "rule_id": rule_id,
                    "step": step,
                    "state_entropy": state_entropy(states),
                    "compression_ratio": compression_ratio(
                        serialize_snapshot(snapshot, config.grid_width, config.grid_height)
                    ),
                    "predictability_hamming": predictability,
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
            if uniform_triggered:
                terminated_at = step
                termination_reason = TerminationReason.STATE_UNIFORM.value
                break
            if halt_triggered:
                terminated_at = step
                termination_reason = TerminationReason.HALT.value
                break

            prev_states = states

        survived = termination_reason is None

        rule_payload = {
            "rule_id": rule_id,
            "table": rule_table,
            "survived": survived,
            "filter_results": {"halt": halt_triggered, "state_uniform": uniform_triggered},
            "metadata": {
                "rule_seed": rule_seed,
                "sim_seed": sim_seed,
                "steps": config.steps,
                "halt_window": halt_window,
                "observation_phase": phase.value,
                "terminated_at": terminated_at,
                "termination_reason": termination_reason,
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
    if raw_phase == 1:
        return ObservationPhase.PHASE1_DENSITY
    if raw_phase == 2:
        return ObservationPhase.PHASE2_PROFILE
    raise ValueError("phase must be 1 or 2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run objective-free ALife search")
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--n-rules", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--halt-window", type=int, default=10)
    parser.add_argument("--rule-seed", type=int, default=0)
    parser.add_argument("--sim-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    phase = _parse_phase(args.phase)
    results = run_batch_search(
        n_rules=args.n_rules,
        phase=phase,
        out_dir=args.out_dir,
        steps=args.steps,
        halt_window=args.halt_window,
        base_rule_seed=args.rule_seed,
        base_sim_seed=args.sim_seed,
    )

    summary = {
        "total_rules": len(results),
        "survived": sum(1 for r in results if r.survived),
        "terminated": sum(1 for r in results if not r.survived),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
