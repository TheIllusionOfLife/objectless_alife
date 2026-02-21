"""Transfer-entropy shuffle-null calibration analysis.

Usage:
    uv run python scripts/te_null_analysis.py \
      --data-dir data/stage_d \
      --out-dir data/post_hoc/te_null
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import pyarrow.parquet as pq

from objectless_alife.metrics import neighbor_transfer_entropy, transfer_entropy_shuffle_null
from objectless_alife.run_search import select_top_rules_by_delta_mi
from objectless_alife.world import WorldConfig

_DEFAULT_WORLD = WorldConfig()
DEFAULT_GRID_WIDTH = _DEFAULT_WORLD.grid_width
DEFAULT_GRID_HEIGHT = _DEFAULT_WORLD.grid_height


def _sim_log_for_rule_ids(
    sim_log_path: Path, rule_ids: set[str]
) -> dict[str, list[tuple[int, int, int, int, int]]]:
    if not rule_ids:
        return {}
    table = pq.read_table(
        sim_log_path,
        columns=["rule_id", "step", "agent_id", "x", "y", "state"],
        filters=[("rule_id", "in", sorted(rule_ids))],
    )
    rows = table.to_pylist()
    out: dict[str, list[tuple[int, int, int, int, int]]] = {rid: [] for rid in rule_ids}
    for row in rows:
        rid = str(row["rule_id"])
        if rid not in out:
            continue
        out[rid].append(
            (
                int(row["step"]),
                int(row["agent_id"]),
                int(row["x"]),
                int(row["y"]),
                int(row["state"]),
            )
        )
    return out


def _rule_ids_for_phase_from_runs(data_dir: Path, phase: int, seeds: set[int]) -> set[str]:
    runs_path = data_dir / "logs" / "experiment_runs.parquet"
    if not runs_path.exists():
        return set()
    runs = pq.read_table(runs_path, columns=["rule_id", "phase", "rule_seed"]).to_pylist()
    return {
        str(row["rule_id"])
        for row in runs
        if int(row["phase"]) == phase and int(row["rule_seed"]) in seeds
    }


def _rule_ids_for_phase_fallback(phase: int, seeds: set[int]) -> set[str]:
    return {f"phase{phase}_rs{seed}_ss{seed}" for seed in seeds}


def _phase_rules_dir(data_dir: Path, phase: int) -> Path:
    if phase == 3:
        control_rules = data_dir / "control" / "rules"
        if control_rules.exists():
            return control_rules
    return data_dir / f"phase_{phase}" / "rules"


def _grid_dims_for_phase(data_dir: Path, phase: int, rule_ids: set[str]) -> tuple[int, int]:
    rules_dir = _phase_rules_dir(data_dir, phase)
    for rule_id in sorted(rule_ids):
        path = rules_dir / f"{rule_id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        metadata = payload.get("metadata", {})
        grid_w = metadata.get("grid_width")
        grid_h = metadata.get("grid_height")
        if isinstance(grid_w, int) and isinstance(grid_h, int):
            return grid_w, grid_h
    return DEFAULT_GRID_WIDTH, DEFAULT_GRID_HEIGHT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TE shuffle-null calibration analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage_d"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/te_null"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n-shuffles", type=int, default=200)
    parser.add_argument("--quick", action="store_true", help="Run a small sanity-sized preset")
    args = parser.parse_args(argv)
    if args.quick:
        args.top_k = 10
        args.n_shuffles = 20

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p2_metrics = data_dir / "phase_2" / "logs" / "metrics_summary.parquet"
    p2_rules = data_dir / "phase_2" / "rules"
    top_seeds = select_top_rules_by_delta_mi(p2_metrics, p2_rules, top_k=args.top_k)
    top_seed_set = set(top_seeds)

    control_log = data_dir / "control" / "logs" / "simulation_log.parquet"
    if not control_log.exists():
        control_log = data_dir / "phase_3" / "logs" / "simulation_log.parquet"
    conditions = {
        "phase_1": (1, data_dir / "phase_1" / "logs" / "simulation_log.parquet"),
        "phase_2": (2, data_dir / "phase_2" / "logs" / "simulation_log.parquet"),
        "control": (3, control_log),
    }
    summary: dict[str, dict[str, float | int]] = {}

    for label, (phase, sim_log_path) in conditions.items():
        rule_ids = _rule_ids_for_phase_from_runs(data_dir, phase, top_seed_set)
        if not rule_ids:
            rule_ids = _rule_ids_for_phase_fallback(phase, top_seed_set)
        grid_width, grid_height = _grid_dims_for_phase(data_dir, phase, rule_ids)
        logs_by_rule = _sim_log_for_rule_ids(sim_log_path, rule_ids)
        te_vals: list[float] = []
        te_null_vals: list[float] = []
        te_excess_vals: list[float] = []
        for idx, rule_id in enumerate(sorted(logs_by_rule)):
            tuples = logs_by_rule[rule_id]
            if not tuples:
                continue
            te = neighbor_transfer_entropy(tuples, grid_width, grid_height)
            te_null = transfer_entropy_shuffle_null(
                tuples,
                grid_width,
                grid_height,
                n_shuffles=args.n_shuffles,
                rng=random.Random(idx),
            )
            te_excess = max(te - te_null, 0.0)
            te_vals.append(te)
            te_null_vals.append(te_null)
            te_excess_vals.append(te_excess)
        summary[label] = {
            "n_rules": len(te_vals),
            "te_median": statistics.median(te_vals) if te_vals else 0.0,
            "te_null_median": statistics.median(te_null_vals) if te_null_vals else 0.0,
            "te_excess_median": statistics.median(te_excess_vals) if te_excess_vals else 0.0,
        }

    payload = {
        "top_k": args.top_k,
        "n_shuffles": args.n_shuffles,
        "conditions": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    csv_rows = [
        {
            "condition": condition,
            "n_rules": int(stats["n_rules"]),
            "te_median": float(stats["te_median"]),
            "te_null_median": float(stats["te_null_median"]),
            "te_excess_median": float(stats["te_excess_median"]),
        }
        for condition, stats in summary.items()
    ]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "n_rules",
                "te_median",
                "te_null_median",
                "te_excess_median",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
