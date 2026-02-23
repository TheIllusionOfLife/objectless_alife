"""Population-level per-step delta-MI(t) trajectories.

For each of P1, P2, Control: draw 500 random surviving rules, compute
per-step delta_MI(t) = MI(t) - mean_null(t) at t in {0,10,20,...,190,199},
plot median +/- IQR across rules.

Usage:
    uv run python scripts/population_mi_timeseries.py
    uv run python scripts/population_mi_timeseries.py --n-rules 100 --n-shuffles 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow.parquet as pq  # noqa: E402

from objectless_alife.metrics import (  # noqa: E402
    neighbor_mutual_information,
    shuffle_null_mi,
)

DATA_DIR = PROJECT_ROOT / "data" / "stage_d"
GRID_W, GRID_H = 20, 20

TIMESTEPS = sorted(set(list(range(0, 200, 10)) + [199]))

CONDITION_LABELS = {
    "phase_1": "Phase 1",
    "phase_2": "Phase 2",
    "control": "Control",
}
CONDITION_COLORS = {
    "phase_1": "steelblue",
    "phase_2": "darkorange",
    "control": "seagreen",
}


def get_survivor_rule_ids(condition: str) -> list[str]:
    """Return surviving rule_ids for a given condition."""
    if condition in ("phase_1", "phase_2"):
        phase_num = 1 if condition == "phase_1" else 2
        exp_path = DATA_DIR / "logs" / "experiment_runs.parquet"
        table = pq.read_table(exp_path, columns=["rule_id", "phase", "survived"])
        rows = table.to_pylist()
        return [r["rule_id"] for r in rows if r["phase"] == phase_num and r["survived"]]
    else:  # control
        metrics_path = DATA_DIR / "control" / "logs" / "metrics_summary.parquet"
        table = pq.read_table(metrics_path, columns=["rule_id", "step"])
        rows = table.to_pylist()
        max_steps: dict[str, int] = {}
        for r in rows:
            rid = r["rule_id"]
            step = r["step"]
            if rid not in max_steps or step > max_steps[rid]:
                max_steps[rid] = step
        return [rid for rid, ms in max_steps.items() if ms == 199]


def load_all_step_snapshots(
    sim_log_path: Path,
    rule_ids: set[str],
) -> dict[str, dict[int, tuple[tuple[int, int, int, int], ...]]]:
    """Load all step snapshots for given rule_ids.

    Returns dict mapping rule_id -> {step -> snapshot_tuple}.
    """
    table = pq.read_table(
        sim_log_path,
        columns=["rule_id", "step", "agent_id", "x", "y", "state"],
        filters=[("rule_id", "in", list(rule_ids))],
    )

    if len(table) == 0:
        return {}

    # Build grouped structure
    rid_col = table.column("rule_id").to_pylist()
    step_col = table.column("step").to_pylist()
    aid_col = table.column("agent_id").to_pylist()
    x_col = table.column("x").to_pylist()
    y_col = table.column("y").to_pylist()
    state_col = table.column("state").to_pylist()

    grouped: dict[str, dict[int, list[tuple[int, int, int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rid, step, aid, x, y, s in zip(
        rid_col, step_col, aid_col, x_col, y_col, state_col, strict=True
    ):
        grouped[rid][step].append((int(aid), int(x), int(y), int(s)))

    return {
        rid: {step: tuple(agents) for step, agents in steps_dict.items()}
        for rid, steps_dict in grouped.items()
    }


def compute_population_delta_mi_timeseries(
    snapshots: dict[str, dict[int, tuple[tuple[int, int, int, int], ...]]],
    timesteps: list[int],
    n_shuffles: int,
    seed: int,
) -> dict[int, dict[str, float]]:
    """Compute per-timestep delta_MI statistics across all rules.

    Returns {timestep: {"median": float, "q25": float, "q75": float}}.
    """
    result: dict[int, dict[str, float]] = {}

    for t in timesteps:
        delta_mi_values: list[float] = []
        for i, (_rid, steps_dict) in enumerate(sorted(snapshots.items())):
            # Find snapshot at step t; if rule terminated before t, use last available step
            available_steps = sorted(steps_dict.keys())
            if not available_steps:
                continue
            if t in steps_dict:
                snap = steps_dict[t]
            else:
                steps_before_t = [s for s in available_steps if s <= t]
                if not steps_before_t:
                    continue  # No snapshot at or before t; skip this rule for this timestep
                snap = steps_dict[max(steps_before_t)]

            rng = random.Random(seed + i * 1000 + t)
            mi = neighbor_mutual_information(snap, GRID_W, GRID_H)
            mi_null = shuffle_null_mi(snap, GRID_W, GRID_H, n_shuffles=n_shuffles, rng=rng)
            delta_mi_values.append(mi - mi_null)

        if delta_mi_values:
            sorted_vals = sorted(delta_mi_values)
            n = len(sorted_vals)
            result[t] = {
                "median": statistics.median(sorted_vals),
                "q25": sorted_vals[max(0, int(n * 0.25))],
                "q75": sorted_vals[min(n - 1, int(n * 0.75))],
            }
        else:
            result[t] = {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}

    return result


def plot_timeseries(
    all_stats: dict[str, dict[int, dict[str, float]]],
    output_path: Path,
) -> None:
    """Plot per-condition delta_MI(t) timeseries with IQR bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for condition in ("phase_1", "phase_2", "control"):
        stats = all_stats[condition]
        ts = sorted(stats.keys())
        medians = [stats[t]["median"] for t in ts]
        q25s = [stats[t]["q25"] for t in ts]
        q75s = [stats[t]["q75"] for t in ts]
        color = CONDITION_COLORS[condition]
        label = CONDITION_LABELS[condition]

        ax.plot(ts, medians, marker=".", linewidth=1.5, color=color, label=label, markersize=4)
        ax.fill_between(ts, q25s, q75s, alpha=0.2, color=color)

    ax.axhline(0, linestyle="--", color="grey", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"$\Delta$MI (bits)")
    ax.set_title(r"Population-level $\Delta$MI(t) trajectories (median $\pm$ IQR)")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Population-level delta-MI(t) timeseries")
    parser.add_argument("--n-rules", type=int, default=500, help="Rules to sample per condition")
    parser.add_argument("--n-shuffles", type=int, default=10, help="Shuffle-null iterations per MI")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument(
        "--output-dir", type=str, default="paper/figures/", help="Output directory for figure"
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    all_stats: dict[str, dict[int, dict[str, float]]] = {}
    summary_data: dict[str, dict] = {}

    sim_log_paths = {
        "phase_1": DATA_DIR / "phase_1" / "logs" / "simulation_log.parquet",
        "phase_2": DATA_DIR / "phase_2" / "logs" / "simulation_log.parquet",
        "control": DATA_DIR / "control" / "logs" / "simulation_log.parquet",
    }

    for condition in ("phase_1", "phase_2", "control"):
        print(f"\n--- {CONDITION_LABELS[condition]} ---")
        survivor_ids = get_survivor_rule_ids(condition)
        print(f"  Total survivors: {len(survivor_ids)}")

        if len(survivor_ids) > args.n_rules:
            sampled_ids = rng.sample(survivor_ids, args.n_rules)
        else:
            sampled_ids = survivor_ids
        print(f"  Sampled: {len(sampled_ids)}")

        print("  Loading simulation logs...")
        snapshots = load_all_step_snapshots(sim_log_paths[condition], set(sampled_ids))
        print(f"  Loaded snapshots for {len(snapshots)} rules")

        print("  Computing per-step delta_MI...")
        stats = compute_population_delta_mi_timeseries(
            snapshots, TIMESTEPS, n_shuffles=args.n_shuffles, seed=args.seed
        )
        all_stats[condition] = stats

        # Fraction of timesteps with positive median
        positive_median_count = sum(1 for t in TIMESTEPS if stats[t]["median"] > 0)
        frac_positive = positive_median_count / len(TIMESTEPS)
        print(f"  Fraction of timesteps with positive median delta_MI: {frac_positive:.2%}")

        summary_data[condition] = {
            "total_survivors": len(survivor_ids),
            "n_sampled": len(sampled_ids),
            "fraction_timesteps_positive_median": round(frac_positive, 4),
        }

    # Plot
    output_dir = Path(args.output_dir)
    plot_timeseries(all_stats, output_dir / "figQ1_population_delta_mi_timeseries.pdf")

    # Save summary JSON
    summary_dir = PROJECT_ROOT / "data" / "post_hoc" / "population_mi_timeseries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "n_rules_per_condition": args.n_rules,
                "n_shuffles": args.n_shuffles,
                "seed": args.seed,
                "timesteps": TIMESTEPS,
                "conditions": summary_data,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
