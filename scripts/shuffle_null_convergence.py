"""Shuffle-null MI convergence analysis.

Evaluates how shuffle-null MI stabilises as N_shuffles increases,
producing a convergence plot for the supplementary material.

Usage:
    uv run python scripts/shuffle_null_convergence.py
    uv run python scripts/shuffle_null_convergence.py --top-k 50 --seed 42
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow.parquet as pq  # noqa: E402

from objectless_alife.metrics import shuffle_null_mi  # noqa: E402
from objectless_alife.run_search import select_top_rules_by_delta_mi  # noqa: E402
from scripts._common import load_final_snapshots  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "stage_d"
DEFAULT_N_SHUFFLES = 200
GRID_W, GRID_H = 20, 20
N_SWEEP_VALUES: tuple[int, ...] = (10, 25, 50, 100, 200, 500)


def run_convergence_analysis(
    snapshots: dict[str, tuple[tuple[int, int, int, int], ...]],
    n_values: list[int],
    seed: int,
    grid_width: int = GRID_W,
    grid_height: int = GRID_H,
) -> dict[int, dict[str, float]]:
    """Compute mean and std of shuffle-null MI across snapshots for each N.

    Returns ``{N: {"mean": float, "std": float}}``.
    """
    result: dict[int, dict[str, float]] = {}
    snapshot_list = list(snapshots.values())

    for n in n_values:
        mi_values: list[float] = []
        for i, snap in enumerate(snapshot_list):
            rng = random.Random(seed + i)
            mi_null = shuffle_null_mi(snap, grid_width, grid_height, n_shuffles=n, rng=rng)
            mi_values.append(mi_null)
        mean_val = statistics.mean(mi_values) if mi_values else float("nan")
        std_val = statistics.stdev(mi_values) if len(mi_values) >= 2 else float("nan")
        result[n] = {"mean": mean_val, "std": std_val}

    return result


def plot_convergence(
    result: dict[int, dict[str, float]],
    output_path: Path,
) -> None:
    """Plot shuffle-null MI convergence and save to *output_path*."""
    n_vals = sorted(result.keys())
    means = [result[n]["mean"] for n in n_vals]
    stds = [result[n]["std"] for n in n_vals]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_vals, means, marker="o", linewidth=1.5, color="steelblue")
    ax.fill_between(
        n_vals,
        [m - s for m, s in zip(means, stds, strict=True)],
        [m + s for m, s in zip(means, stds, strict=True)],
        alpha=0.25,
        color="steelblue",
    )
    ax.axvline(DEFAULT_N_SHUFFLES, linestyle="--", color="grey", linewidth=1, label="default")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Shuffles (N)")
    ax.set_ylabel("Mean Shuffle-Null MI (bits)")
    ax.set_title("Shuffle-Null MI Convergence")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def main(argv: list[str] | None = None) -> None:
    """Run the shuffle-null MI convergence analysis and save the output figure."""
    parser = argparse.ArgumentParser(description="Shuffle-null MI convergence analysis")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top rules to use")
    parser.add_argument(
        "--output-dir", type=str, default="paper/figures/", help="Output directory for figure"
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    args = parser.parse_args(argv)

    metrics_path = DATA_DIR / "phase_2" / "logs" / "metrics_summary.parquet"
    rules_dir = DATA_DIR / "phase_2" / "rules"
    sim_log_path = DATA_DIR / "phase_2" / "logs" / "simulation_log.parquet"

    print("Selecting top-k Phase 2 rules...")
    top_seeds = select_top_rules_by_delta_mi(metrics_path, rules_dir, top_k=args.top_k)
    print(f"  Found {len(top_seeds)} seeds")

    # Derive exact rule_ids from experiment_runs to avoid assuming rule_seed==sim_seed.
    exp_parquet = DATA_DIR / "phase_2" / "logs" / "experiment_runs.parquet"
    top_seed_set = set(top_seeds)
    if exp_parquet.exists():
        exp_rows = pq.read_table(exp_parquet, columns=["rule_id", "rule_seed"]).to_pylist()
        rule_ids = {
            str(r["rule_id"])
            for r in exp_rows
            if r["rule_seed"] is not None and int(r["rule_seed"]) in top_seed_set
        }
    else:
        rule_ids = {f"phase2_rs{s}_ss{s}" for s in top_seeds}
    print("Loading final-step snapshots...")
    snapshots = load_final_snapshots(sim_log_path, rule_ids)
    print(f"  Loaded {len(snapshots)} snapshots")

    n_values = list(N_SWEEP_VALUES)
    print("Running convergence analysis...")
    result = run_convergence_analysis(snapshots, n_values, seed=args.seed)

    # Print summary table
    print(f"\n{'N_shuffles':>12}  {'Mean MI_null':>12}  {'Std MI_null':>12}")
    print("-" * 42)
    for n in n_values:
        stats = result[n]
        print(f"{n:>12d}  {stats['mean']:>12.6f}  {stats['std']:>12.6f}")

    output_dir = Path(args.output_dir)
    plot_convergence(result, output_dir / "figO1_shuffle_null_convergence.pdf")


if __name__ == "__main__":
    main()
