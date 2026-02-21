"""MI vs neighbor-pair-count analysis across conditions.

For each condition, loads simulation logs and metrics, computes
neighbor_pair_count per rule, bins rules by pair count, and plots
median delta_mi per bin with bootstrap CIs.

Usage:
    uv run python scripts/mi_vs_n_pairs_analysis.py [--output-dir DIR] [--seed N] [--n-bootstrap N]
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

from objectless_alife.metrics import neighbor_pair_count  # noqa: E402
from objectless_alife.stats import load_final_step_metrics  # noqa: E402
from scripts._common import load_final_snapshots  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "stage_d"
GRID_W, GRID_H = 20, 20
DEFAULT_N_BOOTSTRAPS = 2000

CONDITIONS: dict[str, str] = {
    "phase_2": "Phase 2",
    "phase_1": "Phase 1",
    "control": "Control",
    "random_walk": "RW",
}

DEFAULT_BINS: list[tuple[int, int | None, str]] = [
    (1, 3, "1\u20133"),
    (4, 6, "4\u20136"),
    (7, 12, "7\u201312"),
    (13, None, "\u226513"),
]


def assign_bin(n_pairs: int, bins: list[tuple[int, int | None, str]]) -> str | None:
    """Return the bin label for *n_pairs*, or None if out of range."""
    for lo, hi, label in bins:
        if hi is None:
            if n_pairs >= lo:
                return label
        else:
            if lo <= n_pairs <= hi:
                return label
    return None


def bootstrap_median_ci_single(
    values: list[float],
    n_bootstrap: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI for the median of *values*.

    Returns ``(nan, nan)`` when *values* is empty or *n_bootstrap* is zero.
    """
    if not values:
        return float("nan"), float("nan")
    if n_bootstrap <= 0:
        return float("nan"), float("nan")
    n = len(values)
    medians: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo_idx = max(0, int(n_bootstrap * 0.025))
    hi_idx = min(n_bootstrap - 1, int(n_bootstrap * 0.975))
    return medians[lo_idx], medians[hi_idx]


def compute_bin_stats(
    data: list[tuple[int, float]],
    bins: list[tuple[int, int | None, str]],
    n_bootstrap: int,
    rng: random.Random,
) -> dict[str, dict]:
    """Bin (n_pairs, delta_mi) pairs and compute per-bin median + CI.

    Returns ``{bin_label: {"median": float, "ci_low": float, "ci_high": float, "n": int}}``.
    """
    binned: dict[str, list[float]] = {}
    for n_pairs, delta_mi in data:
        label = assign_bin(n_pairs, bins)
        if label is None:
            continue
        binned.setdefault(label, []).append(delta_mi)

    result: dict[str, dict] = {}
    for _, _, label in bins:
        vals = binned.get(label, [])
        if not vals:
            continue
        med = statistics.median(vals)
        if len(vals) >= 2:
            ci_lo, ci_hi = bootstrap_median_ci_single(vals, n_bootstrap, rng)
        else:
            ci_lo, ci_hi = med, med
        result[label] = {"median": med, "ci_low": ci_lo, "ci_high": ci_hi, "n": len(vals)}
    return result


def _plot(
    all_stats: dict[str, dict[str, dict]],
    bins: list[tuple[int, int | None, str]],
    output_path: Path,
) -> None:
    """Grouped bar chart: one group per bin, one bar per condition."""
    bin_labels = [label for _, _, label in bins]
    cond_names = list(all_stats.keys())
    n_conds = len(cond_names)
    bar_width = 0.8 / max(n_conds, 1)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, cond in enumerate(cond_names):
        stats = all_stats[cond]
        xs, ys, errs = [], [], []
        for j, bl in enumerate(bin_labels):
            if bl not in stats:
                continue
            s = stats[bl]
            xs.append(j + (i - n_conds / 2 + 0.5) * bar_width)
            ys.append(s["median"])
            errs.append([s["median"] - s["ci_low"], s["ci_high"] - s["median"]])
        if xs:
            err_lo = [max(0.0, e[0]) for e in errs]
            err_hi = [max(0.0, e[1]) for e in errs]
            ax.bar(
                xs,
                ys,
                width=bar_width,
                label=cond,
                color=colors[i % len(colors)],
                yerr=[err_lo, err_hi],
                capsize=3,
            )

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Neighbor pair count")
    ax.set_ylabel("Median \u0394MI")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Run the MI vs n_pairs analysis across all conditions and save the figure."""
    parser = argparse.ArgumentParser(description="MI vs neighbor-pair-count analysis")
    parser.add_argument("--output-dir", type=str, default="paper/figures/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAPS)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    all_stats: dict[str, dict[str, dict]] = {}

    print("condition,bin,n_rules,median_delta_mi,ci_low,ci_high")
    for cond_dir, cond_label in CONDITIONS.items():
        sim_log_path = DATA_DIR / cond_dir / "logs" / "simulation_log.parquet"
        metrics_path = DATA_DIR / cond_dir / "logs" / "metrics_summary.parquet"

        # Load final-step snapshots and compute pair counts
        snapshots = load_final_snapshots(sim_log_path)
        pair_counts: dict[str, int] = {}
        for rid, snap in snapshots.items():
            pair_counts[rid] = neighbor_pair_count(snap, GRID_W, GRID_H)

        # Load delta_mi per rule
        metrics_table = load_final_step_metrics(metrics_path)
        mi_col = metrics_table.column("delta_mi").to_pylist()
        rid_col = metrics_table.column("rule_id").to_pylist()
        delta_mi_map: dict[str, float] = {}
        for rid, dmi in zip(rid_col, mi_col, strict=True):
            if dmi is not None:
                delta_mi_map[rid] = float(dmi)

        # Join
        data: list[tuple[int, float]] = []
        for rid in pair_counts:
            if rid in delta_mi_map:
                data.append((pair_counts[rid], delta_mi_map[rid]))

        stats = compute_bin_stats(data, DEFAULT_BINS, args.n_bootstrap, rng)
        all_stats[cond_label] = stats

        for _, _, label in DEFAULT_BINS:
            if label in stats:
                s = stats[label]
                print(
                    f"{cond_label},{label},{s['n']},{s['median']:.6f},{s['ci_low']:.6f},"
                    f"{s['ci_high']:.6f}"
                )

    _plot(all_stats, DEFAULT_BINS, output_dir / "figP1_mi_vs_n_pairs.pdf")
    print(f"\nFigure saved to {output_dir / 'figP1_mi_vs_n_pairs.pdf'}", file=sys.stderr)


if __name__ == "__main__":
    main()
