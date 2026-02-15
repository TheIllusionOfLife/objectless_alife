"""Cross-condition metric comparison for supplementary Section L.

Extracts final-step values for all metric families across conditions,
computes per-condition summary statistics, and runs pairwise Mann-Whitney U
tests with Holm-Bonferroni correction.

Usage:
    uv run python scripts/metric_comparison.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stats import _holm_bonferroni  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "stage_d"

COMPARISON_METRICS = [
    "compression_ratio",
    "action_entropy_mean",
    "action_entropy_variance",
    "cluster_count",
    "quasi_periodicity_peaks",
    "phase_transition_max_delta",
    "state_entropy",
]

CONDITIONS = [
    ("Phase 1", "phase_1"),
    ("Phase 2", "phase_2"),
    ("Control", "control"),
    ("Random Walk", "random_walk"),
]


def load_survivor_final_metrics(
    metrics_path: Path,
    rules_dir: Path,
) -> list[dict]:
    """Load final-step metrics for surviving rules only.

    Returns a list of metric dicts, one per surviving rule at its final step.
    """
    # Load survival status from rule JSONs
    survived_ids: set[str] = set()
    for path in sorted(rules_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("survived", False):
            survived_ids.add(data["rule_id"])

    # Read metrics and find max step per rule
    table = pq.read_table(metrics_path)
    rows = table.to_pylist()

    max_steps: dict[str, int] = {}
    for row in rows:
        rid = row["rule_id"]
        if rid not in survived_ids:
            continue
        step = int(row["step"])
        if rid not in max_steps or step > max_steps[rid]:
            max_steps[rid] = step

    # Collect final-step rows for survivors
    final_rows: list[dict] = []
    for row in rows:
        rid = row["rule_id"]
        if rid not in survived_ids:
            continue
        if int(row["step"]) != max_steps[rid]:
            continue
        final_rows.append(row)

    return final_rows


def compute_condition_stats(values: list[float]) -> dict:
    """Compute median and IQR for a list of values.

    Returns dict with median, q1, q3, and n.
    """
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan"), "n": 0}

    if len(clean) < 4:
        return {
            "median": statistics.median(clean),
            "q1": min(clean),
            "q3": max(clean),
            "n": len(clean),
        }

    quartiles = statistics.quantiles(clean, n=4)
    return {
        "median": statistics.median(clean),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "n": len(clean),
    }


def compare_conditions(
    condition_values: dict[str, list[float]],
    metric_name: str,
) -> dict:
    """Run pairwise Mann-Whitney U tests between all condition pairs.

    Returns dict keyed by "{label_a} vs {label_b}" with test results.
    """
    labels = list(condition_values.keys())
    results: dict[str, dict] = {}
    raw_pvals: list[float] = []
    pair_keys: list[str] = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            vals_a = [v for v in condition_values[labels[i]] if v is not None and not math.isnan(v)]
            vals_b = [v for v in condition_values[labels[j]] if v is not None and not math.isnan(v)]

            if len(vals_a) < 2 or len(vals_b) < 2:
                continue

            u_stat, p_value = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            n_a, n_b = len(vals_a), len(vals_b)
            cliffs_d = (2.0 * u_stat) / (n_a * n_b) - 1.0

            key = f"{labels[i]} vs {labels[j]}"
            results[key] = {
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "cliffs_delta": float(cliffs_d),
                "n_a": n_a,
                "n_b": n_b,
                "median_a": float(statistics.median(vals_a)),
                "median_b": float(statistics.median(vals_b)),
                "metric": metric_name,
            }
            raw_pvals.append(float(p_value))
            pair_keys.append(key)

    # Apply Holm-Bonferroni correction
    if raw_pvals:
        corrected = _holm_bonferroni(raw_pvals)
        for key, cp in zip(pair_keys, corrected, strict=True):
            results[key]["p_value_corrected"] = cp

    return results


def main() -> None:
    print("Cross-Condition Metric Comparison")
    print("=" * 60)

    # Load survivor final metrics for each condition
    condition_data: dict[str, list[dict]] = {}
    for label, dir_name in CONDITIONS:
        metrics_path = DATA_DIR / dir_name / "logs" / "metrics_summary.parquet"
        rules_dir = DATA_DIR / dir_name / "rules"
        if not metrics_path.exists():
            print(f"  Skipping {label}: {metrics_path} not found")
            continue
        rows = load_survivor_final_metrics(metrics_path, rules_dir)
        condition_data[label] = rows
        print(f"  Loaded {len(rows)} surviving rules for {label}")

    # Per-metric comparison
    all_results: dict[str, dict] = {}
    for metric in COMPARISON_METRICS:
        print(f"\n--- {metric} ---")
        metric_values: dict[str, list[float]] = {}

        for label, rows in condition_data.items():
            values = [
                float(r[metric])
                for r in rows
                if r.get(metric) is not None and float(r[metric]) == float(r[metric])
            ]
            metric_values[label] = values
            stats = compute_condition_stats(values)
            print(
                f"  {label:12s}: median={stats['median']:.4f}, "
                f"Q1={stats['q1']:.4f}, Q3={stats['q3']:.4f}, n={stats['n']}"
            )

        comparisons = compare_conditions(metric_values, metric)
        all_results[metric] = {"stats": {}, "comparisons": comparisons}
        for label, values in metric_values.items():
            all_results[metric]["stats"][label] = compute_condition_stats(values)

        for key, comp in comparisons.items():
            p_corr = comp.get("p_value_corrected", comp["p_value"])
            sig = "*" if p_corr < 0.05 else ""
            print(f"    {key}: delta={comp['cliffs_delta']:.3f}, p={p_corr:.4g} {sig}")

    # Save results
    output_path = PROJECT_ROOT / "data" / "post_hoc" / "metric_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
