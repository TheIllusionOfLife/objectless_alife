"""Statistical significance tests for phase comparison experiments.

Provides Mann-Whitney U tests for metric comparisons and chi-squared tests
for survival rate differences between observation phases.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr

from src.run_search import PHASE_SUMMARY_METRIC_NAMES


def bootstrap_median_ci(
    vals1: list[float],
    vals2: list[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the median difference (vals2 - vals1).

    Returns (lower, upper) percentile interval at the given *ci_level*.
    """
    if not vals1 or not vals2:
        return (float("nan"), float("nan"))

    if rng is None:
        rng = random.Random()

    alpha = 1.0 - ci_level
    diffs: list[float] = []
    n1, n2 = len(vals1), len(vals2)
    for _ in range(n_bootstrap):
        sample1 = [vals1[rng.randrange(n1)] for _ in range(n1)]
        sample2 = [vals2[rng.randrange(n2)] for _ in range(n2)]
        med1 = statistics.median(sample1)
        med2 = statistics.median(sample2)
        diffs.append(med2 - med1)

    diffs.sort()
    lo_idx = int(n_bootstrap * (alpha / 2.0))
    hi_idx = int(n_bootstrap * (1.0 - alpha / 2.0)) - 1
    return diffs[lo_idx], diffs[hi_idx]


def load_final_step_metrics(parquet_path: Path) -> pa.Table:
    """Load metrics_summary.parquet and return one row per rule at its final step."""
    table = pq.read_table(parquet_path)
    rule_ids = table.column("rule_id")
    steps = table.column("step")

    # Find max step per rule_id
    max_steps: dict[str, int] = {}
    for i in range(table.num_rows):
        rid = rule_ids[i].as_py()
        step = steps[i].as_py()
        if rid not in max_steps or step > max_steps[rid]:
            max_steps[rid] = step

    # Filter to final-step rows
    mask = [
        rule_ids[i].as_py() in max_steps and steps[i].as_py() == max_steps[rule_ids[i].as_py()]
        for i in range(table.num_rows)
    ]
    return table.filter(mask)


def phase_comparison_tests(
    phase1_metrics: pa.Table,
    phase2_metrics: pa.Table,
    metrics_to_test: list[str],
) -> dict:
    """Run Mann-Whitney U test for each metric between two phases.

    Returns a dict keyed by metric name. Metrics with insufficient non-null
    data are silently skipped.
    """
    results: dict[str, dict] = {}

    for metric in metrics_to_test:
        if metric not in phase1_metrics.column_names or metric not in phase2_metrics.column_names:
            continue
        col1 = phase1_metrics.column(metric)
        col2 = phase2_metrics.column(metric)

        # Drop nulls and NaNs
        vals1 = [v for v in col1.to_pylist() if v is not None and v == v]
        vals2 = [v for v in col2.to_pylist() if v is not None and v == v]

        if len(vals1) < 2 or len(vals2) < 2:
            continue

        u_stat, p_value = mannwhitneyu(vals1, vals2, alternative="two-sided")
        n1, n2 = len(vals1), len(vals2)
        # Rank-biserial correlation: r = 1 - (2U)/(n1*n2)
        effect_size_r = 1.0 - (2.0 * u_stat) / (n1 * n2)

        ci_lo, ci_hi = bootstrap_median_ci(
            vals1, vals2, n_bootstrap=10000, rng=random.Random(zlib.adler32(metric.encode()))
        )

        results[metric] = {
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "effect_size_r": float(effect_size_r),
            "cliffs_delta": float(effect_size_r),
            "median_diff_ci_lower": float(ci_lo),
            "median_diff_ci_upper": float(ci_hi),
            "n_phase1": n1,
            "n_phase2": n2,
            "phase1_median": float(statistics.median(vals1)),
            "phase2_median": float(statistics.median(vals2)),
        }

    # Holm-Bonferroni correction for multiple comparisons
    if results:
        metrics_tested = list(results.keys())
        raw_pvals = [results[m]["p_value"] for m in metrics_tested]
        corrected = _holm_bonferroni(raw_pvals)
        for m, cp in zip(metrics_tested, corrected, strict=True):
            results[m]["p_value_corrected"] = cp

    return results


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values."""
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value, keeping original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    cumulative_max = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adjusted = pval * (n - rank)
        # Enforce monotonicity: corrected p-values never decrease in sorted order
        cumulative_max = max(cumulative_max, adjusted)
        corrected[orig_idx] = min(cumulative_max, 1.0)
    return corrected


def survival_rate_test(runs_table: pa.Table) -> dict:
    """Chi-squared test on survival counts between phases (2x2 contingency)."""
    phases = runs_table.column("phase").to_pylist()
    survived = runs_table.column("survived").to_pylist()

    counts: dict[int, dict[str, int]] = {}
    for phase, surv in zip(phases, survived, strict=True):
        p = int(phase)
        if p not in counts:
            counts[p] = {"survived": 0, "terminated": 0}
        if surv:
            counts[p]["survived"] += 1
        else:
            counts[p]["terminated"] += 1

    sorted_phases = sorted(counts.keys())
    if len(sorted_phases) != 2:
        raise ValueError(f"Expected exactly 2 phases, found {len(sorted_phases)}: {sorted_phases}")
    p1, p2 = sorted_phases[0], sorted_phases[1]

    contingency = [
        [counts[p1]["survived"], counts[p1]["terminated"]],
        [counts[p2]["survived"], counts[p2]["terminated"]],
    ]

    # chi2_contingency fails when any expected frequency is zero (e.g. both
    # phases have identical survival rates of 0% or 100%).  Return NaN in
    # that degenerate case.
    total_survived = counts[p1]["survived"] + counts[p2]["survived"]
    total_terminated = counts[p1]["terminated"] + counts[p2]["terminated"]
    if total_survived == 0 or total_terminated == 0:
        chi2_val = float("nan")
        p_value_val = float("nan")
    else:
        chi2_val, p_value_val, _, _ = chi2_contingency(contingency)

    return {
        "chi2": float(chi2_val),
        "p_value": float(p_value_val),
        "phase1_survived": counts[p1]["survived"],
        "phase1_total": counts[p1]["survived"] + counts[p1]["terminated"],
        "phase2_survived": counts[p2]["survived"],
        "phase2_total": counts[p2]["survived"] + counts[p2]["terminated"],
    }


def pairwise_metric_comparison(
    metrics_path_a: Path,
    metrics_path_b: Path,
    metrics_to_test: list[str],
) -> dict:
    """Run Mann-Whitney U tests between two arbitrary metrics parquet files.

    Loads final-step metrics from each file and runs the same statistical
    pipeline as phase_comparison_tests(), returning results in the same format.
    """
    table_a = load_final_step_metrics(metrics_path_a)
    table_b = load_final_step_metrics(metrics_path_b)
    return phase_comparison_tests(table_a, table_b, metrics_to_test)


def pairwise_survival_comparison(rules_dir_a: Path, rules_dir_b: Path) -> dict:
    """Chi-squared test on survival counts between two arbitrary rule directories."""

    def _count_survival(rules_dir: Path) -> tuple[int, int]:
        survived = 0
        total = 0
        for path in sorted(rules_dir.glob("*.json")):
            data = json.loads(path.read_text())
            total += 1
            if data.get("survived", False):
                survived += 1
        return survived, total

    a_survived, a_total = _count_survival(rules_dir_a)
    b_survived, b_total = _count_survival(rules_dir_b)

    contingency = [
        [a_survived, a_total - a_survived],
        [b_survived, b_total - b_survived],
    ]

    total_survived = a_survived + b_survived
    total_terminated = (a_total - a_survived) + (b_total - b_survived)
    if total_survived == 0 or total_terminated == 0 or a_total == 0 or b_total == 0:
        chi2_val = float("nan")
        p_value_val = float("nan")
    else:
        chi2_val, p_value_val, _, _ = chi2_contingency(contingency)

    return {
        "chi2": float(chi2_val),
        "p_value": float(p_value_val),
        "a_survived": a_survived,
        "a_total": a_total,
        "b_survived": b_survived,
        "b_total": b_total,
    }


def filter_metric_independence(metrics_path: Path, rules_dir: Path) -> dict:
    """Compute point-biserial correlation between survival and final-step MI.

    Tests whether viability filters inadvertently select for high-MI rules.
    Returns correlation, p-value, and per-group medians.
    """
    # Load survival status from rule JSONs
    survival: dict[str, bool] = {}
    for path in sorted(rules_dir.glob("*.json")):
        data = json.loads(path.read_text())
        survival[data["rule_id"]] = bool(data.get("survived", False))

    # Load final-step MI values
    final_metrics = load_final_step_metrics(metrics_path)
    rule_ids = final_metrics.column("rule_id").to_pylist()
    mi_values = final_metrics.column("neighbor_mutual_information").to_pylist()

    # Pair MI with survival, drop nulls/NaN
    surv_flags: list[int] = []
    mi_list: list[float] = []
    survived_mi: list[float] = []
    terminated_mi: list[float] = []

    for rid, mi in zip(rule_ids, mi_values, strict=True):
        if mi is None or mi != mi:
            continue
        surv = survival.get(str(rid))
        if surv is None:
            continue
        surv_flags.append(1 if surv else 0)
        mi_list.append(float(mi))
        if surv:
            survived_mi.append(float(mi))
        else:
            terminated_mi.append(float(mi))

    if len(surv_flags) < 3 or len(set(surv_flags)) < 2:
        return {
            "correlation": float("nan"),
            "p_value": float("nan"),
            "survived_median_mi": float("nan"),
            "terminated_median_mi": float("nan"),
            "n_survived": len(survived_mi),
            "n_terminated": len(terminated_mi),
        }

    corr, pval = pointbiserialr(surv_flags, mi_list)
    return {
        "correlation": float(corr),
        "p_value": float(pval),
        "survived_median_mi": float(statistics.median(survived_mi)) if survived_mi else float("nan"),
        "terminated_median_mi": float(statistics.median(terminated_mi))
        if terminated_mi
        else float("nan"),
        "n_survived": len(survived_mi),
        "n_terminated": len(terminated_mi),
    }


def run_pairwise_analysis(
    metrics_a: Path,
    metrics_b: Path,
    rules_a: Path,
    rules_b: Path,
    label_a: str,
    label_b: str,
) -> dict:
    """Orchestrator combining pairwise metric + survival tests."""
    metric_results = pairwise_metric_comparison(metrics_a, metrics_b, PHASE_SUMMARY_METRIC_NAMES)
    surv_result = pairwise_survival_comparison(rules_a, rules_b)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_a": label_a,
        "label_b": label_b,
        "metric_tests": metric_results,
        "survival_test": surv_result,
    }


def run_statistical_analysis(data_dir: Path) -> dict:
    """Orchestrator: load data, run all tests, return full results dict."""
    data_dir = Path(data_dir)

    # Load experiment runs
    runs_path = data_dir / "logs" / "experiment_runs.parquet"
    runs_table = pq.read_table(runs_path)

    # Load per-phase final-step metrics
    p1_metrics = load_final_step_metrics(data_dir / "phase_1" / "logs" / "metrics_summary.parquet")
    p2_metrics = load_final_step_metrics(data_dir / "phase_2" / "logs" / "metrics_summary.parquet")

    # Metric tests
    metric_results = phase_comparison_tests(p1_metrics, p2_metrics, PHASE_SUMMARY_METRIC_NAMES)

    # Survival test
    surv_result = survival_rate_test(runs_table)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "metric_tests": metric_results,
        "survival_test": surv_result,
    }


def save_results(results: dict, output_path: Path) -> None:
    """Persist results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for statistical analysis."""
    parser = argparse.ArgumentParser(description="Run statistical significance tests")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to experiment data directory (e.g. data/stage_b)",
    )
    mode_group.add_argument(
        "--pairwise",
        action="store_true",
        help="Run pairwise comparison between two arbitrary directories",
    )
    parser.add_argument(
        "--dir-a",
        type=Path,
        default=None,
        help="First data directory for pairwise comparison",
    )
    parser.add_argument(
        "--dir-b",
        type=Path,
        default=None,
        help="Second data directory for pairwise comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path",
    )
    args = parser.parse_args(argv)

    if args.pairwise:
        if not args.dir_a or not args.dir_b:
            parser.error("--pairwise requires --dir-a and --dir-b")
        metrics_a = args.dir_a / "logs" / "metrics_summary.parquet"
        metrics_b = args.dir_b / "logs" / "metrics_summary.parquet"
        rules_a = args.dir_a / "rules"
        rules_b = args.dir_b / "rules"
        label_a = args.dir_a.name
        label_b = args.dir_b.name
        results = run_pairwise_analysis(metrics_a, metrics_b, rules_a, rules_b, label_a, label_b)
        output_path = args.output or (args.dir_a.parent / "logs" / "pairwise_tests.json")
    else:
        output_path = args.output or (args.data_dir / "logs" / "statistical_tests.json")
        results = run_statistical_analysis(args.data_dir)

    save_results(results, output_path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
