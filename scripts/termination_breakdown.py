"""Termination mode breakdown and full-distribution ΔMI analysis.

Aggregates per-condition: N halt, N state_uniform, N survived, totals + percentages.
Also computes full-distribution ΔMI (survived + terminated, MI at last recorded step).

Usage:
    uv run python scripts/termination_breakdown.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from objectless_alife.stats import load_final_step_metrics

DATA_DIR = Path("data/stage_d")
OUTPUT_DIR = Path("data/post_hoc/termination_breakdown")

MAX_STEP = 199


def load_termination_counts(exp_runs_path: Path, phase_value: int) -> dict:
    """Load experiment_runs.parquet and return termination breakdown for a phase.

    Returns dict with n_survived, n_halt, n_state_uniform, n_total.
    """
    table = pq.read_table(exp_runs_path)
    phase_mask = pc.equal(table.column("phase"), phase_value)
    phase_table = table.filter(phase_mask)

    reasons = phase_table.column("termination_reason").to_pylist()
    survived_flags = phase_table.column("survived").to_pylist()

    n_survived = sum(1 for s in survived_flags if s)
    n_halt = sum(1 for r in reasons if r == "halt")
    n_state_uniform = sum(1 for r in reasons if r == "state_uniform")
    n_total = len(reasons)

    return {
        "n_survived": n_survived,
        "n_halt": n_halt,
        "n_state_uniform": n_state_uniform,
        "n_total": n_total,
    }


def load_control_counts(metrics_path: Path) -> dict:
    """Derive control termination counts from metrics_summary.parquet.

    Control has no experiment_runs.parquet, so we infer survival from max step.
    Rules with max_step == 199 are survivors; others terminated with unknown reason.

    Returns dict with n_survived, n_not_recorded, n_total.
    """
    table = pq.read_table(metrics_path)
    grouped = table.group_by("rule_id").aggregate([("step", "max")])
    max_steps = grouped.column("step_max").to_pylist()

    n_survived = sum(1 for s in max_steps if s == MAX_STEP)
    n_total = len(max_steps)

    return {
        "n_survived": n_survived,
        "n_not_recorded": n_total - n_survived,
        "n_total": n_total,
    }


def compute_full_dist_delta_mi(metrics_path: Path) -> dict:
    """Compute full-distribution ΔMI stats (all rules, survived + terminated).

    Uses load_final_step_metrics which returns one row per rule at its last
    recorded step, with delta_mi already computed.

    Returns dict with median and fraction_positive.
    """
    table = load_final_step_metrics(metrics_path)
    delta_mi_col = pc.cast(table.column("delta_mi"), "float64", safe=False)
    finite_mask = pc.and_(pc.is_valid(delta_mi_col), pc.is_finite(delta_mi_col))
    finite_vals = pc.filter(delta_mi_col, finite_mask).to_pylist()

    if not finite_vals:
        return {"median": float("nan"), "fraction_positive": float("nan")}

    median_val = statistics.median(finite_vals)
    fraction_positive = sum(1 for v in finite_vals if v > 0) / len(finite_vals)

    return {
        "median": median_val,
        "fraction_positive": fraction_positive,
    }


def build_summary(data_dir: Path) -> dict:
    """Build full termination breakdown summary for all conditions.

    Saves result to data/post_hoc/termination_breakdown/summary.json.
    Returns the summary dict.
    """
    data_dir = Path(data_dir)
    exp_runs = data_dir / "logs" / "experiment_runs.parquet"

    conditions: dict[str, dict] = {}

    # Phase 1 and Phase 2 from experiment_runs
    for phase_value, label in [(1, "phase_1"), (2, "phase_2")]:
        counts = load_termination_counts(exp_runs, phase_value)
        metrics_path = data_dir / label / "logs" / "metrics_summary.parquet"
        delta_mi = compute_full_dist_delta_mi(metrics_path)
        conditions[label] = {**counts, "full_dist_delta_mi": delta_mi}

    # Control from metrics_summary only
    control_metrics = data_dir / "control" / "logs" / "metrics_summary.parquet"
    control_counts = load_control_counts(control_metrics)
    control_delta_mi = compute_full_dist_delta_mi(control_metrics)
    conditions["control"] = {**control_counts, "full_dist_delta_mi": control_delta_mi}

    summary = {"conditions": conditions}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def _format_table(summary: dict) -> str:
    """Format summary as a readable table string."""
    lines = []
    lines.append(f"{'Condition':<12} {'Survived':>10} {'Halt':>10} {'StateUnif':>10} {'Total':>10}")
    lines.append("-" * 56)
    for label, data in summary["conditions"].items():
        n_surv = data.get("n_survived", 0)
        n_halt = data.get("n_halt", data.get("n_not_recorded", 0))
        n_su = data.get("n_state_uniform", 0)
        n_total = data.get("n_total", 0)
        halt_label = "N/R" if "n_not_recorded" in data else str(n_halt)
        su_label = "N/R" if "n_not_recorded" in data else str(n_su)
        lines.append(f"{label:<12} {n_surv:>10} {halt_label:>10} {su_label:>10} {n_total:>10}")
    lines.append("")
    lines.append("Full-distribution ΔMI:")
    for label, data in summary["conditions"].items():
        dmi = data["full_dist_delta_mi"]
        lines.append(
            f"  {label}: median={dmi['median']:.6f}, frac(>0)={dmi['fraction_positive']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    summary = build_summary(DATA_DIR)
    print(_format_table(summary))


if __name__ == "__main__":
    main()
