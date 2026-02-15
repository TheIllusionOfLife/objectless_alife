"""Cascaded filter analysis — weak vs weak+medium filter survival.

Runs the original vision's cascade experiment: applies medium filters
(short-period + low-activity) alongside the existing weak filters to see
how survival counts and metric distributions change.

Usage:
    uv run python scripts/cascaded_filter_analysis.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rules import ObservationPhase  # noqa: E402
from src.run_search import (  # noqa: E402
    SearchConfig,
    SimulationResult,
    run_batch_search,
)

DATA_DIR = PROJECT_ROOT / "data" / "stage_d"

CONDITIONS = [
    ("Phase 1", ObservationPhase.PHASE1_DENSITY, "phase_1"),
    ("Phase 2", ObservationPhase.PHASE2_PROFILE, "phase_2"),
    ("Control", ObservationPhase.CONTROL_DENSITY_CLOCK, "control"),
]


def run_with_medium_filters(
    n_rules: int,
    phase: ObservationPhase,
    out_dir: Path,
    base_rule_seed: int = 0,
    steps: int = 200,
) -> list[SimulationResult]:
    """Run batch search with both weak and medium filters enabled.

    Medium filters: short-period detection and low-activity detection.
    """
    config = SearchConfig(
        steps=steps,
        halt_window=10,
        filter_short_period=True,
        short_period_max_period=2,
        short_period_history_size=8,
        filter_low_activity=True,
        low_activity_window=5,
        low_activity_min_unique_ratio=0.2,
    )
    return run_batch_search(
        n_rules=n_rules,
        phase=phase,
        out_dir=out_dir,
        base_rule_seed=base_rule_seed,
        base_sim_seed=base_rule_seed,
        config=config,
    )


def _load_mi_excess_for_survivors(
    metrics_path: Path,
    rules_dir: Path,
) -> list[float]:
    """Load final-step MI_excess values for surviving rules."""
    survived_ids: set[str] = set()
    for path in sorted(rules_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("survived", False):
            survived_ids.add(data["rule_id"])

    table = pq.read_table(
        metrics_path,
        columns=["rule_id", "step", "neighbor_mutual_information", "mi_shuffle_null"],
    )
    rows = table.to_pylist()

    max_steps: dict[str, int] = {}
    for row in rows:
        rid = row["rule_id"]
        if rid not in survived_ids:
            continue
        step = int(row["step"])
        if rid not in max_steps or step > max_steps[rid]:
            max_steps[rid] = step

    mi_excess_vals: list[float] = []
    for row in rows:
        rid = row["rule_id"]
        if rid not in survived_ids:
            continue
        if int(row["step"]) != max_steps[rid]:
            continue
        mi = row.get("neighbor_mutual_information")
        null = row.get("mi_shuffle_null")
        if mi is not None and null is not None:
            mi_excess_vals.append(max(float(mi) - float(null), 0.0))

    return mi_excess_vals


def main() -> None:
    print("Cascaded Filter Analysis — Weak vs Weak+Medium")
    print("=" * 60)

    n_rules = 5000
    all_results: dict[str, dict] = {}

    for label, phase, dir_name in CONDITIONS:
        print(f"\n--- {label} ---")

        # Run with medium filters (same seeds as stage_d)
        medium_dir = DATA_DIR / f"cascaded_{dir_name}"
        print(f"  Running {n_rules} rules with medium filters...")
        medium_results = run_with_medium_filters(
            n_rules=n_rules,
            phase=phase,
            out_dir=medium_dir,
        )

        # Load weak-only results from stage_d rule JSONs
        weak_dir = DATA_DIR / dir_name
        weak_rules = list(sorted((weak_dir / "rules").glob("*.json")))
        weak_survived = sum(
            1 for p in weak_rules if json.loads(p.read_text()).get("survived", False)
        )

        medium_survived = sum(1 for r in medium_results if r.survived)

        # MI_excess for medium-filter survivors
        medium_mi_excess = _load_mi_excess_for_survivors(
            medium_dir / "logs" / "metrics_summary.parquet",
            medium_dir / "rules",
        )
        median_mi = statistics.median(medium_mi_excess) if medium_mi_excess else 0.0

        result = {
            "weak_survived": weak_survived,
            "weak_total": len(weak_rules),
            "weak_survival_rate": weak_survived / len(weak_rules) if weak_rules else 0.0,
            "medium_survived": medium_survived,
            "medium_total": n_rules,
            "medium_survival_rate": medium_survived / n_rules,
            "medium_median_mi_excess": median_mi,
        }
        all_results[label] = result

        print(
            f"  Weak-only: {weak_survived}/{len(weak_rules)} survived "
            f"({result['weak_survival_rate']:.1%})"
        )
        print(
            f"  Weak+Medium: {medium_survived}/{n_rules} survived "
            f"({result['medium_survival_rate']:.1%})"
        )
        print(f"  Medium survivors median MI_excess: {median_mi:.4f}")

    # Print cascade table
    print("\n" + "=" * 60)
    print("CASCADE SURVIVAL TABLE")
    print("=" * 60)
    print(f"{'Condition':12s} | {'Weak Only':>12s} | {'Weak+Medium':>12s} | {'Delta':>8s}")
    print("-" * 52)
    for label, r in all_results.items():
        weak_pct = f"{r['weak_survival_rate']:.1%}"
        med_pct = f"{r['medium_survival_rate']:.1%}"
        delta = r["medium_survived"] - r["weak_survived"]
        print(f"{label:12s} | {weak_pct:>12s} | {med_pct:>12s} | {delta:>+8d}")

    # Save
    output_path = PROJECT_ROOT / "data" / "post_hoc" / "cascaded_filter_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
