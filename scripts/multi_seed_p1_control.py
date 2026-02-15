"""Multi-seed robustness analysis for Phase 1 and Control conditions.

Extends the existing Phase 2 multi-seed analysis (supplementary Section B)
to Phase 1 and Control, completing the evidence that MI levels are rule
properties rather than seed-specific accidents.

Usage:
    uv run python scripts/multi_seed_p1_control.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rules import ObservationPhase  # noqa: E402
from src.run_search import (  # noqa: E402
    MultiSeedConfig,
    run_multi_seed_robustness,
    select_top_rules_by_excess_mi,
)

DATA_DIR = PROJECT_ROOT / "data" / "stage_d"

PHASE_DIR_LABELS: dict[ObservationPhase, str] = {
    ObservationPhase.PHASE1_DENSITY: "phase_1",
    ObservationPhase.CONTROL_DENSITY_CLOCK: "control",
}


def run_multi_seed_for_phase(
    phase: ObservationPhase,
    data_dir: Path,
    out_dir: Path,
    top_k: int = 50,
    n_sim_seeds: int = 20,
) -> Path:
    """Select top rules by MI_excess and run multi-seed robustness for a phase.

    Returns the path to the output parquet file.
    """
    phase_label = PHASE_DIR_LABELS.get(phase)
    if phase_label is None:
        raise ValueError(f"Unsupported phase for multi-seed analysis: {phase}")

    metrics_path = data_dir / phase_label / "logs" / "metrics_summary.parquet"
    rules_dir = data_dir / phase_label / "rules"

    top_seeds = select_top_rules_by_excess_mi(metrics_path, rules_dir, top_k=top_k)
    if not top_seeds:
        raise ValueError(f"No surviving rules found for {phase_label}")

    config = MultiSeedConfig(
        rule_seeds=tuple(top_seeds),
        n_sim_seeds=n_sim_seeds,
        out_dir=out_dir,
        phase=phase,
    )
    return run_multi_seed_robustness(config)


def summarize_multi_seed_results(results_path: Path) -> dict:
    """Summarize multi-seed results from parquet.

    Returns dict with:
      - total_rules: number of distinct rule seeds
      - rules_with_positive_median: rules whose median MI_excess > 0
      - fraction_with_positive_median: ratio of above
      - mean_positive_fraction: mean P(MI_excess > 0) across seeds per rule
      - overall_survival_rate: fraction of (rule, seed) pairs that survived
    """
    table = pq.read_table(results_path)
    rows = table.to_pylist()

    rule_seeds = sorted(set(r["rule_seed"] for r in rows))
    rules_with_positive_median = 0
    mi_excess_positive_fracs: list[float] = []
    overall_survived = 0
    overall_total = 0

    for rs in rule_seeds:
        rule_rows = [r for r in rows if r["rule_seed"] == rs]
        survived_rows = [r for r in rule_rows if r["survived"]]
        overall_survived += len(survived_rows)
        overall_total += len(rule_rows)

        mi_excess_vals = [r["mi_excess"] for r in survived_rows]
        if mi_excess_vals:
            median_mi = statistics.median(mi_excess_vals)
            if median_mi > 0:
                rules_with_positive_median += 1
            positive_frac = sum(1 for v in mi_excess_vals if v > 0) / len(mi_excess_vals)
            mi_excess_positive_fracs.append(positive_frac)
        else:
            mi_excess_positive_fracs.append(0.0)

    n_rules = len(rule_seeds)
    return {
        "total_rules": n_rules,
        "rules_with_positive_median": rules_with_positive_median,
        "fraction_with_positive_median": (rules_with_positive_median / n_rules if n_rules else 0.0),
        "mean_positive_fraction": (
            statistics.mean(mi_excess_positive_fracs) if mi_excess_positive_fracs else 0.0
        ),
        "overall_survival_rate": (overall_survived / overall_total if overall_total > 0 else 0.0),
    }


def main() -> None:
    print("Multi-Seed Robustness — Phase 1 + Control")
    print("=" * 60)

    for phase, label in [
        (ObservationPhase.PHASE1_DENSITY, "Phase 1"),
        (ObservationPhase.CONTROL_DENSITY_CLOCK, "Control"),
    ]:
        print(f"\n--- {label} ---")
        out_dir = DATA_DIR / f"multi_seed_{label.lower().replace(' ', '_')}"
        result_path = run_multi_seed_for_phase(phase, DATA_DIR, out_dir)
        summary = summarize_multi_seed_results(result_path)
        print(
            f"  Rules with positive median MI_excess: "
            f"{summary['rules_with_positive_median']}/{summary['total_rules']} "
            f"({summary['fraction_with_positive_median']:.1%})"
        )
        print(f"  Mean P(MI_excess > 0) across seeds: {summary['mean_positive_fraction']:.3f}")
        print(f"  Overall survival rate: {summary['overall_survival_rate']:.1%}")


if __name__ == "__main__":
    main()
