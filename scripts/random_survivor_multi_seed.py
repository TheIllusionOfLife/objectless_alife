"""Multi-seed robustness on random survivors (not just top-50).

For Phase 1 and Phase 2: draw B=5 bootstrap resamples of 200 random
surviving rule_seeds each, run multi-seed robustness with n_sim_seeds=10,
and aggregate results with bootstrap CIs.

Control is skipped because experiment_runs.parquet has no control entries
(no rule_seed available).

Usage:
    uv run python scripts/random_survivor_multi_seed.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from objectless_alife.rules import ObservationPhase  # noqa: E402
from objectless_alife.run_search import (  # noqa: E402
    MultiSeedConfig,
    run_multi_seed_robustness,
)
from scripts.multi_seed_p1_control import summarize_multi_seed_results  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "stage_d"

CONDITION_MAP = {
    "phase_1": (ObservationPhase.PHASE1_DENSITY, 1),
    "phase_2": (ObservationPhase.PHASE2_PROFILE, 2),
}


def get_survivor_rule_seeds(phase_num: int) -> list[int]:
    """Return surviving rule_seeds for a given phase from experiment_runs."""
    exp_path = DATA_DIR / "logs" / "experiment_runs.parquet"
    table = pq.read_table(exp_path, columns=["phase", "survived", "rule_seed"])
    rows = table.to_pylist()
    return sorted(
        set(
            r["rule_seed"]
            for r in rows
            if r["phase"] == phase_num and r["survived"] and r["rule_seed"] is not None
        )
    )


def compute_bootstrap_ci(values: list[float], ci: float = 0.95) -> dict[str, float]:
    """Compute mean and percentile bootstrap CI from B resample values."""
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    mean_val = statistics.mean(values)
    alpha = 1 - ci
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    lower_idx = max(0, int(n * alpha / 2))
    upper_idx = min(n - 1, int(n * (1 - alpha / 2)))
    return {
        "mean": round(mean_val, 4),
        "ci_lower": round(sorted_vals[lower_idx], 4),
        "ci_upper": round(sorted_vals[upper_idx], 4),
    }


def main() -> None:
    print("Multi-Seed Robustness — Random Survivors (B=5 resamples)")
    print("=" * 60)

    n_sample = 200
    n_resamples = 5
    n_sim_seeds = 10

    for condition, (phase, phase_num) in CONDITION_MAP.items():
        print(f"\n--- {condition} ---")
        all_seeds = get_survivor_rule_seeds(phase_num)
        n_total = len(all_seeds)
        print(f"  Total surviving rule_seeds: {n_total}")

        if n_total < n_sample:
            print(f"  WARNING: fewer survivors ({n_total}) than sample size ({n_sample})")

        per_resample: list[dict] = []
        frac_positive_medians: list[float] = []
        p_positive_means: list[float] = []

        for b in range(n_resamples):
            print(f"  Resample b={b}...")
            rng = random.Random(b)
            sample_size = min(n_sample, n_total)
            sampled_seeds = rng.sample(all_seeds, sample_size)

            out_dir = DATA_DIR / "multi_seed_random" / condition / f"resample_{b}"
            config = MultiSeedConfig(
                rule_seeds=tuple(sampled_seeds),
                n_sim_seeds=n_sim_seeds,
                out_dir=out_dir,
                phase=phase,
            )
            result_path = run_multi_seed_robustness(config)
            summary = summarize_multi_seed_results(result_path)

            fpm = summary["fraction_with_positive_median"]
            ppm = summary["mean_positive_fraction"]
            frac_positive_medians.append(fpm)
            p_positive_means.append(ppm)

            per_resample.append(
                {
                    "b": b,
                    "fraction_positive_median": round(fpm, 4),
                    "p_positive_mean": round(ppm, 4),
                    "overall_survival_rate": round(summary["overall_survival_rate"], 4),
                }
            )

            print(
                f"    fraction_positive_median={fpm:.3f}, "
                f"p_positive_mean={ppm:.3f}, "
                f"survival_rate={summary['overall_survival_rate']:.1%}"
            )

        fpm_ci = compute_bootstrap_ci(frac_positive_medians)
        ppm_ci = compute_bootstrap_ci(p_positive_means)

        output = {
            "condition": condition,
            "total_survivors": n_total,
            "n_sample_per_resample": min(n_sample, n_total),
            "n_resamples": n_resamples,
            "n_sim_seeds_per_rule": n_sim_seeds,
            "fraction_positive_median": fpm_ci,
            "p_positive_mean": ppm_ci,
            "per_resample": per_resample,
        }

        summary_dir = DATA_DIR / "multi_seed_random" / condition
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Summary saved to {summary_path}")

        print(f"\n  Aggregate (B={n_resamples}):")
        print(
            f"    fraction_positive_median: "
            f"{fpm_ci['mean']:.3f} [{fpm_ci['ci_lower']:.3f}, {fpm_ci['ci_upper']:.3f}]"
        )
        print(
            f"    p_positive_mean: "
            f"{ppm_ci['mean']:.3f} [{ppm_ci['ci_lower']:.3f}, {ppm_ci['ci_upper']:.3f}]"
        )


if __name__ == "__main__":
    main()
