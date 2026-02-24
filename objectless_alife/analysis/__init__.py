"""Analysis layer: statistical significance testing and metric comparisons."""

from objectless_alife.analysis.stats import (
    bootstrap_median_ci,
    filter_metric_independence,
    load_final_step_metrics,
    pairwise_metric_comparison,
    pairwise_survival_comparison,
    pairwise_survival_tests,
    phase_comparison_tests,
    run_pairwise_analysis,
    run_statistical_analysis,
    save_results,
    survival_rate_test,
    wilson_score_ci,
)

__all__ = [
    "bootstrap_median_ci",
    "filter_metric_independence",
    "load_final_step_metrics",
    "phase_comparison_tests",
    "pairwise_metric_comparison",
    "pairwise_survival_comparison",
    "pairwise_survival_tests",
    "run_pairwise_analysis",
    "run_statistical_analysis",
    "save_results",
    "survival_rate_test",
    "wilson_score_ci",
]
