"""Backward-compatibility shim: stats module moved to analysis/stats.py."""

from objectless_alife.analysis.stats import (
    _cliffs_delta_from_u as _cliffs_delta_from_u,
)
from objectless_alife.analysis.stats import (
    _holm_bonferroni as _holm_bonferroni,
)
from objectless_alife.analysis.stats import (
    _survival_counts_by_phase as _survival_counts_by_phase,
)
from objectless_alife.analysis.stats import (
    bootstrap_median_ci as bootstrap_median_ci,
)
from objectless_alife.analysis.stats import (
    filter_metric_independence as filter_metric_independence,
)
from objectless_alife.analysis.stats import (
    load_final_step_metrics as load_final_step_metrics,
)
from objectless_alife.analysis.stats import (
    main as main,
)
from objectless_alife.analysis.stats import (
    pairwise_metric_comparison as pairwise_metric_comparison,
)
from objectless_alife.analysis.stats import (
    pairwise_survival_comparison as pairwise_survival_comparison,
)
from objectless_alife.analysis.stats import (
    pairwise_survival_tests as pairwise_survival_tests,
)
from objectless_alife.analysis.stats import (
    phase_comparison_tests as phase_comparison_tests,
)
from objectless_alife.analysis.stats import (
    run_pairwise_analysis as run_pairwise_analysis,
)
from objectless_alife.analysis.stats import (
    run_statistical_analysis as run_statistical_analysis,
)
from objectless_alife.analysis.stats import (
    save_results as save_results,
)
from objectless_alife.analysis.stats import (
    survival_rate_test as survival_rate_test,
)
from objectless_alife.analysis.stats import (
    wilson_score_ci as wilson_score_ci,
)
