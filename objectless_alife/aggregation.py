"""Backward-compatibility shim: aggregation module moved to experiments/ subpackage."""

from objectless_alife.experiments.density_sweep import (
    run_density_sweep as run_density_sweep,
)
from objectless_alife.experiments.experiment import (
    run_experiment as run_experiment,
)
from objectless_alife.experiments.robustness import (
    run_halt_window_sweep as run_halt_window_sweep,
)
from objectless_alife.experiments.robustness import (
    run_multi_seed_robustness as run_multi_seed_robustness,
)
from objectless_alife.experiments.selection import (
    select_top_rules_by_delta_mi as select_top_rules_by_delta_mi,
)
from objectless_alife.experiments.summaries import (
    _mean as _mean,
)
from objectless_alife.experiments.summaries import (
    _percentile_pre_sorted as _percentile_pre_sorted,
)
from objectless_alife.experiments.summaries import (
    _to_float_list as _to_float_list,
)
from objectless_alife.experiments.summaries import (
    collect_final_metric_rows as collect_final_metric_rows,
)

# Private alias for backward compatibility with scripts using the old name
_collect_final_metric_rows = collect_final_metric_rows
