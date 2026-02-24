"""Compatibility shim for visualization APIs and CLI.

Rendering functionality now lives in ``objectless_alife.viz.render`` and
CLI parsing/dispatch in ``objectless_alife.viz.cli``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt  # noqa: F401 — re-exported for tests that patch via this module
from matplotlib import animation  # noqa: F401 — re-exported for tests that patch via this module

from objectless_alife.viz.cli import main as main
from objectless_alife.viz.render import (
    METRIC_LABELS as METRIC_LABELS,
)
from objectless_alife.viz.render import (
    PHASE_DESCRIPTIONS as PHASE_DESCRIPTIONS,
)
from objectless_alife.viz.render import (
    _build_grid_array as _build_grid_array,
)
from objectless_alife.viz.render import (
    _state_cmap as _state_cmap,
)
from objectless_alife.viz.render import (
    render_batch as render_batch,
)
from objectless_alife.viz.render import (
    render_filmstrip as render_filmstrip,
)
from objectless_alife.viz.render import (
    render_metric_distribution as render_metric_distribution,
)
from objectless_alife.viz.render import (
    render_metric_timeseries as render_metric_timeseries,
)
from objectless_alife.viz.render import (
    render_rule_animation as render_rule_animation,
)
from objectless_alife.viz.render import (
    render_snapshot_grid as render_snapshot_grid,
)
from objectless_alife.viz.render import (
    select_top_rules as select_top_rules,
)

if __name__ == "__main__":
    main()
