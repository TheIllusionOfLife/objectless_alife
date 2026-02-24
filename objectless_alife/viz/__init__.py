"""Visualization layer: themes, renderers, CLI, and web export."""

from objectless_alife.viz.cli import main
from objectless_alife.viz.export_web import (
    export_batch,
    export_gallery,
    export_paired,
    export_single,
)
from objectless_alife.viz.render import (
    render_batch,
    render_filmstrip,
    render_metric_distribution,
    render_metric_timeseries,
    render_rule_animation,
    render_snapshot_grid,
    select_top_rules,
)
from objectless_alife.viz.theme import (
    DEFAULT_THEME,
    PAPER_THEME,
    REGISTERED_THEMES,
    Theme,
    get_theme,
)

__all__ = [
    "DEFAULT_THEME",
    "PAPER_THEME",
    "REGISTERED_THEMES",
    "Theme",
    "export_batch",
    "export_gallery",
    "export_paired",
    "export_single",
    "get_theme",
    "main",
    "render_batch",
    "render_filmstrip",
    "render_metric_distribution",
    "render_metric_timeseries",
    "render_rule_animation",
    "render_snapshot_grid",
    "select_top_rules",
]
