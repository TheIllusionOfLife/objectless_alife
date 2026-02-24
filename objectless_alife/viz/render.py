"""Matplotlib-based rendering functions for simulation visualizations."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib import animation
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from matplotlib.patches import Patch

from objectless_alife.analysis.stats import load_final_step_metrics
from objectless_alife.io.paths import resolve_within_base as _resolve_within_base
from objectless_alife.viz.theme import DEFAULT_THEME, Theme

# Backward-compatible module-level aliases (read-only snapshots of default).
METRIC_LABELS: dict[str, str] = DEFAULT_THEME.metric_labels
METRIC_COLORS: dict[str, str] = DEFAULT_THEME.metric_colors
PHASE_COLORS: dict[str, str] = DEFAULT_THEME.phase_colors
PHASE_DESCRIPTIONS: dict[str, str] = DEFAULT_THEME.phase_descriptions
STATE_COLORS: list[str] = list(DEFAULT_THEME.state_colors)
EMPTY_CELL_COLOR: str = DEFAULT_THEME.empty_cell_color
EMPTY_CELL_COLOR_DARK: str = DEFAULT_THEME.empty_cell_color_dark
GRID_LINE_COLOR: str = DEFAULT_THEME.grid_line_color
GRID_LINE_COLOR_DARK: str = DEFAULT_THEME.grid_line_color_dark

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _resolve_grid_dimension(
    explicit: int | None,
    metadata: dict[str, object],
    metadata_key: str,
    rows: list[dict[str, object]],
    axis_key: str,
) -> int:
    """Resolve grid dimension from explicit arg, metadata, then row maxima."""
    if explicit is not None:
        return explicit

    from_metadata = metadata.get(metadata_key)
    if isinstance(from_metadata, int):
        return from_metadata

    if not rows:
        raise ValueError(
            f"Cannot infer {metadata_key}: rows are empty and no explicit value provided"
        )

    return max(int(row[axis_key]) for row in rows) + 1  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# Cell-fill helpers
# ---------------------------------------------------------------------------


def _build_grid_array(
    rows: list[dict[str, object]], grid_width: int, grid_height: int
) -> np.ndarray:
    """Return (H, W) int array: 0-3 for agent states, 4 for empty cells.

    Out-of-range state values are clamped to the empty-cell sentinel (4).
    Out-of-bounds positions are silently skipped.
    """
    grid = np.full((grid_height, grid_width), 4, dtype=int)
    for row in rows:
        x, y, state = int(row["x"]), int(row["y"]), int(row["state"])  # type: ignore[call-overload]
        if not (0 <= state <= 3):
            state = 4
        if 0 <= y < grid_height and 0 <= x < grid_width:
            grid[y, x] = state
    return grid


def _state_cmap(
    dark: bool = False, theme: Theme = DEFAULT_THEME
) -> tuple[ListedColormap, BoundaryNorm]:
    """Discrete 5-color colormap (4 states + empty cell)."""
    empty = theme.empty_cell_color_dark if dark else theme.empty_cell_color
    colors = list(theme.state_colors) + [empty]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    return cmap, norm


def _build_state_legend_handles(dark: bool = False, theme: Theme = DEFAULT_THEME) -> list[Patch]:
    """Build legend patch handles for the 4 agent states and empty cells."""
    empty = theme.empty_cell_color_dark if dark else theme.empty_cell_color
    handles = [
        Patch(facecolor=c, edgecolor="gray", label=f"State {i}")
        for i, c in enumerate(theme.state_colors)
    ]
    handles.append(Patch(facecolor=empty, edgecolor="gray", label="Empty"))
    return handles


def _draw_cell_grid(
    ax: plt.Axes,
    grid: np.ndarray,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    dark: bool = False,
    theme: Theme = DEFAULT_THEME,
) -> AxesImage:
    """Shared renderer: imshow with subtle grid lines on *ax*."""
    img = ax.imshow(grid, cmap=cmap, norm=norm, origin="upper", aspect="equal")
    h, w = grid.shape
    line_color = theme.grid_line_color_dark if dark else theme.grid_line_color
    for x in range(w + 1):
        ax.axvline(x - 0.5, color=line_color, linewidth=0.5)
    for y in range(h + 1):
        ax.axhline(y - 0.5, color=line_color, linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if dark:
        ax.set_facecolor(theme.empty_cell_color_dark)
    return img


def _annotate_significance(ax: plt.Axes, x1: float, x2: float, text: str, y: float) -> None:
    """Draw bracket with significance stars between two x-positions.

    Uses additive offset derived from the axis y-range so the bracket
    remains visible even when *y* is zero or negative.
    """
    ylo, yhi = ax.get_ylim()
    offset = max(0.02 * (yhi - ylo), 1e-6)
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + offset, y + offset, y],
        color="black",
        linewidth=1,
    )
    ax.text(
        (x1 + x2) / 2,
        y + 1.5 * offset,
        text,
        ha="center",
        va="bottom",
        fontsize=10,
    )


# ---------------------------------------------------------------------------
# select_top_rules
# ---------------------------------------------------------------------------


def select_top_rules(
    metrics_path: Path,
    metric_name: str = "neighbor_mutual_information",
    top_n: int = 3,
) -> list[str]:
    """Return rule_ids ranked by final-step metric value, descending."""
    final_table = load_final_step_metrics(metrics_path)
    rows = final_table.to_pylist()

    scored: list[tuple[str, float]] = []
    for row in rows:
        val = row.get(metric_name)
        if val is None:
            continue
        fval = float(val)
        if math.isnan(fval):
            continue
        scored.append((row["rule_id"], fval))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [rule_id for rule_id, _ in scored[:top_n]]


# ---------------------------------------------------------------------------
# render_rule_animation (enhanced with multi-metric support)
# ---------------------------------------------------------------------------


def render_rule_animation(
    simulation_log_path: Path,
    metrics_summary_path: Path,
    rule_json_path: Path,
    output_path: Path,
    fps: int = 8,
    base_dir: Path | None = None,
    grid_width: int | None = None,
    grid_height: int | None = None,
    metric_names: list[str] | None = None,
    theme: Theme = DEFAULT_THEME,
) -> None:
    """Render one rule's trajectory and metric trend as an animation."""
    if base_dir is None:
        simulation_log_path = Path(simulation_log_path).resolve()
        metrics_summary_path = Path(metrics_summary_path).resolve()
        rule_json_path = Path(rule_json_path).resolve()
        output_path = Path(output_path).resolve()
    else:
        base_dir = Path(base_dir).resolve()
        simulation_log_path = _resolve_within_base(Path(simulation_log_path), base_dir)
        metrics_summary_path = _resolve_within_base(Path(metrics_summary_path), base_dir)
        rule_json_path = _resolve_within_base(Path(rule_json_path), base_dir)
        output_path = _resolve_within_base(Path(output_path), base_dir)

    rule_payload = json.loads(rule_json_path.read_text())
    rule_id = rule_payload.get("rule_id")
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError("Rule JSON must include non-empty string field 'rule_id'")

    sim_rows = pq.read_table(simulation_log_path, filters=[("rule_id", "=", rule_id)]).to_pylist()
    metric_rows = pq.read_table(
        metrics_summary_path, filters=[("rule_id", "=", rule_id)]
    ).to_pylist()
    if not sim_rows:
        raise ValueError(f"No simulation rows found for rule_id={rule_id}")
    if not metric_rows:
        raise ValueError(f"No metric rows found for rule_id={rule_id}")

    steps = sorted({int(row["step"]) for row in sim_rows})
    if not steps:
        raise ValueError(f"No simulation steps found for rule_id={rule_id}")
    by_step: dict[int, list[dict[str, object]]] = {step: [] for step in steps}
    for row in sim_rows:
        by_step[int(row["step"])].append(row)

    metric_by_step = {int(row["step"]): row for row in metric_rows}
    missing_metric_steps = [step for step in steps if step not in metric_by_step]
    if missing_metric_steps:
        raise ValueError(
            f"Missing metrics for steps {missing_metric_steps[:5]} for rule_id={rule_id}"
        )
    raw_metadata = rule_payload.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    resolved_width = _resolve_grid_dimension(
        explicit=grid_width,
        metadata=metadata,
        metadata_key="grid_width",
        rows=sim_rows,
        axis_key="x",
    )
    resolved_height = _resolve_grid_dimension(
        explicit=grid_height,
        metadata=metadata,
        metadata_key="grid_height",
        rows=sim_rows,
        axis_key="y",
    )

    if resolved_width < 1:
        raise ValueError("grid_width must be >= 1")
    if resolved_height < 1:
        raise ValueError("grid_height must be >= 1")

    # Default: single metric (backward compatible)
    effective_metrics = metric_names if metric_names is not None else ["state_entropy"]
    n_metrics = len(effective_metrics)

    if n_metrics == 1:
        # Backward-compatible: 1×2 layout
        fig, (ax_world, ax_metric) = plt.subplots(1, 2, figsize=(10, 5))
        metric_axes = [ax_metric]
    else:
        # Multi-metric: gridspec with scatter on left, stacked metrics on right
        fig = plt.figure(figsize=(12, max(5, 2.5 * n_metrics)))
        gs = GridSpec(n_metrics, 2, figure=fig, width_ratios=[1, 1])
        ax_world = fig.add_subplot(gs[:, 0])
        metric_axes = [fig.add_subplot(gs[i, 1]) for i in range(n_metrics)]

    cmap, norm = _state_cmap(dark=True, theme=theme)
    init_grid = _build_grid_array(by_step[steps[0]], resolved_width, resolved_height)
    img = _draw_cell_grid(ax_world, init_grid, cmap, norm, dark=True, theme=theme)
    ax_world.set_xlim(-0.5, resolved_width - 0.5)
    ax_world.set_ylim(resolved_height - 0.5, -0.5)
    ax_world.set_title("Agent States")

    x_max = max(steps)
    metric_values_list: list[list[float]] = []
    metric_lines: list[Any] = []

    for idx, m_name in enumerate(effective_metrics):
        ax_m = metric_axes[idx]
        values = [float(metric_by_step[step].get(m_name, 0.0) or 0.0) for step in steps]
        metric_values_list.append(values)
        max_val = max(values) if values else 1.0
        ax_m.set_xlim(0, 1 if x_max == 0 else x_max)
        ax_m.set_ylim(0, max(1.0, max_val * 1.1))
        label = theme.metric_labels.get(m_name, m_name)
        color = theme.metric_colors.get(m_name, "tab:blue")
        ax_m.set_title(label)
        ax_m.set_xlabel("Step")
        ax_m.set_ylabel(label)
        (line,) = ax_m.plot([], [], color=color)
        metric_lines.append(line)

    fig.tight_layout()

    def update(frame_index: int) -> tuple[Any, ...]:
        step = steps[frame_index]
        rows = by_step[step]
        grid = _build_grid_array(rows, resolved_width, resolved_height)
        img.set_data(grid)
        ax_world.set_title(f"Agent States (step={step})")

        for line, values in zip(metric_lines, metric_values_list, strict=True):
            line.set_data(steps[: frame_index + 1], values[: frame_index + 1])
        return (img, *metric_lines)

    anim = animation.FuncAnimation(
        fig, update, frames=len(steps), interval=max(1, int(1000 / fps)), blit=False
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    writer: animation.PillowWriter | animation.FFMpegWriter
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)


# ---------------------------------------------------------------------------
# render_batch
# ---------------------------------------------------------------------------


def render_batch(
    phase_dirs: list[tuple[str, Path]],
    output_dir: Path,
    metric_name: str = "neighbor_mutual_information",
    top_n: int = 3,
    fps: int = 8,
    metric_names: list[str] | None = None,
    theme: Theme = DEFAULT_THEME,
) -> list[Path]:
    """Batch-render top-N rule animations per phase."""
    if metric_names is None:
        metric_names = ["neighbor_mutual_information", "state_entropy"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for label, phase_dir in phase_dirs:
        safe_label = Path(label).name
        if not _SAFE_NAME_RE.match(safe_label):
            raise ValueError(f"Unsafe label for filename: {label!r}")
        phase_dir = Path(phase_dir)
        metrics_path = phase_dir / "logs" / "metrics_summary.parquet"
        sim_log_path = phase_dir / "logs" / "simulation_log.parquet"
        rules_dir = phase_dir / "rules"

        top_rule_ids = select_top_rules(metrics_path, metric_name=metric_name, top_n=top_n)

        for rank, rule_id in enumerate(top_rule_ids, start=1):
            safe_rule_id = Path(rule_id).name
            if not _SAFE_NAME_RE.match(safe_rule_id):
                raise ValueError(f"Unsafe rule_id for filename: {rule_id!r}")
            rule_json_path = rules_dir / f"{safe_rule_id}.json"
            out_name = f"{safe_label}_top{rank}_{metric_name}.gif"
            out_path = output_dir / out_name
            render_rule_animation(
                simulation_log_path=sim_log_path,
                metrics_summary_path=metrics_path,
                rule_json_path=rule_json_path,
                output_path=out_path,
                fps=fps,
                metric_names=metric_names,
                theme=theme,
            )
            created.append(out_path)

    return created


# ---------------------------------------------------------------------------
# render_snapshot_grid
# ---------------------------------------------------------------------------


def render_snapshot_grid(
    phase_configs: list[tuple[str, Path, Path, str]],
    snapshot_steps: list[int],
    output_path: Path,
    grid_width: int = 20,
    grid_height: int = 20,
    theme: Theme = DEFAULT_THEME,
) -> None:
    """Render (n_phases x n_steps) grid of cell-fill agent state panels."""
    n_phases = len(phase_configs)
    n_steps = len(snapshot_steps)

    fig, axes = plt.subplots(n_phases, n_steps, figsize=(3 * n_steps, 3 * n_phases), squeeze=False)
    cmap, norm = _state_cmap(dark=False, theme=theme)

    for row_idx, (label, sim_log_path, metrics_path, rule_id) in enumerate(phase_configs):
        sim_rows = pq.read_table(sim_log_path, filters=[("rule_id", "=", rule_id)]).to_pylist()
        if not sim_rows:
            raise ValueError(f"No simulation rows for rule_id={rule_id} in {sim_log_path}")
        available_steps = sorted({int(r["step"]) for r in sim_rows})
        by_step: dict[int, list[dict[str, object]]] = {s: [] for s in available_steps}
        for r in sim_rows:
            by_step[int(r["step"])].append(r)

        # Load final MI value for row label
        final_table = load_final_step_metrics(metrics_path)
        mi_val = None
        for r in final_table.to_pylist():
            if r["rule_id"] == rule_id:
                v = r.get("neighbor_mutual_information")
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    mi_val = float(v)
                break

        for col_idx, target_step in enumerate(snapshot_steps):
            ax = axes[row_idx, col_idx]
            # Find nearest available step
            actual_step = min(available_steps, key=lambda s: abs(s - target_step))
            rows = by_step[actual_step]
            grid = _build_grid_array(rows, grid_width, grid_height)
            _draw_cell_grid(ax, grid, cmap, norm, dark=False, theme=theme)

            if row_idx == 0:
                ax.set_title(f"Step {target_step}", fontsize=10)
            if col_idx == 0:
                desc = theme.phase_descriptions.get(label, label)
                mi_str = f"\n(MI = {mi_val:.3f})" if mi_val is not None else ""
                ax.set_ylabel(f"{desc}{mi_str}", fontsize=9)

    handles = _build_state_legend_handles(dark=False, theme=theme)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# render_metric_distribution
# ---------------------------------------------------------------------------


def _aggregate_metric_data(
    phase_data: list[tuple[str, Path]],
    metric_name: str,
) -> tuple[list[list[float]], list[str]]:
    """Aggregate metric values from multiple phases for a single metric."""
    all_data: list[list[float]] = []
    labels: list[str] = []

    for label, metrics_path in phase_data:
        final_table = load_final_step_metrics(metrics_path)
        vals: list[float] = []
        if metric_name in final_table.column_names:
            col = final_table.column(metric_name).cast(pa.float64())
            vals = pc.filter(col, pc.is_finite(col)).to_pylist()
        all_data.append(vals)
        labels.append(label)

    return all_data, labels


def render_metric_distribution(
    phase_data: list[tuple[str, Path]],
    metric_names: list[str],
    output_path: Path,
    stats_path: Path | None = None,
    theme: Theme = DEFAULT_THEME,
) -> None:
    """Phase-colored box plots with scatter strip and optional significance brackets."""
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), squeeze=False)

    stats: dict[str, Any] | None = None
    if stats_path is not None:
        stats = json.loads(Path(stats_path).read_text())

    for m_idx, m_name in enumerate(metric_names):
        ax = axes[0, m_idx]
        all_data, labels = _aggregate_metric_data(phase_data, m_name)

        # Phase-colored box plots + scatter strip
        positions = list(range(1, len(all_data) + 1))
        for i, (data, label) in enumerate(zip(all_data, labels, strict=True)):
            if not data:
                continue
            pos = positions[i]
            color = theme.phase_colors.get(label, "tab:blue")
            bp = ax.boxplot(
                [data],
                positions=[pos],
                widths=0.5,
                patch_artist=True,
                medianprops={"color": "white", "linewidth": 1.5},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            # Scatter strip overlay
            rng = np.random.default_rng(seed=i)
            jitter = rng.uniform(-0.15, 0.15, size=len(data))
            ax.scatter(
                [pos + j for j in jitter],
                data,
                color=color,
                alpha=0.3,
                s=8,
                zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(theme.metric_labels.get(m_name, m_name))
        ax.set_ylabel(theme.metric_labels.get(m_name, m_name))

        # Significance annotation
        non_empty = [d for d in all_data if d]
        if stats is not None and non_empty:
            metric_stats = stats.get("metric_tests", {}).get(m_name)
            if metric_stats and "P1" in labels and "P2" in labels:
                p_val = metric_stats.get("p_value_corrected")
                if p_val is not None:
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    elif p_val < 0.05:
                        stars = "*"
                    else:
                        stars = "n.s."
                    p1_idx = labels.index("P1")
                    p2_idx = labels.index("P2")
                    y_max = max(max(d) for d in non_empty)
                    _annotate_significance(
                        ax,
                        positions[p1_idx],
                        positions[p2_idx],
                        stars,
                        y_max * 1.05,
                    )

    if stats is not None:
        fig.text(
            0.99,
            0.01,
            "* p < 0.05   ** p < 0.01   *** p < 0.001",
            ha="right",
            va="bottom",
            fontsize=7,
            color="gray",
            transform=fig.transFigure,
        )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# render_metric_timeseries
# ---------------------------------------------------------------------------


def render_metric_timeseries(
    phase_configs: list[tuple[str, Path, list[str]]],
    metric_name: str,
    output_path: Path,
    shared_ylim: bool = True,
    theme: Theme = DEFAULT_THEME,
) -> None:
    """Time-series overlay of metric trajectories per phase."""
    n_phases = len(phase_configs)
    fig, axes = plt.subplots(1, n_phases, figsize=(5 * n_phases, 4), squeeze=False)

    for p_idx, (label, metrics_path, rule_ids) in enumerate(phase_configs):
        ax = axes[0, p_idx]
        color = theme.phase_colors.get(label, "tab:blue")

        for rule_id in rule_ids:
            metric_rows = pq.read_table(
                metrics_path, filters=[("rule_id", "=", rule_id)]
            ).to_pylist()
            steps = sorted(int(r["step"]) for r in metric_rows)
            vals = [
                float(r.get(metric_name, 0.0) or 0.0)
                for r in sorted(metric_rows, key=lambda r: int(r["step"]))
            ]
            ax.plot(steps, vals, color=color, alpha=0.4, linewidth=1.8)

        ax.set_title(theme.phase_descriptions.get(label, label))
        ax.set_xlabel("Step")
        ax.set_ylabel(theme.metric_labels.get(metric_name, metric_name))
        ax.grid(True, alpha=0.3)

    if shared_ylim and n_phases > 0:
        all_ylims = [axes[0, i].get_ylim() for i in range(n_phases)]
        global_ymin = min(yl[0] for yl in all_ylims)
        global_ymax = max(yl[1] for yl in all_ylims)
        for i in range(n_phases):
            axes[0, i].set_ylim(global_ymin, global_ymax)

    fig.suptitle(theme.metric_labels.get(metric_name, metric_name), fontsize=14)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# render_filmstrip
# ---------------------------------------------------------------------------


def render_filmstrip(
    simulation_log_path: Path,
    rule_json_path: Path,
    output_path: Path,
    n_frames: int = 6,
    base_dir: Path | None = None,
    grid_width: int | None = None,
    grid_height: int | None = None,
    theme: Theme = DEFAULT_THEME,
) -> None:
    """Render horizontal filmstrip of cell-fill panels (dark mode) with step labels."""
    if base_dir is None:
        simulation_log_path = Path(simulation_log_path).resolve()
        rule_json_path = Path(rule_json_path).resolve()
        output_path = Path(output_path).resolve()
    else:
        base_dir = Path(base_dir).resolve()
        simulation_log_path = _resolve_within_base(Path(simulation_log_path), base_dir)
        rule_json_path = _resolve_within_base(Path(rule_json_path), base_dir)
        output_path = _resolve_within_base(Path(output_path), base_dir)

    rule_payload = json.loads(rule_json_path.read_text())
    rule_id = rule_payload.get("rule_id")
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError("Rule JSON must include non-empty string field 'rule_id'")

    sim_rows = pq.read_table(simulation_log_path, filters=[("rule_id", "=", rule_id)]).to_pylist()
    if not sim_rows:
        raise ValueError(f"No simulation rows found for rule_id={rule_id}")

    steps = sorted({int(row["step"]) for row in sim_rows})
    by_step: dict[int, list[dict[str, object]]] = {step: [] for step in steps}
    for row in sim_rows:
        by_step[int(row["step"])].append(row)

    raw_metadata = rule_payload.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    resolved_width = _resolve_grid_dimension(
        explicit=grid_width,
        metadata=metadata,
        metadata_key="grid_width",
        rows=sim_rows,
        axis_key="x",
    )
    resolved_height = _resolve_grid_dimension(
        explicit=grid_height,
        metadata=metadata,
        metadata_key="grid_height",
        rows=sim_rows,
        axis_key="y",
    )

    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    actual_n = max(1, min(n_frames, len(steps)))
    indices = [int(i * (len(steps) - 1) / max(1, actual_n - 1)) for i in range(actual_n)]
    selected_steps = [steps[i] for i in indices]

    cmap, norm = _state_cmap(dark=True, theme=theme)
    fig, axes = plt.subplots(1, actual_n, figsize=(3 * actual_n, 3), squeeze=False)
    fig.patch.set_facecolor(theme.empty_cell_color_dark)

    for col_idx, step in enumerate(selected_steps):
        ax = axes[0, col_idx]
        grid = _build_grid_array(by_step[step], resolved_width, resolved_height)
        _draw_cell_grid(ax, grid, cmap, norm, dark=True, theme=theme)
        ax.set_title(f"Step {step}", fontsize=9, color="white")

    fig.suptitle(f"Rule: {rule_id}", fontsize=11, color="white")
    handles = _build_state_legend_handles(dark=True, theme=theme)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        labelcolor="white",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
