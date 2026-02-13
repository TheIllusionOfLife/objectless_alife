from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib import animation
from matplotlib.gridspec import GridSpec

from src.stats import load_final_step_metrics

METRIC_LABELS: dict[str, str] = {
    "state_entropy": "State Entropy",
    "neighbor_mutual_information": "Neighbor MI",
    "compression_ratio": "Compression Ratio",
    "morans_i": "Moran's I",
    "cluster_count": "Cluster Count",
    "predictability_hamming": "Hamming Distance",
    "quasi_periodicity_peaks": "Periodicity Peaks",
    "phase_transition_max_delta": "Phase Transition",
    "action_entropy_mean": "Action Entropy (mean)",
    "action_entropy_variance": "Action Entropy (var)",
    "block_ncd": "Block NCD",
}

METRIC_COLORS: dict[str, str] = {
    "state_entropy": "tab:blue",
    "neighbor_mutual_information": "tab:red",
    "compression_ratio": "tab:green",
    "morans_i": "tab:orange",
    "cluster_count": "tab:purple",
    "predictability_hamming": "tab:brown",
    "quasi_periodicity_peaks": "tab:pink",
    "phase_transition_max_delta": "tab:gray",
    "action_entropy_mean": "tab:olive",
    "action_entropy_variance": "tab:cyan",
    "block_ncd": "darkblue",
}

PHASE_COLORS: dict[str, str] = {
    "P1": "tab:blue",
    "P2": "tab:red",
    "Control": "tab:gray",
}

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _resolve_within_base(path: Path, base_dir: Path) -> Path:
    """Resolve path and ensure it stays within the trusted base directory."""
    candidate = path if path.is_absolute() else base_dir / path
    resolved = candidate.resolve()
    base_resolved = base_dir.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise ValueError(f"Path escapes base_dir: {path}")
    return resolved


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

    return max(int(row[axis_key]) for row in rows) + 1


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

    scatter = ax_world.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=3, s=80)
    ax_world.set_xlim(-0.5, resolved_width - 0.5)
    ax_world.set_ylim(-0.5, resolved_height - 0.5)
    ax_world.set_title("Agent States")
    ax_world.set_aspect("equal")
    ax_world.invert_yaxis()

    x_max = max(steps)
    metric_values_list: list[list[float]] = []
    metric_lines: list[object] = []

    for idx, m_name in enumerate(effective_metrics):
        ax_m = metric_axes[idx]
        values = [float(metric_by_step[step].get(m_name, 0.0) or 0.0) for step in steps]
        metric_values_list.append(values)
        max_val = max(values) if values else 1.0
        ax_m.set_xlim(0, 1 if x_max == 0 else x_max)
        ax_m.set_ylim(0, max(1.0, max_val * 1.1))
        label = METRIC_LABELS.get(m_name, m_name)
        color = METRIC_COLORS.get(m_name, "tab:blue")
        ax_m.set_title(label)
        ax_m.set_xlabel("Step")
        ax_m.set_ylabel(label)
        (line,) = ax_m.plot([], [], color=color)
        metric_lines.append(line)

    fig.tight_layout()

    def update(frame_index: int) -> tuple[object, ...]:
        step = steps[frame_index]
        rows = by_step[step]
        xs = [float(row["x"]) for row in rows]
        ys = [float(row["y"]) for row in rows]
        states = [float(row["state"]) for row in rows]
        scatter.set_offsets(list(zip(xs, ys, strict=True)))
        scatter.set_array(states)
        ax_world.set_title(f"Agent States (step={step})")

        for line, values in zip(metric_lines, metric_values_list, strict=True):
            line.set_data(steps[: frame_index + 1], values[: frame_index + 1])
        return (scatter, *metric_lines)

    anim = animation.FuncAnimation(
        fig, update, frames=len(steps), interval=max(1, int(1000 / fps)), blit=False
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
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
) -> None:
    """Render (n_phases x n_steps) grid of agent position scatter plots."""
    n_phases = len(phase_configs)
    n_steps = len(snapshot_steps)

    fig, axes = plt.subplots(n_phases, n_steps, figsize=(3 * n_steps, 3 * n_phases), squeeze=False)

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
            xs = [float(r["x"]) for r in rows]
            ys = [float(r["y"]) for r in rows]
            states = [float(r["state"]) for r in rows]
            ax.scatter(xs, ys, c=states, cmap="viridis", vmin=0, vmax=3, s=20)
            ax.set_xlim(-0.5, grid_width - 0.5)
            ax.set_ylim(-0.5, grid_height - 0.5)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"Step {target_step}", fontsize=10)
            if col_idx == 0:
                mi_str = f" (MI={mi_val:.3f})" if mi_val is not None else ""
                ax.set_ylabel(f"{label}{mi_str}", fontsize=10)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# render_metric_distribution
# ---------------------------------------------------------------------------


def render_metric_distribution(
    phase_data: list[tuple[str, Path]],
    metric_names: list[str],
    output_path: Path,
) -> None:
    """Violin plots comparing metric distributions across phases."""
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), squeeze=False)

    for m_idx, m_name in enumerate(metric_names):
        ax = axes[0, m_idx]
        all_data: list[list[float]] = []
        labels: list[str] = []

        for label, metrics_path in phase_data:
            final_table = load_final_step_metrics(metrics_path)
            vals = []
            for row in final_table.to_pylist():
                v = row.get(m_name)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals.append(float(v))
            all_data.append(vals)
            labels.append(label)

        # Only create violin if we have data
        non_empty = [d for d in all_data if d]
        if non_empty:
            ax.violinplot(non_empty, showmedians=True)
            ax.set_xticks(range(1, len(non_empty) + 1))
            # Map xticks to labels of non-empty datasets
            non_empty_labels = [lb for lb, d in zip(labels, all_data, strict=True) if d]
            ax.set_xticklabels(non_empty_labels)

        ax.set_title(METRIC_LABELS.get(m_name, m_name))
        ax.set_ylabel(METRIC_LABELS.get(m_name, m_name))

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
) -> None:
    """Time-series overlay of metric trajectories per phase."""
    n_phases = len(phase_configs)
    fig, axes = plt.subplots(1, n_phases, figsize=(5 * n_phases, 4), squeeze=False)

    for p_idx, (label, metrics_path, rule_ids) in enumerate(phase_configs):
        ax = axes[0, p_idx]
        color = PHASE_COLORS.get(label, "tab:blue")

        for rule_id in rule_ids:
            metric_rows = pq.read_table(
                metrics_path, filters=[("rule_id", "=", rule_id)]
            ).to_pylist()
            steps = sorted(int(r["step"]) for r in metric_rows)
            vals = [
                float(r.get(metric_name, 0.0) or 0.0)
                for r in sorted(metric_rows, key=lambda r: int(r["step"]))
            ]
            ax.plot(steps, vals, color=color, alpha=0.6, linewidth=1)

        ax.set_title(f"{label}")
        ax.set_xlabel("Step")
        ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))

    fig.suptitle(METRIC_LABELS.get(metric_name, metric_name), fontsize=14)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI with subcommands
# ---------------------------------------------------------------------------


def _build_single_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("single", help="Render animation for a single rule")
    p.add_argument("--simulation-log", type=Path, required=True)
    p.add_argument("--metrics-summary", type=Path, required=True)
    p.add_argument("--rule-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--grid-width", type=int, default=None)
    p.add_argument("--grid-height", type=int, default=None)


def _build_batch_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("batch", help="Batch-render top-N rules per phase")
    p.add_argument(
        "--phase-dir",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Phase directory as label=path (can repeat)",
    )
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--base-dir", type=Path, default=Path("."))


def _build_figure_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("figure", help="Generate all static paper figures")
    p.add_argument("--p1-dir", type=Path, required=True)
    p.add_argument("--p2-dir", type=Path, required=True)
    p.add_argument("--control-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--base-dir", type=Path, default=Path("."))


def _parse_phase_dirs(raw: list[str]) -> list[tuple[str, Path]]:
    result = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Expected label=path format, got: {item}")
        label, path_str = item.split("=", 1)
        result.append((label, Path(path_str)))
    return result


def main() -> None:
    """CLI entrypoint with subcommands."""
    parser = argparse.ArgumentParser(description="Visualization tools for simulation data")
    sub = parser.add_subparsers(dest="command", required=True)
    _build_single_parser(sub)
    _build_batch_parser(sub)
    _build_figure_parser(sub)
    args = parser.parse_args()

    if args.command == "single":
        render_rule_animation(
            simulation_log_path=args.simulation_log,
            metrics_summary_path=args.metrics_summary,
            rule_json_path=args.rule_json,
            output_path=args.output,
            fps=args.fps,
            base_dir=args.base_dir,
            grid_width=args.grid_width,
            grid_height=args.grid_height,
        )
    elif args.command == "batch":
        base_dir = Path(args.base_dir).resolve()
        phase_dirs = _parse_phase_dirs(args.phase_dir)
        # Validate all phase dirs and output dir stay within base_dir
        for _label, pdir in phase_dirs:
            _resolve_within_base(pdir, base_dir)
        _resolve_within_base(args.output_dir, base_dir)
        render_batch(
            phase_dirs=phase_dirs,
            output_dir=args.output_dir,
            top_n=args.top_n,
            fps=args.fps,
        )
    elif args.command == "figure":
        base_dir = Path(args.base_dir).resolve()
        # Validate all dirs stay within base_dir
        for pdir in [args.p1_dir, args.p2_dir, args.control_dir, args.output_dir]:
            _resolve_within_base(pdir, base_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        phases = [
            ("P1", args.p1_dir),
            ("P2", args.p2_dir),
            ("Control", args.control_dir),
        ]

        # Select top rules per phase
        top_rules: dict[str, list[str]] = {}
        for label, pdir in phases:
            metrics_path = pdir / "logs" / "metrics_summary.parquet"
            top_rules[label] = select_top_rules(metrics_path, top_n=args.top_n)

        # Fig 1: Snapshot grid (top-1 rule per phase)
        snapshot_configs = []
        for label, pdir in phases:
            if top_rules[label]:
                snapshot_configs.append(
                    (
                        label,
                        pdir / "logs" / "simulation_log.parquet",
                        pdir / "logs" / "metrics_summary.parquet",
                        top_rules[label][0],
                    )
                )
        render_snapshot_grid(
            phase_configs=snapshot_configs,
            snapshot_steps=[0, 25, 50, 75, 100],
            output_path=output_dir / "fig1_snapshot_grid.pdf",
        )

        # Fig 2: MI distribution
        render_metric_distribution(
            phase_data=[
                (label, pdir / "logs" / "metrics_summary.parquet") for label, pdir in phases
            ],
            metric_names=["neighbor_mutual_information"],
            output_path=output_dir / "fig2_mi_distribution.pdf",
        )

        # Fig 3: MI time-series (top-3 rules per phase)
        render_metric_timeseries(
            phase_configs=[
                (label, pdir / "logs" / "metrics_summary.parquet", top_rules[label])
                for label, pdir in phases
            ],
            metric_name="neighbor_mutual_information",
            output_path=output_dir / "fig3_mi_timeseries.pdf",
        )


if __name__ == "__main__":
    main()
