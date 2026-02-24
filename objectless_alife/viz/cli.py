"""CLI entrypoint and subcommand handlers for the visualization tools."""

from __future__ import annotations

import argparse
from pathlib import Path

from objectless_alife.viz.render import (
    _resolve_within_base,
    render_batch,
    render_filmstrip,
    render_metric_distribution,
    render_metric_timeseries,
    render_rule_animation,
    render_snapshot_grid,
    select_top_rules,
)
from objectless_alife.viz.theme import get_theme


def _build_single_parser(sub: argparse._SubParsersAction) -> None:
    """Register the 'single' subcommand: animate one rule's trajectory."""
    p = sub.add_parser("single", help="Render animation for a single rule")
    p.set_defaults(func=_handle_single)
    p.add_argument("--simulation-log", type=Path, required=True)
    p.add_argument("--metrics-summary", type=Path, required=True)
    p.add_argument("--rule-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--grid-width", type=int, default=None)
    p.add_argument("--grid-height", type=int, default=None)


def _build_batch_parser(sub: argparse._SubParsersAction) -> None:
    """Register the 'batch' subcommand: render top-N rules per phase."""
    p = sub.add_parser("batch", help="Batch-render top-N rules per phase")
    p.set_defaults(func=_handle_batch)
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
    """Register the 'figure' subcommand: generate all static paper figures."""
    p = sub.add_parser("figure", help="Generate all static paper figures")
    p.set_defaults(func=_handle_figure)
    p.add_argument("--p1-dir", type=Path, required=True)
    p.add_argument("--p2-dir", type=Path, required=True)
    p.add_argument("--control-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--random-walk-dir", type=Path, default=None)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--stats-path", type=Path, default=None)


def _build_filmstrip_parser(sub: argparse._SubParsersAction) -> None:
    """Register the 'filmstrip' subcommand: render a horizontal filmstrip."""
    p = sub.add_parser("filmstrip", help="Render filmstrip of simulation frames")
    p.set_defaults(func=_handle_filmstrip)
    p.add_argument("--simulation-log", type=Path, required=True)
    p.add_argument("--rule-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=6)
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--grid-width", type=int, default=None)
    p.add_argument("--grid-height", type=int, default=None)


def _parse_phase_dirs(raw: list[str]) -> list[tuple[str, Path]]:
    result = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Expected label=path format, got: {item}")
        label, path_str = item.split("=", 1)
        result.append((label, Path(path_str)))
    return result


def _handle_single(args: argparse.Namespace) -> None:
    """Dispatch the 'single' subcommand to render_rule_animation."""
    render_rule_animation(
        simulation_log_path=args.simulation_log,
        metrics_summary_path=args.metrics_summary,
        rule_json_path=args.rule_json,
        output_path=args.output,
        fps=args.fps,
        base_dir=args.base_dir,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        theme=args.theme_obj,
    )


def _handle_batch(args: argparse.Namespace) -> None:
    """Dispatch the 'batch' subcommand to render_batch."""
    base_dir = Path(args.base_dir).resolve()
    phase_dirs = _parse_phase_dirs(args.phase_dir)
    phase_dirs = [(label, _resolve_within_base(pdir, base_dir)) for label, pdir in phase_dirs]
    output_dir = _resolve_within_base(args.output_dir, base_dir)
    render_batch(
        phase_dirs=phase_dirs,
        output_dir=output_dir,
        top_n=args.top_n,
        fps=args.fps,
        theme=args.theme_obj,
    )


def _handle_figure(args: argparse.Namespace) -> None:
    """Dispatch the 'figure' subcommand to generate paper figures."""
    base_dir = Path(args.base_dir).resolve()
    p1_dir = _resolve_within_base(args.p1_dir, base_dir)
    p2_dir = _resolve_within_base(args.p2_dir, base_dir)
    control_dir = _resolve_within_base(args.control_dir, base_dir)
    output_dir = _resolve_within_base(args.output_dir, base_dir)
    random_walk_dir = (
        _resolve_within_base(args.random_walk_dir, base_dir)
        if args.random_walk_dir is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = [("RW", random_walk_dir)] if random_walk_dir is not None else []
    phases += [
        ("Control", control_dir),
        ("P1", p1_dir),
        ("P2", p2_dir),
    ]

    top_rules: dict[str, list[str]] = {
        label: select_top_rules(pdir / "logs" / "metrics_summary.parquet", top_n=args.top_n)
        for label, pdir in phases
    }

    snapshot_configs = [
        (
            label,
            pdir / "logs" / "simulation_log.parquet",
            pdir / "logs" / "metrics_summary.parquet",
            top_rules[label][0],
        )
        for label, pdir in phases
        if top_rules[label]
    ]
    render_snapshot_grid(
        phase_configs=snapshot_configs,
        snapshot_steps=[0, 25, 50, 75, 100],
        output_path=output_dir / "fig1_snapshot_grid.pdf",
        theme=args.theme_obj,
    )

    stats_path = getattr(args, "stats_path", None)
    if stats_path is not None:
        stats_path = _resolve_within_base(stats_path, base_dir)
    render_metric_distribution(
        phase_data=[(label, pdir / "logs" / "metrics_summary.parquet") for label, pdir in phases],
        metric_names=["neighbor_mutual_information"],
        output_path=output_dir / "fig2_mi_distribution.pdf",
        stats_path=stats_path,
        theme=args.theme_obj,
    )

    render_metric_timeseries(
        phase_configs=[
            (label, pdir / "logs" / "metrics_summary.parquet", top_rules[label])
            for label, pdir in phases
        ],
        metric_name="neighbor_mutual_information",
        output_path=output_dir / "fig3_mi_timeseries.pdf",
        theme=args.theme_obj,
    )


def _handle_filmstrip(args: argparse.Namespace) -> None:
    """Dispatch the 'filmstrip' subcommand to render_filmstrip."""
    render_filmstrip(
        simulation_log_path=args.simulation_log,
        rule_json_path=args.rule_json,
        output_path=args.output,
        n_frames=args.n_frames,
        base_dir=args.base_dir,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        theme=args.theme_obj,
    )


def main() -> None:
    """CLI entrypoint with subcommands."""
    parser = argparse.ArgumentParser(description="Visualization tools for simulation data")
    parser.add_argument(
        "--theme",
        type=str,
        default="default",
        help="Theme preset name (default, paper)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _build_single_parser(sub)
    _build_batch_parser(sub)
    _build_figure_parser(sub)
    _build_filmstrip_parser(sub)
    args = parser.parse_args()

    # Apply theme
    args.theme_obj = get_theme(args.theme)

    args.func(args)


if __name__ == "__main__":
    main()
