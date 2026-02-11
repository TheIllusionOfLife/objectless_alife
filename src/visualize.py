from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib import animation


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


def render_rule_animation(
    simulation_log_path: Path,
    metrics_summary_path: Path,
    rule_json_path: Path,
    output_path: Path,
    fps: int = 8,
    base_dir: Path | None = None,
    grid_width: int | None = None,
    grid_height: int | None = None,
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

    fig, (ax_world, ax_metric) = plt.subplots(1, 2, figsize=(10, 5))
    scatter = ax_world.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=3, s=80)
    ax_world.set_xlim(-0.5, resolved_width - 0.5)
    ax_world.set_ylim(-0.5, resolved_height - 0.5)
    ax_world.set_title("Agent States")
    ax_world.set_aspect("equal")
    ax_world.invert_yaxis()

    x_max = max(steps)
    ax_metric.set_xlim(0, 1 if x_max == 0 else x_max)
    ax_metric.set_title("State Entropy")
    ax_metric.set_xlabel("Step")
    ax_metric.set_ylabel("Entropy")
    entropy_values = [float(metric_by_step[step]["state_entropy"]) for step in steps]
    max_entropy = max(entropy_values) if entropy_values else 1.0
    ax_metric.set_ylim(0, max(1.0, max_entropy * 1.1))
    (entropy_line,) = ax_metric.plot([], [], color="tab:blue")

    def update(frame_index: int) -> tuple[object, ...]:
        step = steps[frame_index]
        rows = by_step[step]
        xs = [float(row["x"]) for row in rows]
        ys = [float(row["y"]) for row in rows]
        states = [float(row["state"]) for row in rows]
        scatter.set_offsets(list(zip(xs, ys, strict=True)))
        scatter.set_array(states)
        ax_world.set_title(f"Agent States (step={step})")

        entropy_line.set_data(steps[: frame_index + 1], entropy_values[: frame_index + 1])
        return (scatter, entropy_line)

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


def main() -> None:
    """CLI entrypoint for rendering a single rule animation."""
    parser = argparse.ArgumentParser(description="Render simulation animation for one rule")
    parser.add_argument("--simulation-log", type=Path, required=True)
    parser.add_argument("--metrics-summary", type=Path, required=True)
    parser.add_argument("--rule-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--grid-width", type=int, default=None)
    parser.add_argument("--grid-height", type=int, default=None)
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
