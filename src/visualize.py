from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib import animation


def render_rule_animation(
    simulation_log_path: Path,
    metrics_summary_path: Path,
    rule_json_path: Path,
    output_path: Path,
    fps: int = 8,
) -> None:
    """Render one rule's trajectory and metric trend as an animation."""
    simulation_log_path = Path(simulation_log_path)
    metrics_summary_path = Path(metrics_summary_path)
    rule_json_path = Path(rule_json_path)
    output_path = Path(output_path)

    rule_payload = json.loads(rule_json_path.read_text())
    rule_id = rule_payload["rule_id"]

    sim_rows = [
        row for row in pq.read_table(simulation_log_path).to_pylist() if row["rule_id"] == rule_id
    ]
    metric_rows = [
        row for row in pq.read_table(metrics_summary_path).to_pylist() if row["rule_id"] == rule_id
    ]
    if not sim_rows:
        raise ValueError(f"No simulation rows found for rule_id={rule_id}")
    if not metric_rows:
        raise ValueError(f"No metric rows found for rule_id={rule_id}")

    steps = sorted({int(row["step"]) for row in sim_rows})
    by_step: dict[int, list[dict[str, object]]] = {step: [] for step in steps}
    for row in sim_rows:
        by_step[int(row["step"])].append(row)

    metric_by_step = {int(row["step"]): row for row in metric_rows}
    width = max(int(row["x"]) for row in sim_rows) + 1
    height = max(int(row["y"]) for row in sim_rows) + 1

    fig, (ax_world, ax_metric) = plt.subplots(1, 2, figsize=(10, 5))
    scatter = ax_world.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=3, s=80)
    ax_world.set_xlim(-0.5, width - 0.5)
    ax_world.set_ylim(-0.5, height - 0.5)
    ax_world.set_title("Agent States")
    ax_world.set_aspect("equal")
    ax_world.invert_yaxis()

    ax_metric.set_xlim(0, max(steps))
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
