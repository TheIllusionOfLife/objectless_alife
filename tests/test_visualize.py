import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.rules import ObservationPhase
from src.run_search import run_batch_search
from src.visualize import render_rule_animation


class _DummyAnimation:
    def __init__(self, fig: object, update: object, frames: int, interval: int, blit: bool) -> None:
        self.fig = fig
        self.update = update
        self.frames = frames
        self.interval = interval
        self.blit = blit

    def save(self, output_path: Path, writer: object) -> None:
        Path(output_path).write_bytes(b"GIF89a")


def test_render_rule_animation_creates_gif(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=4,
        base_rule_seed=2,
        base_sim_seed=3,
    )
    rule_json = next((tmp_path / "rules").glob("*.json"))
    output_path = tmp_path / "preview.gif"
    render_rule_animation(
        simulation_log_path=tmp_path / "logs" / "simulation_log.parquet",
        metrics_summary_path=tmp_path / "logs" / "metrics_summary.parquet",
        rule_json_path=rule_json,
        output_path=output_path,
        fps=2,
    )
    assert output_path.exists()


def test_render_rule_animation_rejects_paths_outside_base_dir(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=3,
        base_rule_seed=1,
        base_sim_seed=1,
    )
    rule_json = next((tmp_path / "rules").glob("*.json"))
    outside_output = tmp_path.parent / "outside.gif"
    with pytest.raises(ValueError):
        render_rule_animation(
            simulation_log_path=tmp_path / "logs" / "simulation_log.parquet",
            metrics_summary_path=tmp_path / "logs" / "metrics_summary.parquet",
            rule_json_path=rule_json,
            output_path=outside_output,
            fps=2,
            base_dir=tmp_path,
        )


def test_render_rule_animation_rejects_missing_rule_id(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=3,
        base_rule_seed=1,
        base_sim_seed=1,
    )
    rule_json = next((tmp_path / "rules").glob("*.json"))
    rule_json.write_text("{}")
    with pytest.raises(ValueError, match="rule_id"):
        render_rule_animation(
            simulation_log_path=tmp_path / "logs" / "simulation_log.parquet",
            metrics_summary_path=tmp_path / "logs" / "metrics_summary.parquet",
            rule_json_path=rule_json,
            output_path=tmp_path / "preview.gif",
            fps=2,
        )


def test_render_rule_animation_rejects_missing_metric_steps(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=4,
        base_rule_seed=2,
        base_sim_seed=3,
    )
    metric_path = tmp_path / "logs" / "metrics_summary.parquet"
    table = pq.read_table(metric_path)
    rows = table.to_pylist()
    max_step = max(int(row["step"]) for row in rows)
    filtered = [row for row in rows if int(row["step"]) != max_step]
    pq.write_table(pa.Table.from_pylist(filtered), metric_path)

    rule_json = next((tmp_path / "rules").glob("*.json"))
    with pytest.raises(ValueError, match="Missing metrics for steps"):
        render_rule_animation(
            simulation_log_path=tmp_path / "logs" / "simulation_log.parquet",
            metrics_summary_path=metric_path,
            rule_json_path=rule_json,
            output_path=tmp_path / "preview.gif",
            fps=2,
        )


def test_render_rule_animation_uses_explicit_grid_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rule_id = "phase1_rs1_ss1"
    rule_json = tmp_path / "rules" / f"{rule_id}.json"
    rule_json.parent.mkdir(parents=True, exist_ok=True)
    rule_json.write_text(
        '{"rule_id":"phase1_rs1_ss1","metadata":{"grid_width":20,"grid_height":20}}'
    )

    sim_rows = [
        {"rule_id": rule_id, "step": 0, "agent_id": 0, "x": 1, "y": 2, "state": 0, "action": 8},
        {"rule_id": rule_id, "step": 1, "agent_id": 0, "x": 2, "y": 2, "state": 0, "action": 8},
    ]
    metric_rows = [
        {"rule_id": rule_id, "step": 0, "state_entropy": 0.0},
        {"rule_id": rule_id, "step": 1, "state_entropy": 0.0},
    ]
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sim_rows), logs_dir / "simulation_log.parquet")
    pq.write_table(pa.Table.from_pylist(metric_rows), logs_dir / "metrics_summary.parquet")

    import src.visualize as visualize

    captured: dict[str, object] = {}
    original_subplots = visualize.plt.subplots

    def _spy_subplots(*args: object, **kwargs: object) -> tuple[object, object]:
        fig, axes = original_subplots(*args, **kwargs)
        captured["ax_world"] = axes[0]
        return fig, axes

    monkeypatch.setattr(visualize.plt, "subplots", _spy_subplots)
    monkeypatch.setattr(visualize.animation, "FuncAnimation", _DummyAnimation)
    monkeypatch.setattr(visualize.animation, "PillowWriter", lambda fps: object())

    render_rule_animation(
        simulation_log_path=logs_dir / "simulation_log.parquet",
        metrics_summary_path=logs_dir / "metrics_summary.parquet",
        rule_json_path=rule_json,
        output_path=tmp_path / "preview.gif",
        fps=2,
        grid_width=20,
        grid_height=20,
    )

    ax_world = captured["ax_world"]
    assert ax_world.get_xlim() == (-0.5, 19.5)


def test_visualize_cli_default_base_dir_rejects_absolute_paths_outside_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=4,
        base_rule_seed=2,
        base_sim_seed=3,
    )
    rule_json = next((tmp_path / "rules").glob("*.json")).resolve()
    output_path = (tmp_path / "cli_preview.gif").resolve()

    import src.visualize as visualize

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize",
            "--simulation-log",
            str((tmp_path / "logs" / "simulation_log.parquet").resolve()),
            "--metrics-summary",
            str((tmp_path / "logs" / "metrics_summary.parquet").resolve()),
            "--rule-json",
            str(rule_json),
            "--output",
            str(output_path),
            "--fps",
            "2",
        ],
    )
    with pytest.raises(ValueError, match="Path escapes base_dir"):
        visualize.main()


def test_visualize_cli_accepts_absolute_paths_with_explicit_base_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=4,
        base_rule_seed=2,
        base_sim_seed=3,
    )
    rule_json = next((tmp_path / "rules").glob("*.json")).resolve()
    output_path = (tmp_path / "cli_preview.gif").resolve()

    import src.visualize as visualize

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize",
            "--simulation-log",
            str((tmp_path / "logs" / "simulation_log.parquet").resolve()),
            "--metrics-summary",
            str((tmp_path / "logs" / "metrics_summary.parquet").resolve()),
            "--rule-json",
            str(rule_json),
            "--output",
            str(output_path),
            "--fps",
            "2",
            "--base-dir",
            str(tmp_path.resolve()),
        ],
    )
    visualize.main()
    assert output_path.exists()
