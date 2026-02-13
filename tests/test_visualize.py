import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.rules import ObservationPhase
from src.run_search import run_batch_search
from src.visualize import (
    render_batch,
    render_metric_distribution,
    render_metric_timeseries,
    render_rule_animation,
    render_snapshot_grid,
    select_top_rules,
)


class _DummyAnimation:
    def __init__(self, fig: object, update: object, frames: int, interval: int, blit: bool) -> None:
        self.fig = fig
        self.update = update
        self.frames = frames
        self.interval = interval
        self.blit = blit

    def save(self, output_path: Path, writer: object) -> None:
        Path(output_path).write_bytes(b"GIF89a")


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI tests (updated for subcommand syntax)
# ---------------------------------------------------------------------------


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
            "single",
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
            "single",
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


# ---------------------------------------------------------------------------
# Stage D: select_top_rules tests
# ---------------------------------------------------------------------------


def test_select_top_rules_returns_correct_order(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=5,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=4,
        base_rule_seed=10,
        base_sim_seed=20,
    )
    metrics_path = tmp_path / "logs" / "metrics_summary.parquet"
    result = select_top_rules(metrics_path, metric_name="neighbor_mutual_information", top_n=3)
    assert len(result) == 3

    # Verify descending order: load final-step metrics and check
    from src.stats import load_final_step_metrics

    final_table = load_final_step_metrics(metrics_path)
    mi_by_rule: dict[str, float] = {}
    for row in final_table.to_pylist():
        val = row["neighbor_mutual_information"]
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            mi_by_rule[row["rule_id"]] = float(val)

    for i in range(len(result) - 1):
        assert mi_by_rule[result[i]] >= mi_by_rule[result[i + 1]]


def test_select_top_rules_skips_nan_values(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=3,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=4,
        base_rule_seed=10,
        base_sim_seed=20,
    )
    metrics_path = tmp_path / "logs" / "metrics_summary.parquet"

    # Inject NaN for the first rule
    table = pq.read_table(metrics_path)
    rows = table.to_pylist()
    first_rule_id = rows[0]["rule_id"]
    for row in rows:
        if row["rule_id"] == first_rule_id:
            row["neighbor_mutual_information"] = float("nan")
    pq.write_table(pa.Table.from_pylist(rows), metrics_path)

    result = select_top_rules(metrics_path, metric_name="neighbor_mutual_information", top_n=3)
    assert first_rule_id not in result


# ---------------------------------------------------------------------------
# Stage D: multi-metric animation tests
# ---------------------------------------------------------------------------


def test_render_rule_animation_with_multiple_metrics(
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
    rule_json = next((tmp_path / "rules").glob("*.json"))

    import src.visualize as visualize

    captured_fig: list[object] = []
    original_func_anim = _DummyAnimation

    class _CaptureDummyAnimation(_DummyAnimation):
        def __init__(self, fig: object, *args: object, **kwargs: object) -> None:
            super().__init__(fig, *args, **kwargs)
            captured_fig.append(fig)

    monkeypatch.setattr(visualize.animation, "FuncAnimation", _CaptureDummyAnimation)
    monkeypatch.setattr(visualize.animation, "PillowWriter", lambda fps: object())

    render_rule_animation(
        simulation_log_path=tmp_path / "logs" / "simulation_log.parquet",
        metrics_summary_path=tmp_path / "logs" / "metrics_summary.parquet",
        rule_json_path=rule_json,
        output_path=tmp_path / "preview.gif",
        fps=2,
        metric_names=["neighbor_mutual_information", "state_entropy"],
    )

    fig = captured_fig[0]
    # 1 scatter + 2 metric panels = 3 axes
    assert len(fig.get_axes()) == 3


def test_render_rule_animation_default_metrics_backward_compatible(
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
    rule_json = next((tmp_path / "rules").glob("*.json"))

    import src.visualize as visualize

    captured_fig: list[object] = []

    class _CaptureDummyAnimation(_DummyAnimation):
        def __init__(self, fig: object, *args: object, **kwargs: object) -> None:
            super().__init__(fig, *args, **kwargs)
            captured_fig.append(fig)

    monkeypatch.setattr(visualize.animation, "FuncAnimation", _CaptureDummyAnimation)
    monkeypatch.setattr(visualize.animation, "PillowWriter", lambda fps: object())

    render_rule_animation(
        simulation_log_path=tmp_path / "logs" / "simulation_log.parquet",
        metrics_summary_path=tmp_path / "logs" / "metrics_summary.parquet",
        rule_json_path=rule_json,
        output_path=tmp_path / "preview.gif",
        fps=2,
    )

    fig = captured_fig[0]
    # Default: 1 scatter + 1 metric = 2 axes (backward compatible)
    assert len(fig.get_axes()) == 2


# ---------------------------------------------------------------------------
# Stage D: render_batch tests
# ---------------------------------------------------------------------------


def test_render_batch_creates_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase_dir = tmp_path / "phase1"
    run_batch_search(
        n_rules=3,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=phase_dir,
        steps=4,
        base_rule_seed=10,
        base_sim_seed=20,
    )

    import src.visualize as visualize

    monkeypatch.setattr(visualize.animation, "FuncAnimation", _DummyAnimation)
    monkeypatch.setattr(visualize.animation, "PillowWriter", lambda fps: object())

    output_dir = tmp_path / "output"
    result = render_batch(
        phase_dirs=[("p1", phase_dir)],
        output_dir=output_dir,
        top_n=2,
    )

    assert len(result) == 2
    for path in result:
        assert path.exists()
        assert path.suffix == ".gif"
        assert path.name.startswith("p1_top")


# ---------------------------------------------------------------------------
# Stage D: static figure tests
# ---------------------------------------------------------------------------


def test_render_snapshot_grid_creates_pdf(tmp_path: Path) -> None:
    phase_dir = tmp_path / "phase1"
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=phase_dir,
        steps=4,
        base_rule_seed=2,
        base_sim_seed=3,
    )
    rule_json = next((phase_dir / "rules").glob("*.json"))
    import json

    rule_id = json.loads(rule_json.read_text())["rule_id"]

    output_path = tmp_path / "snapshot.pdf"
    render_snapshot_grid(
        phase_configs=[
            (
                "P1",
                phase_dir / "logs" / "simulation_log.parquet",
                phase_dir / "logs" / "metrics_summary.parquet",
                rule_id,
            ),
        ],
        snapshot_steps=[0, 2],
        output_path=output_path,
    )
    assert output_path.exists()


def test_render_metric_distribution_creates_pdf(tmp_path: Path) -> None:
    p1_dir = tmp_path / "phase1"
    p2_dir = tmp_path / "phase2"
    run_batch_search(
        n_rules=3,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=p1_dir,
        steps=4,
        base_rule_seed=10,
        base_sim_seed=20,
    )
    run_batch_search(
        n_rules=3,
        phase=ObservationPhase.PHASE2_DENSITY_CLOCK,
        out_dir=p2_dir,
        steps=4,
        base_rule_seed=30,
        base_sim_seed=40,
    )

    output_path = tmp_path / "distribution.pdf"
    render_metric_distribution(
        phase_data=[
            ("P1", p1_dir / "logs" / "metrics_summary.parquet"),
            ("P2", p2_dir / "logs" / "metrics_summary.parquet"),
        ],
        metric_names=["state_entropy"],
        output_path=output_path,
    )
    assert output_path.exists()


def test_render_metric_timeseries_creates_pdf(tmp_path: Path) -> None:
    phase_dir = tmp_path / "phase1"
    run_batch_search(
        n_rules=2,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=phase_dir,
        steps=4,
        base_rule_seed=10,
        base_sim_seed=20,
    )
    rule_jsons = list((phase_dir / "rules").glob("*.json"))
    import json

    rule_ids = [json.loads(p.read_text())["rule_id"] for p in rule_jsons]

    output_path = tmp_path / "timeseries.pdf"
    render_metric_timeseries(
        phase_configs=[
            ("P1", phase_dir / "logs" / "metrics_summary.parquet", rule_ids),
        ],
        metric_name="state_entropy",
        output_path=output_path,
    )
    assert output_path.exists()
