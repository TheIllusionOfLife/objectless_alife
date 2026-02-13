import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.rules import ObservationPhase
from src.run_search import run_batch_search
from src.visualize import (
    _build_grid_array,
    _state_cmap,
    render_batch,
    render_filmstrip,
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
        phase=ObservationPhase.PHASE2_PROFILE,
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


# ---------------------------------------------------------------------------
# _build_grid_array tests
# ---------------------------------------------------------------------------


def test_build_grid_array_populates_correct_cells() -> None:
    rows = [
        {"x": 0, "y": 0, "state": 1},
        {"x": 2, "y": 1, "state": 3},
    ]
    grid = _build_grid_array(rows, grid_width=3, grid_height=2)
    assert grid.shape == (2, 3)
    assert grid[0, 0] == 1
    assert grid[1, 2] == 3
    # Empty cells should be 4 (sentinel)
    assert grid[0, 1] == 4
    assert grid[0, 2] == 4
    assert grid[1, 0] == 4
    assert grid[1, 1] == 4


def test_build_grid_array_empty_rows_returns_all_empty() -> None:
    grid = _build_grid_array([], grid_width=5, grid_height=5)
    assert grid.shape == (5, 5)
    assert np.all(grid == 4)


# ---------------------------------------------------------------------------
# _state_cmap tests
# ---------------------------------------------------------------------------


def test_state_cmap_returns_five_colors() -> None:
    cmap, norm = _state_cmap(dark=False)
    assert cmap.N == 5  # 4 states + 1 empty


def test_state_cmap_dark_mode_differs() -> None:
    cmap_light, _ = _state_cmap(dark=False)
    cmap_dark, _ = _state_cmap(dark=True)
    # Empty cell color (index 4) should differ between light and dark
    assert cmap_light(4) != cmap_dark(4)


# ---------------------------------------------------------------------------
# render_snapshot_grid uses imshow (cell-fill)
# ---------------------------------------------------------------------------


def test_render_snapshot_grid_uses_imshow(tmp_path: Path) -> None:
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
    rule_id = json.loads(rule_json.read_text())["rule_id"]

    import matplotlib.pyplot as _plt

    import src.visualize as visualize

    captured_axes: list[object] = []
    original_subplots = _plt.subplots

    def _spy_subplots(*args: object, **kwargs: object) -> tuple[object, object]:
        fig, axes = original_subplots(*args, **kwargs)
        captured_axes.append(axes)
        return fig, axes

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(visualize.plt, "subplots", _spy_subplots)
    try:
        output_path = tmp_path / "snapshot.png"
        render_snapshot_grid(
            phase_configs=[
                (
                    "P1",
                    phase_dir / "logs" / "simulation_log.parquet",
                    phase_dir / "logs" / "metrics_summary.parquet",
                    rule_id,
                ),
            ],
            snapshot_steps=[0],
            output_path=output_path,
        )
        # Check that axes contain images (imshow), not just scatter
        axes = captured_axes[0]
        ax = axes[0, 0]
        assert len(ax.images) > 0, "Expected imshow (cell-fill), got no images on axis"
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# render_metric_distribution: phase colors + stats_path
# ---------------------------------------------------------------------------


def test_render_metric_distribution_with_stats_path(tmp_path: Path) -> None:
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
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=p2_dir,
        steps=4,
        base_rule_seed=30,
        base_sim_seed=40,
    )

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "metric_tests": {
                    "neighbor_mutual_information": {
                        "p_value_corrected": 1e-178,
                    }
                }
            }
        )
    )

    output_path = tmp_path / "distribution.pdf"
    render_metric_distribution(
        phase_data=[
            ("P1", p1_dir / "logs" / "metrics_summary.parquet"),
            ("P2", p2_dir / "logs" / "metrics_summary.parquet"),
        ],
        metric_names=["neighbor_mutual_information"],
        output_path=output_path,
        stats_path=stats_path,
    )
    assert output_path.exists()


def test_render_metric_distribution_uses_boxplot(tmp_path: Path) -> None:
    """After overhaul, distribution should use box plots, not violins."""
    p1_dir = tmp_path / "phase1"
    run_batch_search(
        n_rules=3,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=p1_dir,
        steps=4,
        base_rule_seed=10,
        base_sim_seed=20,
    )

    import matplotlib.pyplot as _plt

    import src.visualize as visualize

    captured_axes: list[object] = []
    original_subplots = _plt.subplots

    def _spy_subplots(*args: object, **kwargs: object) -> tuple[object, object]:
        fig, axes = original_subplots(*args, **kwargs)
        captured_axes.append(axes)
        return fig, axes

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(visualize.plt, "subplots", _spy_subplots)
    try:
        output_path = tmp_path / "distribution.pdf"
        render_metric_distribution(
            phase_data=[
                ("P1", p1_dir / "logs" / "metrics_summary.parquet"),
            ],
            metric_names=["state_entropy"],
            output_path=output_path,
        )
        # After overhaul: scatter strip (collections) should exist
        axes = captured_axes[0]
        ax = axes[0, 0]
        # Box plots create Line2D objects — check that axis has lines
        assert len(ax.lines) > 0, "Expected box plot lines on axis"
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# render_metric_timeseries: shared y-axis
# ---------------------------------------------------------------------------


def test_render_metric_timeseries_shared_ylim(tmp_path: Path) -> None:
    """When shared_ylim=True, all panels should have the same y-axis range."""
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
    rule_ids = [json.loads(p.read_text())["rule_id"] for p in rule_jsons]

    # Create a second "phase" with different data to produce different y ranges
    phase_dir2 = tmp_path / "phase2"
    run_batch_search(
        n_rules=2,
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=phase_dir2,
        steps=4,
        base_rule_seed=30,
        base_sim_seed=40,
    )
    rule_jsons2 = list((phase_dir2 / "rules").glob("*.json"))
    rule_ids2 = [json.loads(p.read_text())["rule_id"] for p in rule_jsons2]

    import matplotlib.pyplot as _plt

    import src.visualize as visualize

    captured_fig: list[object] = []
    original_subplots = _plt.subplots

    def _spy_subplots(*args: object, **kwargs: object) -> tuple[object, object]:
        fig, axes = original_subplots(*args, **kwargs)
        captured_fig.append(fig)
        return fig, axes

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(visualize.plt, "subplots", _spy_subplots)
    try:
        output_path = tmp_path / "timeseries.pdf"
        render_metric_timeseries(
            phase_configs=[
                ("P1", phase_dir / "logs" / "metrics_summary.parquet", rule_ids),
                ("P2", phase_dir2 / "logs" / "metrics_summary.parquet", rule_ids2),
            ],
            metric_name="state_entropy",
            output_path=output_path,
            shared_ylim=True,
        )
        fig = captured_fig[0]
        all_axes = fig.get_axes()
        ylims = [ax.get_ylim() for ax in all_axes]
        # All panels should share the same y range
        for ylim in ylims[1:]:
            assert ylim == ylims[0], f"Y-axis not shared: {ylim} != {ylims[0]}"
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# render_rule_animation uses imshow (cell-fill)
# ---------------------------------------------------------------------------


def test_render_rule_animation_uses_imshow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After overhaul, animation world panel should use imshow, not scatter."""
    rule_id = "phase1_rs1_ss1"
    rule_json = tmp_path / "rules" / f"{rule_id}.json"
    rule_json.parent.mkdir(parents=True, exist_ok=True)
    rule_json.write_text(
        '{"rule_id":"phase1_rs1_ss1","metadata":{"grid_width":20,"grid_height":20}}'
    )

    sim_rows = [
        {"rule_id": rule_id, "step": 0, "agent_id": 0, "x": 1, "y": 2, "state": 0, "action": 8},
        {"rule_id": rule_id, "step": 1, "agent_id": 0, "x": 2, "y": 2, "state": 1, "action": 8},
    ]
    metric_rows = [
        {"rule_id": rule_id, "step": 0, "state_entropy": 0.5},
        {"rule_id": rule_id, "step": 1, "state_entropy": 0.7},
    ]
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sim_rows), logs_dir / "simulation_log.parquet")
    pq.write_table(pa.Table.from_pylist(metric_rows), logs_dir / "metrics_summary.parquet")

    import src.visualize as visualize

    captured: dict[str, object] = {}

    class _CapturingAnimation:
        def __init__(self, fig: object, update: object, frames: int, interval: int, blit: bool) -> None:
            self.fig = fig
            # Call update(0) to trigger the first frame so imshow is populated
            update(0)
            captured["fig"] = fig
            captured["ax_world"] = fig.get_axes()[0]

        def save(self, output_path: Path, writer: object) -> None:
            Path(output_path).write_bytes(b"GIF89a")

    monkeypatch.setattr(visualize.animation, "FuncAnimation", _CapturingAnimation)
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
    assert len(ax_world.images) > 0, "Expected imshow (cell-fill), got no images on world axis"
    # xlim should still be (-0.5, 19.5) for backward compat
    assert ax_world.get_xlim() == (-0.5, 19.5)


# ---------------------------------------------------------------------------
# render_filmstrip tests
# ---------------------------------------------------------------------------


def test_render_filmstrip_creates_png(tmp_path: Path) -> None:
    rule_id = "phase1_rs1_ss1"
    rule_json = tmp_path / "rules" / f"{rule_id}.json"
    rule_json.parent.mkdir(parents=True, exist_ok=True)
    rule_json.write_text(json.dumps({"rule_id": rule_id, "metadata": {"grid_width": 5, "grid_height": 5}}))

    sim_rows = [
        {"rule_id": rule_id, "step": s, "agent_id": a, "x": a % 5, "y": a // 5, "state": (s + a) % 4, "action": 8}
        for s in range(10)
        for a in range(25)
    ]
    metric_rows = [
        {"rule_id": rule_id, "step": s, "state_entropy": 0.5 + s * 0.01}
        for s in range(10)
    ]
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sim_rows), logs_dir / "simulation_log.parquet")
    pq.write_table(pa.Table.from_pylist(metric_rows), logs_dir / "metrics_summary.parquet")

    output_path = tmp_path / "filmstrip.png"
    render_filmstrip(
        simulation_log_path=logs_dir / "simulation_log.parquet",
        metrics_summary_path=logs_dir / "metrics_summary.parquet",
        rule_json_path=rule_json,
        output_path=output_path,
        n_frames=6,
        grid_width=5,
        grid_height=5,
    )
    assert output_path.exists()


def test_render_filmstrip_clamps_n_frames(tmp_path: Path) -> None:
    """n_frames > available steps should be clamped, not error."""
    rule_id = "phase1_rs1_ss1"
    rule_json = tmp_path / "rules" / f"{rule_id}.json"
    rule_json.parent.mkdir(parents=True, exist_ok=True)
    rule_json.write_text(json.dumps({"rule_id": rule_id, "metadata": {"grid_width": 3, "grid_height": 3}}))

    sim_rows = [
        {"rule_id": rule_id, "step": s, "agent_id": a, "x": a % 3, "y": a // 3, "state": 0, "action": 8}
        for s in range(3)
        for a in range(9)
    ]
    metric_rows = [
        {"rule_id": rule_id, "step": s, "state_entropy": 0.5}
        for s in range(3)
    ]
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sim_rows), logs_dir / "simulation_log.parquet")
    pq.write_table(pa.Table.from_pylist(metric_rows), logs_dir / "metrics_summary.parquet")

    output_path = tmp_path / "filmstrip.png"
    render_filmstrip(
        simulation_log_path=logs_dir / "simulation_log.parquet",
        metrics_summary_path=logs_dir / "metrics_summary.parquet",
        rule_json_path=rule_json,
        output_path=output_path,
        n_frames=100,  # Way more than available steps (3)
        grid_width=3,
        grid_height=3,
    )
    assert output_path.exists()


def test_render_filmstrip_rejects_paths_outside_base_dir(tmp_path: Path) -> None:
    rule_json = tmp_path / "rules" / "test.json"
    rule_json.parent.mkdir(parents=True, exist_ok=True)
    rule_json.write_text(json.dumps({"rule_id": "test", "metadata": {}}))

    outside_output = tmp_path.parent / "outside.png"
    with pytest.raises(ValueError, match="Path escapes base_dir"):
        render_filmstrip(
            simulation_log_path=tmp_path / "logs" / "sim.parquet",
            metrics_summary_path=tmp_path / "logs" / "metrics.parquet",
            rule_json_path=rule_json,
            output_path=outside_output,
            base_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# filmstrip CLI subcommand test
# ---------------------------------------------------------------------------


def test_filmstrip_cli_subcommand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rule_id = "phase1_rs1_ss1"
    rule_json = tmp_path / "rules" / f"{rule_id}.json"
    rule_json.parent.mkdir(parents=True, exist_ok=True)
    rule_json.write_text(json.dumps({"rule_id": rule_id, "metadata": {"grid_width": 3, "grid_height": 3}}))

    sim_rows = [
        {"rule_id": rule_id, "step": s, "agent_id": a, "x": a % 3, "y": a // 3, "state": 0, "action": 8}
        for s in range(5)
        for a in range(9)
    ]
    metric_rows = [
        {"rule_id": rule_id, "step": s, "state_entropy": 0.5}
        for s in range(5)
    ]
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sim_rows), logs_dir / "simulation_log.parquet")
    pq.write_table(pa.Table.from_pylist(metric_rows), logs_dir / "metrics_summary.parquet")

    output_path = tmp_path / "filmstrip_cli.png"
    import src.visualize as visualize

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize",
            "filmstrip",
            "--simulation-log", str(logs_dir / "simulation_log.parquet"),
            "--metrics-summary", str(logs_dir / "metrics_summary.parquet"),
            "--rule-json", str(rule_json),
            "--output", str(output_path),
            "--n-frames", "4",
            "--base-dir", str(tmp_path),
        ],
    )
    visualize.main()
    assert output_path.exists()
