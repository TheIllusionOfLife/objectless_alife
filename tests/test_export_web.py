"""Tests for the web JSON export module (src/export_web.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.export_web import (
    export_batch,
    export_gallery,
    export_paired,
    export_single,
)
from src.rules import ObservationPhase
from src.run_search import run_batch_search


@pytest.fixture()
def phase2_dir(tmp_path: Path) -> Path:
    """Run a small Phase 2 batch and return the output directory."""
    out = tmp_path / "phase_2"
    run_batch_search(
        n_rules=5,
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=out,
        steps=20,
        halt_window=5,
        base_rule_seed=0,
        base_sim_seed=0,
    )
    return out


@pytest.fixture()
def control_dir(tmp_path: Path) -> Path:
    """Run a small Control batch and return the output directory."""
    out = tmp_path / "control"
    run_batch_search(
        n_rules=5,
        phase=ObservationPhase.CONTROL_DENSITY_CLOCK,
        out_dir=out,
        steps=20,
        halt_window=5,
        base_rule_seed=0,
        base_sim_seed=0,
    )
    return out


# --------------------------------------------------------------------------
# export_single
# --------------------------------------------------------------------------


class TestExportSingle:
    def test_produces_valid_json(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "single.json"
        export_single(data_dir=phase2_dir, rule_id="phase2_rs0_ss0", output=out_file)

        data = json.loads(out_file.read_text())
        assert "meta" in data
        assert "frames" in data
        assert "metrics" in data

    def test_meta_fields(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "single.json"
        export_single(data_dir=phase2_dir, rule_id="phase2_rs0_ss0", output=out_file)

        meta = json.loads(out_file.read_text())["meta"]
        assert meta["rule_id"] == "phase2_rs0_ss0"
        assert meta["grid_width"] == 20
        assert meta["grid_height"] == 20
        assert meta["num_agents"] == 30
        assert meta["steps"] == 20

    def test_frame_count_matches_steps(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "single.json"
        export_single(data_dir=phase2_dir, rule_id="phase2_rs0_ss0", output=out_file)

        data = json.loads(out_file.read_text())
        # Frame count should equal the number of simulation steps recorded
        assert len(data["frames"]) > 0
        # Each frame should have step and agents
        frame = data["frames"][0]
        assert "step" in frame
        assert "agents" in frame

    def test_agents_format(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "single.json"
        export_single(data_dir=phase2_dir, rule_id="phase2_rs0_ss0", output=out_file)

        data = json.loads(out_file.read_text())
        frame = data["frames"][0]
        agents = frame["agents"]
        # Each agent is [x, y, state]
        assert len(agents) == 30
        for agent in agents:
            assert len(agent) == 3
            x, y, state = agent
            assert 0 <= x < 20
            assert 0 <= y < 20
            assert 0 <= state <= 3

    def test_metrics_series(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "single.json"
        export_single(data_dir=phase2_dir, rule_id="phase2_rs0_ss0", output=out_file)

        data = json.loads(out_file.read_text())
        mi = data["metrics"]["neighbor_mutual_information"]
        assert isinstance(mi, list)
        assert len(mi) == len(data["frames"])

    def test_nonexistent_rule_raises(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "bad.json"
        with pytest.raises(ValueError, match="No simulation"):
            export_single(data_dir=phase2_dir, rule_id="nonexistent_rule", output=out_file)

    def test_frames_sorted_by_step(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "single.json"
        export_single(data_dir=phase2_dir, rule_id="phase2_rs0_ss0", output=out_file)

        data = json.loads(out_file.read_text())
        steps = [f["step"] for f in data["frames"]]
        assert steps == sorted(steps)


# --------------------------------------------------------------------------
# export_paired
# --------------------------------------------------------------------------


class TestExportPaired:
    def test_produces_valid_paired_json(
        self, phase2_dir: Path, control_dir: Path, tmp_path: Path
    ) -> None:
        out_file = tmp_path / "paired.json"
        export_paired(
            phase2_dir=phase2_dir,
            control_dir=control_dir,
            sim_seed=0,
            output=out_file,
        )

        data = json.loads(out_file.read_text())
        assert "left" in data
        assert "right" in data
        assert "meta" in data

    def test_paired_meta(self, phase2_dir: Path, control_dir: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "paired.json"
        export_paired(
            phase2_dir=phase2_dir,
            control_dir=control_dir,
            sim_seed=0,
            output=out_file,
        )

        meta = json.loads(out_file.read_text())["meta"]
        assert meta["left_phase"] == "phase_2"
        assert meta["right_phase"] == "control"
        assert meta["sim_seed"] == 0

    def test_both_sides_have_frames(
        self, phase2_dir: Path, control_dir: Path, tmp_path: Path
    ) -> None:
        out_file = tmp_path / "paired.json"
        export_paired(
            phase2_dir=phase2_dir,
            control_dir=control_dir,
            sim_seed=0,
            output=out_file,
        )

        data = json.loads(out_file.read_text())
        assert len(data["left"]["frames"]) > 0
        assert len(data["right"]["frames"]) > 0

    def test_both_sides_have_metrics(
        self, phase2_dir: Path, control_dir: Path, tmp_path: Path
    ) -> None:
        out_file = tmp_path / "paired.json"
        export_paired(
            phase2_dir=phase2_dir,
            control_dir=control_dir,
            sim_seed=0,
            output=out_file,
        )

        data = json.loads(out_file.read_text())
        assert "neighbor_mutual_information" in data["left"]["metrics"]
        assert "neighbor_mutual_information" in data["right"]["metrics"]

    def test_mismatched_sim_seed_raises(
        self, phase2_dir: Path, control_dir: Path, tmp_path: Path
    ) -> None:
        out_file = tmp_path / "bad.json"
        with pytest.raises(ValueError):
            export_paired(
                phase2_dir=phase2_dir,
                control_dir=control_dir,
                sim_seed=9999,
                output=out_file,
            )


# --------------------------------------------------------------------------
# export_batch
# --------------------------------------------------------------------------


class TestExportBatch:
    def test_exports_top_n_files(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "batch_out"
        export_batch(data_dir=phase2_dir, top_n=3, output_dir=out_dir)

        json_files = list(out_dir.glob("*.json"))
        # Should export up to top_n (or fewer if fewer rules survived)
        assert len(json_files) <= 3
        assert len(json_files) > 0

    def test_each_file_valid_schema(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "batch_out"
        export_batch(data_dir=phase2_dir, top_n=2, output_dir=out_dir)

        for f in out_dir.glob("*.json"):
            data = json.loads(f.read_text())
            assert "meta" in data
            assert "frames" in data
            assert "metrics" in data
            assert len(data["frames"]) > 0

    def test_top_n_exceeding_available(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "batch_out"
        # Request more than available; should not error
        export_batch(data_dir=phase2_dir, top_n=100, output_dir=out_dir)

        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) <= 5  # only 5 rules were generated


# --------------------------------------------------------------------------
# export_gallery
# --------------------------------------------------------------------------


class TestExportGallery:
    def test_exports_diverse_set(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "gallery_out"
        export_gallery(data_dir=phase2_dir, count=3, output_dir=out_dir)

        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) <= 3
        assert len(json_files) > 0

    def test_gallery_files_valid(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "gallery_out"
        export_gallery(data_dir=phase2_dir, count=2, output_dir=out_dir)

        for f in out_dir.glob("*.json"):
            data = json.loads(f.read_text())
            assert "meta" in data
            assert "frames" in data
            assert len(data["frames"]) > 0


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------


class TestDegenerate:
    def test_batch_zero_top_n(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "batch_zero"
        export_batch(data_dir=phase2_dir, top_n=0, output_dir=out_dir)

        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) == 0

    def test_gallery_zero_count(self, phase2_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "gallery_zero"
        export_gallery(data_dir=phase2_dir, count=0, output_dir=out_dir)

        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) == 0

    def test_batch_empty_data_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out_dir = tmp_path / "batch_empty"
        export_batch(data_dir=empty_dir, top_n=5, output_dir=out_dir)

        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) == 0

    def test_gallery_empty_data_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out_dir = tmp_path / "gallery_empty"
        export_gallery(data_dir=empty_dir, count=5, output_dir=out_dir)

        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) == 0
