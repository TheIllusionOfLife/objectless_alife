"""Tests for viz/cli.py: argument parsing and subcommand dispatch."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from objectless_alife.viz.cli import _parse_phase_dirs, main


def test_parse_phase_dirs_valid() -> None:
    result = _parse_phase_dirs(["P1=/data/phase_1", "P2=/data/phase_2"])
    assert result == [("P1", Path("/data/phase_1")), ("P2", Path("/data/phase_2"))]


def test_parse_phase_dirs_invalid_format() -> None:
    with pytest.raises(ValueError, match="Expected label=path format"):
        _parse_phase_dirs(["no-equals-sign"])


def test_parse_phase_dirs_empty() -> None:
    assert _parse_phase_dirs([]) == []


def test_main_no_subcommand_exits() -> None:
    with patch.object(sys, "argv", ["visualize"]):
        with pytest.raises(SystemExit):
            main()


def test_main_single_subcommand_dispatches(tmp_path: Path) -> None:
    sim_log = tmp_path / "simulation_log.parquet"
    metrics = tmp_path / "metrics_summary.parquet"
    rule_json = tmp_path / "rule.json"
    output = tmp_path / "out.gif"
    sim_log.touch()
    metrics.touch()
    rule_json.touch()

    argv = [
        "visualize",
        "single",
        "--simulation-log",
        str(sim_log),
        "--metrics-summary",
        str(metrics),
        "--rule-json",
        str(rule_json),
        "--output",
        str(output),
        "--fps",
        "4",
        "--base-dir",
        str(tmp_path),
    ]
    with patch.object(sys, "argv", argv):
        with patch("objectless_alife.viz.cli.render_rule_animation") as mock_render:
            main()
            mock_render.assert_called_once()
            assert mock_render.call_args.kwargs["fps"] == 4


def test_main_filmstrip_subcommand_dispatches(tmp_path: Path) -> None:
    sim_log = tmp_path / "simulation_log.parquet"
    rule_json = tmp_path / "rule.json"
    output = tmp_path / "strip.png"
    sim_log.touch()
    rule_json.touch()

    argv = [
        "visualize",
        "filmstrip",
        "--simulation-log",
        str(sim_log),
        "--rule-json",
        str(rule_json),
        "--output",
        str(output),
        "--n-frames",
        "8",
        "--base-dir",
        str(tmp_path),
    ]
    with patch.object(sys, "argv", argv):
        with patch("objectless_alife.viz.cli.render_filmstrip") as mock_render:
            main()
            mock_render.assert_called_once()
            assert mock_render.call_args.kwargs["n_frames"] == 8


def test_main_unknown_subcommand_exits() -> None:
    with patch.object(sys, "argv", ["visualize", "does-not-exist"]):
        with pytest.raises(SystemExit):
            main()
