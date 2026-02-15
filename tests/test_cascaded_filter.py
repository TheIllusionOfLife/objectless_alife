"""Tests for scripts/cascaded_filter_analysis.py."""

from pathlib import Path

import pytest

from src.rules import ObservationPhase


class TestRunWithMediumFilters:
    def test_produces_results_with_medium_filters_enabled(self, tmp_path: Path) -> None:
        from scripts.cascaded_filter_analysis import run_with_medium_filters

        results = run_with_medium_filters(
            n_rules=5,
            phase=ObservationPhase.PHASE1_DENSITY,
            out_dir=tmp_path / "medium",
            base_rule_seed=0,
            steps=10,
        )
        assert len(results) == 5

    def test_medium_filters_recorded_in_metadata(self, tmp_path: Path) -> None:
        import json

        from scripts.cascaded_filter_analysis import run_with_medium_filters

        out_dir = tmp_path / "medium"
        run_with_medium_filters(
            n_rules=2,
            phase=ObservationPhase.PHASE2_PROFILE,
            out_dir=out_dir,
            base_rule_seed=0,
            steps=10,
        )
        rules_dir = out_dir / "rules"
        json_files = list(rules_dir.glob("*.json"))
        assert len(json_files) == 2
        for f in json_files:
            payload = json.loads(f.read_text())
            assert payload["metadata"]["filter_short_period"] is True
            assert payload["metadata"]["filter_low_activity"] is True
            assert "short_period" in payload["filter_results"]
            assert "low_activity" in payload["filter_results"]


class TestCompareSurvival:
    def test_correct_survival_counts(self) -> None:
        from scripts.cascaded_filter_analysis import compare_survival
        from src.run_search import SimulationResult

        _s = SimulationResult
        weak = [
            _s("r0", survived=True, terminated_at=None, termination_reason=None),
            _s("r1", survived=True, terminated_at=None, termination_reason=None),
            _s("r2", survived=False, terminated_at=50, termination_reason="halt"),
            _s("r3", survived=False, terminated_at=10, termination_reason="state_uniform"),
        ]
        medium = [
            _s("r0", survived=True, terminated_at=None, termination_reason=None),
            _s("r1", survived=False, terminated_at=80, termination_reason="short_period"),
            _s("r2", survived=False, terminated_at=50, termination_reason="halt"),
            _s("r3", survived=False, terminated_at=10, termination_reason="state_uniform"),
        ]

        result = compare_survival(weak, medium)
        assert result["weak_survived"] == 2
        assert result["weak_total"] == 4
        assert result["medium_survived"] == 1
        assert result["medium_total"] == 4
        assert result["weak_survival_rate"] == pytest.approx(0.5)
        assert result["medium_survival_rate"] == pytest.approx(0.25)

    def test_empty_results(self) -> None:
        from scripts.cascaded_filter_analysis import compare_survival

        result = compare_survival([], [])
        assert result["weak_survived"] == 0
        assert result["weak_total"] == 0
        assert result["medium_survived"] == 0
        assert result["medium_total"] == 0


class TestCascadedFilterEnd2End:
    def test_full_pipeline_small_run(self, tmp_path: Path) -> None:
        from scripts.cascaded_filter_analysis import (
            compare_survival,
            run_with_medium_filters,
        )
        from src.run_search import run_batch_search

        phase = ObservationPhase.PHASE1_DENSITY
        n_rules = 10

        # Run weak-only
        weak_results = run_batch_search(
            n_rules=n_rules,
            phase=phase,
            out_dir=tmp_path / "weak",
            steps=10,
            halt_window=3,
        )

        # Run with medium filters
        medium_results = run_with_medium_filters(
            n_rules=n_rules,
            phase=phase,
            out_dir=tmp_path / "medium",
            base_rule_seed=0,
            steps=10,
        )

        comparison = compare_survival(weak_results, medium_results)
        assert comparison["weak_total"] == n_rules
        assert comparison["medium_total"] == n_rules
        # Medium filters can only reduce or maintain survival
        assert comparison["medium_survived"] <= comparison["weak_survived"]
