"""Tests for scripts/multi_seed_p1_control.py."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.run_search import MULTI_SEED_SCHEMA


def _setup_phase_data(
    base_dir: Path,
    phase_label: str,
    phase_value: int,
    n_rules: int,
    steps: int = 10,
) -> None:
    """Create minimal experiment data (rules + metrics) for a phase."""
    from src.rules import ObservationPhase
    from src.run_search import SearchConfig, run_batch_search

    phase_map = {
        1: ObservationPhase.PHASE1_DENSITY,
        2: ObservationPhase.PHASE2_PROFILE,
        3: ObservationPhase.CONTROL_DENSITY_CLOCK,
    }
    phase = phase_map[phase_value]
    phase_dir = base_dir / phase_label
    run_batch_search(
        n_rules=n_rules,
        phase=phase,
        out_dir=phase_dir,
        steps=steps,
        config=SearchConfig(steps=steps, halt_window=3),
    )


class TestRunMultiSeedForPhase:
    def test_produces_output_parquet_for_phase1(self, tmp_path: Path) -> None:
        from scripts.multi_seed_p1_control import run_multi_seed_for_phase

        from src.rules import ObservationPhase

        data_dir = tmp_path / "data"
        _setup_phase_data(data_dir, "phase_1", 1, n_rules=5, steps=8)

        out_dir = tmp_path / "multi_seed_p1"
        result_path = run_multi_seed_for_phase(
            phase=ObservationPhase.PHASE1_DENSITY,
            data_dir=data_dir,
            out_dir=out_dir,
            top_k=3,
            n_sim_seeds=2,
        )
        assert result_path.exists()
        table = pq.read_table(result_path)
        assert set(MULTI_SEED_SCHEMA.names).issubset(table.column_names)
        assert table.num_rows > 0

    def test_produces_output_parquet_for_control(self, tmp_path: Path) -> None:
        from scripts.multi_seed_p1_control import run_multi_seed_for_phase

        from src.rules import ObservationPhase

        data_dir = tmp_path / "data"
        _setup_phase_data(data_dir, "control", 3, n_rules=5, steps=8)

        out_dir = tmp_path / "multi_seed_ctrl"
        result_path = run_multi_seed_for_phase(
            phase=ObservationPhase.CONTROL_DENSITY_CLOCK,
            data_dir=data_dir,
            out_dir=out_dir,
            top_k=3,
            n_sim_seeds=2,
        )
        assert result_path.exists()
        table = pq.read_table(result_path)
        assert table.num_rows > 0


class TestSummarizeMultiSeedResults:
    def test_returns_expected_fields(self, tmp_path: Path) -> None:
        from scripts.multi_seed_p1_control import summarize_multi_seed_results

        # Write synthetic multi-seed data
        rows = [
            {"rule_seed": 0, "sim_seed": 0, "survived": True, "termination_reason": None,
             "neighbor_mutual_information": 0.5, "mi_shuffle_null": 0.1,
             "mi_excess": 0.4, "same_state_adjacency_fraction": 0.3},
            {"rule_seed": 0, "sim_seed": 1, "survived": True, "termination_reason": None,
             "neighbor_mutual_information": 0.3, "mi_shuffle_null": 0.1,
             "mi_excess": 0.2, "same_state_adjacency_fraction": 0.25},
            {"rule_seed": 1, "sim_seed": 10000, "survived": False, "termination_reason": "halt",
             "neighbor_mutual_information": 0.0, "mi_shuffle_null": 0.0,
             "mi_excess": 0.0, "same_state_adjacency_fraction": 0.0},
            {"rule_seed": 1, "sim_seed": 10001, "survived": True, "termination_reason": None,
             "neighbor_mutual_information": 0.0, "mi_shuffle_null": 0.05,
             "mi_excess": 0.0, "same_state_adjacency_fraction": 0.2},
        ]
        path = tmp_path / "logs" / "multi_seed_results.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=MULTI_SEED_SCHEMA), path)

        summary = summarize_multi_seed_results(path)
        assert "total_rules" in summary
        assert "rules_with_positive_median" in summary
        assert "fraction_with_positive_median" in summary
        assert "mean_positive_fraction" in summary
        assert "overall_survival_rate" in summary
        assert summary["total_rules"] == 2

    def test_correctly_identifies_positive_median_rules(self, tmp_path: Path) -> None:
        from scripts.multi_seed_p1_control import summarize_multi_seed_results

        # Rule 0: both seeds have positive MI_excess → positive median
        # Rule 1: all seeds have zero MI_excess → zero median
        rows = [
            {"rule_seed": 0, "sim_seed": 0, "survived": True, "termination_reason": None,
             "neighbor_mutual_information": 0.5, "mi_shuffle_null": 0.1,
             "mi_excess": 0.4, "same_state_adjacency_fraction": 0.3},
            {"rule_seed": 0, "sim_seed": 1, "survived": True, "termination_reason": None,
             "neighbor_mutual_information": 0.3, "mi_shuffle_null": 0.1,
             "mi_excess": 0.2, "same_state_adjacency_fraction": 0.25},
            {"rule_seed": 1, "sim_seed": 10000, "survived": True, "termination_reason": None,
             "neighbor_mutual_information": 0.05, "mi_shuffle_null": 0.05,
             "mi_excess": 0.0, "same_state_adjacency_fraction": 0.2},
            {"rule_seed": 1, "sim_seed": 10001, "survived": True, "termination_reason": None,
             "neighbor_mutual_information": 0.04, "mi_shuffle_null": 0.05,
             "mi_excess": 0.0, "same_state_adjacency_fraction": 0.2},
        ]
        path = tmp_path / "logs" / "multi_seed_results.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=MULTI_SEED_SCHEMA), path)

        summary = summarize_multi_seed_results(path)
        assert summary["rules_with_positive_median"] == 1
        assert summary["fraction_with_positive_median"] == pytest.approx(0.5)
