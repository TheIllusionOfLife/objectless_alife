"""Tests for scripts/metric_comparison.py."""

import json
import statistics
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.run_search import METRICS_SCHEMA


def _make_metric_row(
    rule_id: str,
    step: int,
    *,
    neighbor_mi: float = 0.1,
    compression_ratio: float = 0.5,
    action_entropy_mean: float = 0.5,
    action_entropy_variance: float = 0.01,
    cluster_count: int = 10,
    quasi_periodicity_peaks: int = 0,
    phase_transition_max_delta: float = 0.01,
    state_entropy: float = 1.0,
    mi_shuffle_null: float = 0.05,
) -> dict:
    return {
        "rule_id": rule_id,
        "step": step,
        "state_entropy": state_entropy,
        "compression_ratio": compression_ratio,
        "predictability_hamming": 0.1 if step > 0 else None,
        "morans_i": 0.0,
        "cluster_count": cluster_count,
        "quasi_periodicity_peaks": quasi_periodicity_peaks,
        "phase_transition_max_delta": phase_transition_max_delta,
        "neighbor_mutual_information": neighbor_mi,
        "action_entropy_mean": action_entropy_mean,
        "action_entropy_variance": action_entropy_variance,
        "block_ncd": 0.3 if step >= 20 else None,
        "mi_shuffle_null": mi_shuffle_null,
    }


def _setup_condition(
    base_dir: Path,
    label: str,
    n_rules: int,
    *,
    mi_base: float = 0.1,
    survived_frac: float = 0.8,
    action_entropy_var: float = 0.01,
) -> None:
    """Create metrics parquet + rule JSONs for a condition."""
    logs_dir = base_dir / label / "logs"
    rules_dir = base_dir / label / "rules"
    logs_dir.mkdir(parents=True)
    rules_dir.mkdir(parents=True)

    rows = []
    for i in range(n_rules):
        for step in range(5):
            rows.append(
                _make_metric_row(
                    f"rule_{i}",
                    step,
                    neighbor_mi=mi_base + i * 0.01,
                    action_entropy_variance=action_entropy_var + i * 0.001,
                )
            )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=METRICS_SCHEMA),
        logs_dir / "metrics_summary.parquet",
    )

    n_survived = int(n_rules * survived_frac)
    for i in range(n_rules):
        payload = {
            "rule_id": f"rule_{i}",
            "table": [0] * 20,
            "survived": i < n_survived,
            "filter_results": {},
            "metadata": {"rule_seed": i, "sim_seed": i},
        }
        (rules_dir / f"rule_{i}.json").write_text(json.dumps(payload))


class TestLoadSurvivorFinalMetrics:
    def test_filters_to_survivors_only(self, tmp_path: Path) -> None:
        from scripts.metric_comparison import load_survivor_final_metrics

        # 10 rules, 80% survive → 8 survivors
        _setup_condition(tmp_path, "test_phase", 10, survived_frac=0.8)
        metrics_path = tmp_path / "test_phase" / "logs" / "metrics_summary.parquet"
        rules_dir = tmp_path / "test_phase" / "rules"

        result = load_survivor_final_metrics(metrics_path, rules_dir)
        # Should have 8 rows (one per survivor, final step)
        assert len(result) == 8

    def test_returns_final_step_only(self, tmp_path: Path) -> None:
        from scripts.metric_comparison import load_survivor_final_metrics

        _setup_condition(tmp_path, "test_phase", 5, survived_frac=1.0)
        metrics_path = tmp_path / "test_phase" / "logs" / "metrics_summary.parquet"
        rules_dir = tmp_path / "test_phase" / "rules"

        result = load_survivor_final_metrics(metrics_path, rules_dir)
        # All 5 rules survive, each has steps 0-4, should get step 4 only
        assert len(result) == 5
        for row in result:
            assert row["step"] == 4


class TestComputeConditionStats:
    def test_returns_median_and_iqr(self) -> None:
        from scripts.metric_comparison import compute_condition_stats

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_condition_stats(values)
        assert stats["median"] == pytest.approx(3.0)
        assert stats["q1"] == pytest.approx(statistics.quantiles(values, n=4)[0])
        assert stats["q3"] == pytest.approx(statistics.quantiles(values, n=4)[2])
        assert stats["n"] == 5

    def test_empty_values_returns_nan(self) -> None:
        from scripts.metric_comparison import compute_condition_stats

        import math

        stats = compute_condition_stats([])
        assert math.isnan(stats["median"])
        assert stats["n"] == 0


class TestCompareConditions:
    def test_clearly_different_conditions_yield_significant_result(self, tmp_path: Path) -> None:
        from scripts.metric_comparison import compare_conditions

        # Two conditions with clearly separated MI distributions
        condition_a = [float(i) * 0.01 for i in range(50)]  # low MI
        condition_b = [float(i) * 0.01 + 1.0 for i in range(50)]  # high MI

        result = compare_conditions(
            {"Phase 1": condition_a, "Phase 2": condition_b},
            metric_name="neighbor_mutual_information",
        )
        assert "Phase 1 vs Phase 2" in result
        comparison = result["Phase 1 vs Phase 2"]
        assert comparison["p_value"] < 0.05

    def test_identical_conditions_yield_nonsignificant(self, tmp_path: Path) -> None:
        from scripts.metric_comparison import compare_conditions

        values = [float(i) * 0.01 for i in range(50)]
        result = compare_conditions(
            {"A": values, "B": values},
            metric_name="test_metric",
        )
        assert "A vs B" in result
        assert result["A vs B"]["p_value"] > 0.05
