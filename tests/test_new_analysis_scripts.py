"""Smoke tests for new analysis scripts (2026-02-23 review response).

All synthetic data, no real Parquet files required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _make_synthetic_experiment_runs(path: Path, n_rules: int = 10) -> None:
    """Create a synthetic experiment_runs.parquet with phases 1 and 2."""
    rows = []
    for i in range(n_rules):
        for phase in (1, 2):
            if i % 5 == 0:
                survived = False
                reason = "halt"
            elif i % 5 == 1:
                survived = False
                reason = "state_uniform"
            else:
                survived = True
                reason = None
            rows.append(
                {
                    "schema_version": 1,
                    "rule_id": f"phase{phase}_rs{i}_ss0",
                    "phase": phase,
                    "seed_batch": 0,
                    "rule_seed": i,
                    "sim_seed": 0,
                    "survived": survived,
                    "termination_reason": reason,
                    "terminated_at": None if survived else 50,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)


def _make_synthetic_metrics_parquet(path: Path, n_rules: int = 10, max_step: int = 199) -> None:
    """Create a synthetic metrics_summary.parquet with one row per rule at final step."""
    rows = []
    for i in range(n_rules):
        # Survivors get step=199, others get step=50
        step = max_step if i % 5 >= 2 else 50
        rows.append(
            {
                "rule_id": f"rule_{i}",
                "step": step,
                "state_entropy": 1.5,
                "compression_ratio": 0.5,
                "predictability_hamming": 0.3,
                "morans_i": 0.1,
                "cluster_count": 3,
                "quasi_periodicity_peaks": 0,
                "phase_transition_max_delta": 0.01,
                "neighbor_mutual_information": float(i) * 0.1,
                "action_entropy_mean": 1.0,
                "action_entropy_variance": 0.01,
                "block_ncd": 0.8,
                "mi_shuffle_null": 0.004,
                "same_state_adjacency_fraction": 0.4,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def _make_synthetic_metrics_multi_step(path: Path, n_rules: int = 5) -> None:
    """Create a metrics_summary with multiple steps per rule (for control counts test)."""
    rows = []
    for i in range(n_rules):
        max_step = 199 if i % 2 == 0 else 50
        for step in range(0, max_step + 1, 50):
            rows.append(
                {
                    "rule_id": f"ctrl_rule_{i}",
                    "step": step,
                    "state_entropy": 1.5,
                    "compression_ratio": 0.5,
                    "predictability_hamming": 0.3,
                    "morans_i": 0.1,
                    "cluster_count": 3,
                    "quasi_periodicity_peaks": 0,
                    "phase_transition_max_delta": 0.01,
                    "neighbor_mutual_information": 0.1,
                    "action_entropy_mean": 1.0,
                    "action_entropy_variance": 0.01,
                    "block_ncd": 0.8,
                    "mi_shuffle_null": 0.004,
                    "same_state_adjacency_fraction": 0.4,
                }
            )
        # Add the final step row
        if max_step % 50 != 0:
            rows.append(
                {
                    "rule_id": f"ctrl_rule_{i}",
                    "step": max_step,
                    "state_entropy": 1.5,
                    "compression_ratio": 0.5,
                    "predictability_hamming": 0.3,
                    "morans_i": 0.1,
                    "cluster_count": 3,
                    "quasi_periodicity_peaks": 0,
                    "phase_transition_max_delta": 0.01,
                    "neighbor_mutual_information": 0.1,
                    "action_entropy_mean": 1.0,
                    "action_entropy_variance": 0.01,
                    "block_ncd": 0.8,
                    "mi_shuffle_null": 0.004,
                    "same_state_adjacency_fraction": 0.4,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_termination_breakdown_smoke(tmp_path: Path) -> None:
    """load_termination_counts and compute_full_dist_delta_mi return correct structure."""
    from scripts.termination_breakdown import (
        compute_full_dist_delta_mi,
        load_control_counts,
        load_termination_counts,
    )

    # Create synthetic experiment_runs
    exp_runs_path = tmp_path / "experiment_runs.parquet"
    _make_synthetic_experiment_runs(exp_runs_path, n_rules=10)

    # Create synthetic metrics
    metrics_path = tmp_path / "metrics_summary.parquet"
    _make_synthetic_metrics_parquet(metrics_path, n_rules=10)

    # Test load_termination_counts for phase 1
    counts_p1 = load_termination_counts(exp_runs_path, phase_value=1)
    assert "n_survived" in counts_p1
    assert "n_halt" in counts_p1
    assert "n_state_uniform" in counts_p1
    assert "n_total" in counts_p1
    assert isinstance(counts_p1["n_survived"], int)
    assert isinstance(counts_p1["n_total"], int)
    assert counts_p1["n_total"] == 10
    assert counts_p1["n_survived"] + counts_p1["n_halt"] + counts_p1["n_state_uniform"] == 10

    # Test load_termination_counts for phase 2
    counts_p2 = load_termination_counts(exp_runs_path, phase_value=2)
    assert counts_p2["n_total"] == 10

    # Test load_control_counts
    ctrl_metrics_path = tmp_path / "ctrl_metrics.parquet"
    _make_synthetic_metrics_multi_step(ctrl_metrics_path, n_rules=5)
    ctrl_counts = load_control_counts(ctrl_metrics_path)
    assert "n_survived" in ctrl_counts
    assert "n_not_recorded" in ctrl_counts
    assert "n_total" in ctrl_counts
    assert ctrl_counts["n_total"] == 5
    # Rules 0, 2, 4 have max_step=199 (survived); rules 1, 3 terminated
    assert ctrl_counts["n_survived"] == 3
    assert ctrl_counts["n_not_recorded"] == 2

    # Test compute_full_dist_delta_mi
    delta_mi = compute_full_dist_delta_mi(metrics_path)
    assert "median" in delta_mi
    assert "fraction_positive" in delta_mi
    assert isinstance(delta_mi["median"], float)
    assert 0.0 <= delta_mi["fraction_positive"] <= 1.0


def test_population_mi_timeseries_smoke() -> None:
    """compute_population_delta_mi_timeseries exists and returns expected structure."""
    import math

    from scripts.population_mi_timeseries import compute_population_delta_mi_timeseries

    # Create tiny synthetic snapshots: {rule_id: {step: ((aid, x, y, state), ...)}}
    snapshots = {
        "rule_0": {
            0: ((0, 0, 0, 1), (1, 1, 0, 2), (2, 0, 1, 1)),
            10: ((0, 0, 0, 2), (1, 1, 0, 1), (2, 0, 1, 3)),
        },
        "rule_1": {
            0: ((0, 2, 2, 0), (1, 3, 2, 0), (2, 2, 3, 1)),
            10: ((0, 2, 2, 1), (1, 3, 2, 1), (2, 2, 3, 0)),
        },
    }

    result = compute_population_delta_mi_timeseries(
        snapshots, timesteps=[0, 10], n_shuffles=5, seed=42
    )

    assert set(result.keys()) == {0, 10}
    for t in (0, 10):
        assert "median" in result[t]
        assert "q25" in result[t]
        assert "q75" in result[t]
        assert isinstance(result[t]["median"], float)

    # Degenerate: empty snapshots dict → all timestep entries with NaN
    result_empty = compute_population_delta_mi_timeseries(
        {}, timesteps=[0, 10], n_shuffles=5, seed=42
    )
    assert set(result_empty.keys()) == {0, 10}
    for t in (0, 10):
        assert "median" in result_empty[t]
        assert "q25" in result_empty[t]
        assert "q75" in result_empty[t]
        assert math.isnan(result_empty[t]["median"])

    # Degenerate: rule with empty steps dict → handled gracefully
    result_empty_rule = compute_population_delta_mi_timeseries(
        {"rule_empty": {}}, timesteps=[0, 10], n_shuffles=5, seed=42
    )
    assert set(result_empty_rule.keys()) == {0, 10}
    for t in (0, 10):
        assert math.isnan(result_empty_rule[t]["median"])

    # Degenerate: single-agent snapshot → returns valid float (even if MI is 0)
    result_single = compute_population_delta_mi_timeseries(
        {"rule_single": {0: ((0, 0, 0, 1),), 10: ((0, 0, 0, 2),)}},
        timesteps=[0, 10],
        n_shuffles=5,
        seed=42,
    )
    assert set(result_single.keys()) == {0, 10}
    for t in (0, 10):
        assert isinstance(result_single[t]["median"], float)


def test_random_survivor_multi_seed_smoke() -> None:
    """compute_bootstrap_ci returns correct structure with mean in CI bounds."""
    from scripts.random_survivor_multi_seed import compute_bootstrap_ci

    values = [0.5, 0.6, 0.55, 0.65, 0.7]
    ci = compute_bootstrap_ci(values, ci=0.95)

    assert "mean" in ci
    assert "ci_lower" in ci
    assert "ci_upper" in ci
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    # Test empty input
    empty_ci = compute_bootstrap_ci([], ci=0.95)
    assert empty_ci["mean"] == 0.0


def test_phenotype_taxonomy_all_survivors(tmp_path: Path) -> None:
    """run_all_survivors_analysis returns correct structure with quartile counts."""
    from scripts.phenotype_taxonomy import run_all_survivors_analysis

    # Create synthetic metrics with all survivors (step=199)
    rows = []
    for i in range(20):
        rows.append(
            {
                "rule_id": f"rule_{i}",
                "step": 199,
                "state_entropy": 1.5 + (i % 3) * 0.2,
                "compression_ratio": 0.5,
                "predictability_hamming": 0.1 + (i % 4) * 0.15,
                "morans_i": 0.1,
                "cluster_count": 3,
                "quasi_periodicity_peaks": 0,
                "phase_transition_max_delta": 0.01,
                "neighbor_mutual_information": float(i) * 0.02,
                "action_entropy_mean": 1.0,
                "action_entropy_variance": 0.01,
                "block_ncd": 0.8,
                "mi_shuffle_null": 0.004,
                "same_state_adjacency_fraction": 0.3 + (i % 5) * 0.1,
            }
        )
    metrics_path = tmp_path / "metrics_summary.parquet"
    pq.write_table(pa.Table.from_pylist(rows), metrics_path)

    out_dir = tmp_path / "archetype_full"
    result = run_all_survivors_analysis(metrics_path, out_dir=out_dir)

    assert result["n_survivors"] == 20
    assert "median_state_entropy" in result
    assert "median_delta_mi" in result
    assert "predictability_fraction_median" in result
    assert "overall_counts" in result
    assert "quartile_archetype_counts" in result

    # Check overall_counts has all archetype keys
    for archetype in ("polarized_cluster", "frozen_patch", "mixed_turbulent", "low_signal"):
        assert archetype in result["overall_counts"]

    # Check quartile keys
    for qkey in ("q1_0_25", "q2_25_50", "q3_50_75", "q4_75_100"):
        assert qkey in result["quartile_archetype_counts"]
        q_data = result["quartile_archetype_counts"][qkey]
        assert "delta_mi_range" in q_data
        assert len(q_data["delta_mi_range"]) == 2

    # Check output file was written
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text())
    assert loaded["n_survivors"] == 20
