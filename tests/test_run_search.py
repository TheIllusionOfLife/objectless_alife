import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.filters import ACTION_SPACE_SIZE
from src.metrics import action_entropy
from src.rules import ObservationPhase
from src.run_search import (
    METRICS_SCHEMA,
    PHASE_SUMMARY_METRIC_NAMES,
    DensitySweepConfig,
    ExperimentConfig,
    MultiSeedConfig,
    SearchConfig,
    _entropy_from_action_counts,
    _parse_grid_sizes,
    _parse_phase,
    main,
    run_batch_search,
    run_density_sweep,
    run_experiment,
    run_multi_seed_robustness,
    select_top_rules_by_excess_mi,
)
from src.world import WorldConfig


def test_run_batch_search_writes_json_and_parquet(tmp_path: Path) -> None:
    results = run_batch_search(
        n_rules=2,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=10,
        halt_window=3,
        base_rule_seed=10,
        base_sim_seed=20,
    )

    assert len(results) == 2

    rules_dir = tmp_path / "rules"
    logs_dir = tmp_path / "logs"
    assert rules_dir.exists()
    assert logs_dir.exists()

    json_files = list(rules_dir.glob("*.json"))
    assert len(json_files) == 2

    sim_table = pq.read_table(logs_dir / "simulation_log.parquet")
    metric_table = pq.read_table(logs_dir / "metrics_summary.parquet")

    sim_columns = {"rule_id", "step", "agent_id", "x", "y", "state", "action"}
    metric_columns = {
        "rule_id",
        "step",
        "state_entropy",
        "compression_ratio",
        "predictability_hamming",
        "morans_i",
        "cluster_count",
        "quasi_periodicity_peaks",
        "phase_transition_max_delta",
        "neighbor_mutual_information",
        "action_entropy_mean",
        "action_entropy_variance",
        "block_ncd",
    }
    assert sim_columns.issubset(sim_table.column_names)
    assert metric_columns.issubset(metric_table.column_names)


def test_run_batch_search_deterministic_rule_ids(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=2,
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=tmp_path,
        steps=5,
        base_rule_seed=7,
        base_sim_seed=11,
    )

    rules_dir = tmp_path / "rules"
    file_names = sorted(path.stem for path in rules_dir.glob("*.json"))
    assert file_names == ["phase2_rs7_ss11", "phase2_rs8_ss12"]


def test_run_batch_search_steps_conflict_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_batch_search(
            n_rules=1,
            phase=ObservationPhase.PHASE1_DENSITY,
            out_dir=tmp_path,
            steps=200,
            world_config=WorldConfig(steps=123),
        )


def test_run_batch_search_phase2_executes(tmp_path: Path) -> None:
    results = run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=tmp_path,
        steps=10,
        base_rule_seed=1,
        base_sim_seed=2,
    )
    assert len(results) == 1

    rules_dir = tmp_path / "rules"
    payload = json.loads(next(rules_dir.glob("*.json")).read_text())
    assert payload["metadata"]["observation_phase"] == 2


def test_run_batch_search_accepts_search_config_with_optional_filters(tmp_path: Path) -> None:
    results = run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        config=SearchConfig(
            steps=10,
            halt_window=3,
            filter_short_period=True,
            short_period_max_period=2,
            short_period_history_size=6,
            filter_low_activity=True,
            low_activity_window=3,
            low_activity_min_unique_ratio=0.2,
        ),
    )
    assert len(results) == 1
    payload = json.loads(next((tmp_path / "rules").glob("*.json")).read_text())
    assert "short_period" in payload["filter_results"]
    assert "low_activity" in payload["filter_results"]


def test_run_batch_search_persists_grid_dimensions_in_metadata(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=5,
        world_config=WorldConfig(grid_width=11, grid_height=13, steps=5),
    )

    payload = json.loads(next((tmp_path / "rules").glob("*.json")).read_text())
    metadata = payload["metadata"]
    assert metadata["grid_width"] == 11
    assert metadata["grid_height"] == 13


def test_run_batch_search_termination_metadata_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run_batch_search(
        n_rules=3,
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=out_a,
        steps=15,
        base_rule_seed=30,
        base_sim_seed=40,
    )
    run_batch_search(
        n_rules=3,
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=out_b,
        steps=15,
        base_rule_seed=30,
        base_sim_seed=40,
    )

    payloads_a = {
        path.stem: json.loads(path.read_text()) for path in sorted((out_a / "rules").glob("*.json"))
    }
    payloads_b = {
        path.stem: json.loads(path.read_text()) for path in sorted((out_b / "rules").glob("*.json"))
    }

    assert payloads_a.keys() == payloads_b.keys()
    for rule_id in payloads_a:
        metadata_a = payloads_a[rule_id]["metadata"]
        metadata_b = payloads_b[rule_id]["metadata"]
        assert metadata_a["terminated_at"] == metadata_b["terminated_at"]
        assert metadata_a["termination_reason"] == metadata_b["termination_reason"]


def test_run_experiment_produces_expected_rows_and_artifacts(tmp_path: Path) -> None:
    results = run_experiment(
        ExperimentConfig(
            phases=(ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE),
            n_rules=3,
            n_seed_batches=2,
            out_dir=tmp_path,
            steps=8,
            halt_window=3,
            rule_seed_start=10,
            sim_seed_start=20,
        )
    )

    assert len(results) == 12
    logs_dir = tmp_path / "logs"
    assert (logs_dir / "experiment_runs.parquet").exists()
    assert (logs_dir / "phase_summary.parquet").exists()
    assert (logs_dir / "phase_comparison.json").exists()

    runs = pq.read_table(logs_dir / "experiment_runs.parquet")
    rows = runs.to_pylist()
    assert len(rows) == 12
    assert {int(row["phase"]) for row in rows} == {1, 2}


def test_run_experiment_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    config_a = ExperimentConfig(
        phases=(ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE),
        n_rules=2,
        n_seed_batches=2,
        out_dir=out_a,
        steps=6,
        halt_window=3,
        rule_seed_start=100,
        sim_seed_start=200,
    )
    config_b = ExperimentConfig(
        phases=(ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE),
        n_rules=2,
        n_seed_batches=2,
        out_dir=out_b,
        steps=6,
        halt_window=3,
        rule_seed_start=100,
        sim_seed_start=200,
    )

    run_experiment(config_a)
    run_experiment(config_b)

    runs_a = pq.read_table(out_a / "logs" / "experiment_runs.parquet").to_pylist()
    runs_b = pq.read_table(out_b / "logs" / "experiment_runs.parquet").to_pylist()
    assert runs_a == runs_b

    phase_cmp_a = json.loads((out_a / "logs" / "phase_comparison.json").read_text())
    phase_cmp_b = json.loads((out_b / "logs" / "phase_comparison.json").read_text())
    assert phase_cmp_a == phase_cmp_b


def test_run_experiment_phase_summary_has_required_columns(tmp_path: Path) -> None:
    run_experiment(
        ExperimentConfig(
            phases=(ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE),
            n_rules=2,
            n_seed_batches=1,
            out_dir=tmp_path,
            steps=6,
        )
    )

    summary = pq.read_table(tmp_path / "logs" / "phase_summary.parquet")
    expected_columns = {
        "schema_version",
        "phase",
        "rules_evaluated",
        "survival_rate",
        "termination_rate",
        "mean_terminated_at",
        "state_entropy_mean",
        "state_entropy_p50",
        "quasi_periodicity_peaks_mean",
        "phase_transition_max_delta_mean",
        "block_ncd_mean",
    }
    assert expected_columns.issubset(summary.column_names)
    assert summary.num_rows == 2


def test_run_experiment_phase_comparison_excludes_metadata_delta_keys(tmp_path: Path) -> None:
    run_experiment(
        ExperimentConfig(
            phases=(ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE),
            n_rules=2,
            n_seed_batches=1,
            out_dir=tmp_path,
            steps=6,
        )
    )

    payload = json.loads((tmp_path / "logs" / "phase_comparison.json").read_text())
    deltas = payload["deltas"]
    assert "phase" not in deltas
    assert "schema_version" not in deltas


def test_run_search_main_experiment_mode_generates_aggregate_files(tmp_path: Path) -> None:
    main(
        [
            "--experiment",
            "--phases",
            "1,2",
            "--seed-batches",
            "2",
            "--n-rules",
            "2",
            "--steps",
            "6",
            "--out-dir",
            str(tmp_path),
        ]
    )

    logs_dir = tmp_path / "logs"
    assert (logs_dir / "experiment_runs.parquet").exists()
    assert (logs_dir / "phase_summary.parquet").exists()
    assert (logs_dir / "phase_comparison.json").exists()


def test_run_search_main_experiment_mode_rejects_more_than_two_phases(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly two"):
        main(
            [
                "--experiment",
                "--phases",
                "1,2,1",
                "--seed-batches",
                "1",
                "--n-rules",
                "1",
                "--out-dir",
                str(tmp_path),
            ]
        )


def test_run_search_main_experiment_mode_rejects_invalid_phase_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        main(
            [
                "--experiment",
                "--phases",
                "x,2",
                "--seed-batches",
                "1",
                "--n-rules",
                "1",
                "--out-dir",
                str(tmp_path),
            ]
        )


def test_run_experiment_rejects_excessive_total_workload(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workload"):
        run_experiment(
            ExperimentConfig(
                phases=(ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE),
                n_rules=10_000,
                n_seed_batches=1_000,
                steps=2_000,
                out_dir=tmp_path,
            )
        )


def test_run_batch_search_does_not_leave_partial_parquet_files_on_early_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.run_search as run_search_module

    def _fail(*args: object, **kwargs: object) -> list[int]:
        raise RuntimeError("boom")

    monkeypatch.setattr(run_search_module, "generate_rule_table", _fail)
    with pytest.raises(RuntimeError, match="boom"):
        run_batch_search(
            n_rules=1,
            phase=ObservationPhase.PHASE1_DENSITY,
            out_dir=tmp_path,
            steps=4,
        )

    assert not (tmp_path / "logs" / "simulation_log.parquet").exists()
    assert not (tmp_path / "logs" / "metrics_summary.parquet").exists()


def test_run_batch_search_metrics_schema_is_stable_when_column_values_are_all_nulls(
    tmp_path: Path,
) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        config=SearchConfig(steps=3, block_ncd_window=99),
    )

    metrics = pq.read_table(tmp_path / "logs" / "metrics_summary.parquet")
    schema = metrics.schema
    assert str(schema.field("block_ncd").type) == "double"
    assert str(schema.field("predictability_hamming").type) == "double"


def test_entropy_from_action_counts_matches_sequence_entropy() -> None:
    # We intentionally test this private helper because it encodes the
    # incremental entropy math used by the hot simulation loop.
    actions = [0, 0, 1, 8, 8, 8]
    counts = [0] * ACTION_SPACE_SIZE
    for action in actions:
        counts[action] += 1

    assert _entropy_from_action_counts(counts, len(actions)) == pytest.approx(
        action_entropy(actions)
    )


def test_run_batch_search_fixed_seed_metrics_rows_are_stable(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        base_rule_seed=123,
        base_sim_seed=456,
        config=SearchConfig(steps=8, halt_window=3, block_ncd_window=4),
    )

    metrics = pq.read_table(tmp_path / "logs" / "metrics_summary.parquet").to_pylist()
    assert len(metrics) == 8

    first = metrics[0]
    assert first["rule_id"] == "phase1_rs123_ss456"
    assert first["step"] == 0
    assert first["state_entropy"] == pytest.approx(0.783776947484701)
    assert first["compression_ratio"] == pytest.approx(0.1525)
    assert first["predictability_hamming"] is None
    assert first["morans_i"] == pytest.approx(-0.7391304347826086)
    assert first["cluster_count"] == 29
    assert first["quasi_periodicity_peaks"] == 0
    assert first["phase_transition_max_delta"] == pytest.approx(0.0)
    assert first["neighbor_mutual_information"] == pytest.approx(0.32192809488736224)
    assert first["action_entropy_mean"] == pytest.approx(0.0)
    assert first["action_entropy_variance"] == pytest.approx(0.0)
    assert first["block_ncd"] is None

    last = metrics[-1]
    assert last["step"] == 7
    assert last["state_entropy"] == pytest.approx(0.783776947484701)
    assert last["compression_ratio"] == pytest.approx(0.1575)
    assert last["predictability_hamming"] == pytest.approx(0.0)
    assert last["morans_i"] == pytest.approx(-0.015084294587400132)
    assert last["cluster_count"] == 27
    assert last["quasi_periodicity_peaks"] == 0
    assert last["phase_transition_max_delta"] == pytest.approx(0.3267753036985779)
    assert last["neighbor_mutual_information"] == pytest.approx(0.0)
    assert last["action_entropy_mean"] == pytest.approx(0.8866142457376974)
    assert last["action_entropy_variance"] == pytest.approx(0.2869433329132004)
    assert last["block_ncd"] == pytest.approx(0.8421052631578947)


def test_run_batch_search_fixed_seed_simulation_rows_are_stable(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        base_rule_seed=123,
        base_sim_seed=456,
        config=SearchConfig(steps=8, halt_window=3, block_ncd_window=4),
    )

    sim = pq.read_table(tmp_path / "logs" / "simulation_log.parquet").to_pylist()
    assert len(sim) == 240
    assert sim[0] == {
        "rule_id": "phase1_rs123_ss456",
        "step": 0,
        "agent_id": 0,
        "x": 14,
        "y": 13,
        "state": 0,
        "action": 0,
    }
    assert sim[-1] == {
        "rule_id": "phase1_rs123_ss456",
        "step": 7,
        "agent_id": 29,
        "x": 14,
        "y": 19,
        "state": 1,
        "action": 1,
    }


def test_parse_grid_sizes_accepts_multiple_pairs() -> None:
    assert _parse_grid_sizes("5x7,20x20") == ((5, 7), (20, 20))


def test_parse_grid_sizes_accepts_uppercase_separator() -> None:
    assert _parse_grid_sizes("20X20,5x7") == ((20, 20), (5, 7))


@pytest.mark.parametrize("raw", ["", "20", "20x", "x20", "0x20", "20x0", "-1x20"])
def test_parse_grid_sizes_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        _parse_grid_sizes(raw)


def test_run_density_sweep_produces_expected_rows_and_artifacts(tmp_path: Path) -> None:
    results = run_density_sweep(
        DensitySweepConfig(
            grid_sizes=((5, 5),),
            agent_counts=(3, 5),
            n_rules=2,
            n_seed_batches=1,
            out_dir=tmp_path,
            steps=6,
            halt_window=3,
            rule_seed_start=10,
            sim_seed_start=20,
        )
    )

    assert len(results) == 8

    logs_dir = tmp_path / "logs"
    assert (logs_dir / "density_sweep_runs.parquet").exists()
    assert (logs_dir / "density_phase_summary.parquet").exists()
    assert (logs_dir / "density_phase_comparison.parquet").exists()

    runs = pq.read_table(logs_dir / "density_sweep_runs.parquet").to_pylist()
    assert len(runs) == 8
    assert {int(row["phase"]) for row in runs} == {1, 2}
    assert {int(row["num_agents"]) for row in runs} == {3, 5}
    assert {float(row["density_ratio"]) for row in runs} == {0.12, 0.2}


def test_run_density_sweep_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    config_a = DensitySweepConfig(
        grid_sizes=((5, 5),),
        agent_counts=(3, 5),
        n_rules=1,
        n_seed_batches=1,
        out_dir=out_a,
        steps=5,
        halt_window=3,
        rule_seed_start=100,
        sim_seed_start=200,
    )
    config_b = DensitySweepConfig(
        grid_sizes=((5, 5),),
        agent_counts=(3, 5),
        n_rules=1,
        n_seed_batches=1,
        out_dir=out_b,
        steps=5,
        halt_window=3,
        rule_seed_start=100,
        sim_seed_start=200,
    )

    run_density_sweep(config_a)
    run_density_sweep(config_b)

    runs_a = pq.read_table(out_a / "logs" / "density_sweep_runs.parquet").to_pylist()
    runs_b = pq.read_table(out_b / "logs" / "density_sweep_runs.parquet").to_pylist()
    assert runs_a == runs_b

    summary_a = pq.read_table(out_a / "logs" / "density_phase_summary.parquet").to_pylist()
    summary_b = pq.read_table(out_b / "logs" / "density_phase_summary.parquet").to_pylist()
    assert summary_a == summary_b

    cmp_a = pq.read_table(out_a / "logs" / "density_phase_comparison.parquet").to_pylist()
    cmp_b = pq.read_table(out_b / "logs" / "density_phase_comparison.parquet").to_pylist()
    assert cmp_a == cmp_b


def test_run_search_main_density_sweep_mode_generates_aggregate_files(tmp_path: Path) -> None:
    main(
        [
            "--density-sweep",
            "--grid-sizes",
            "5x5",
            "--agent-counts",
            "3,5",
            "--n-rules",
            "1",
            "--seed-batches",
            "1",
            "--steps",
            "5",
            "--out-dir",
            str(tmp_path),
        ]
    )

    logs_dir = tmp_path / "logs"
    assert (logs_dir / "density_sweep_runs.parquet").exists()
    assert (logs_dir / "density_phase_summary.parquet").exists()
    assert (logs_dir / "density_phase_comparison.parquet").exists()


def test_run_batch_search_control_phase_produces_valid_artifacts(tmp_path: Path) -> None:
    results = run_batch_search(
        n_rules=2,
        phase=ObservationPhase.CONTROL_DENSITY_CLOCK,
        out_dir=tmp_path,
        steps=10,
        base_rule_seed=0,
        base_sim_seed=0,
    )
    assert len(results) == 2

    rules_dir = tmp_path / "rules"
    json_files = list(rules_dir.glob("*.json"))
    assert len(json_files) == 2

    payload = json.loads(json_files[0].read_text())
    assert payload["metadata"]["observation_phase"] == 3
    assert len(payload["table"]) == 100

    metric_table = pq.read_table(tmp_path / "logs" / "metrics_summary.parquet")
    assert metric_table.num_rows > 0


def test_parse_phase_returns_control_for_3() -> None:
    assert _parse_phase(3) == ObservationPhase.CONTROL_DENSITY_CLOCK


def test_mi_shuffle_null_in_metrics_schema() -> None:
    """mi_shuffle_null column exists in METRICS_SCHEMA."""
    field_names = [field.name for field in METRICS_SCHEMA]
    assert "mi_shuffle_null" in field_names


def test_mi_shuffle_null_in_phase_summary_metric_names() -> None:
    """mi_shuffle_null is included in PHASE_SUMMARY_METRIC_NAMES."""
    assert "mi_shuffle_null" in PHASE_SUMMARY_METRIC_NAMES


def test_mi_shuffle_null_column_in_output_parquet(tmp_path: Path) -> None:
    """mi_shuffle_null column appears in metrics_summary.parquet output."""
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=5,
    )
    metrics = pq.read_table(tmp_path / "logs" / "metrics_summary.parquet")
    assert "mi_shuffle_null" in metrics.column_names
    # Should be a constant value (backfilled to all steps)
    values = metrics.column("mi_shuffle_null").to_pylist()
    assert all(v == values[0] for v in values)


def test_mi_excess_in_phase_summary_metric_names() -> None:
    """mi_excess is included in PHASE_SUMMARY_METRIC_NAMES."""
    assert "mi_excess" in PHASE_SUMMARY_METRIC_NAMES


def test_mi_excess_columns_in_phase_summary(tmp_path: Path) -> None:
    """Phase summary parquet contains mi_excess_mean/p25/p50/p75 columns."""
    run_experiment(
        ExperimentConfig(
            phases=(ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE),
            n_rules=2,
            n_seed_batches=1,
            out_dir=tmp_path,
            steps=6,
        )
    )
    summary = pq.read_table(tmp_path / "logs" / "phase_summary.parquet")
    for suffix in ("mean", "p25", "p50", "p75"):
        assert f"mi_excess_{suffix}" in summary.column_names


def test_mi_excess_columns_in_density_phase_summary(tmp_path: Path) -> None:
    """Density phase summary parquet contains mi_excess columns."""
    run_density_sweep(
        DensitySweepConfig(
            grid_sizes=((5, 5),),
            agent_counts=(3,),
            n_rules=2,
            n_seed_batches=1,
            out_dir=tmp_path,
            steps=6,
            halt_window=3,
        )
    )
    summary = pq.read_table(tmp_path / "logs" / "density_phase_summary.parquet")
    for suffix in ("mean", "p25", "p50", "p75"):
        assert f"mi_excess_{suffix}" in summary.column_names


def test_mi_excess_is_nonnegative(tmp_path: Path) -> None:
    """mi_excess values in phase summary are always >= 0."""
    run_experiment(
        ExperimentConfig(
            phases=(ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE),
            n_rules=3,
            n_seed_batches=1,
            out_dir=tmp_path,
            steps=8,
        )
    )
    summary = pq.read_table(tmp_path / "logs" / "phase_summary.parquet").to_pylist()
    for row in summary:
        for suffix in ("mean", "p25", "p50", "p75"):
            val = row[f"mi_excess_{suffix}"]
            if val is not None:
                assert val >= 0.0, f"mi_excess_{suffix} = {val} < 0"


def test_select_top_rules_by_excess_mi(tmp_path: Path) -> None:
    """select_top_rules_by_excess_mi returns top-K rule seeds by MI_excess."""
    # Run a small experiment to generate data
    run_experiment(
        ExperimentConfig(
            phases=(ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE),
            n_rules=5,
            n_seed_batches=1,
            out_dir=tmp_path,
            steps=10,
        )
    )
    metrics_path = tmp_path / "phase_2" / "logs" / "metrics_summary.parquet"
    rules_dir = tmp_path / "phase_2" / "rules"
    top = select_top_rules_by_excess_mi(metrics_path, rules_dir, top_k=3)
    assert len(top) <= 3
    assert all(isinstance(seed, int) for seed in top)


def test_multi_seed_robustness_output_schema(tmp_path: Path) -> None:
    """run_multi_seed_robustness produces parquet with expected columns."""
    # First generate source data
    run_experiment(
        ExperimentConfig(
            phases=(ObservationPhase.PHASE1_DENSITY, ObservationPhase.PHASE2_PROFILE),
            n_rules=3,
            n_seed_batches=1,
            out_dir=tmp_path / "source",
            steps=8,
        )
    )
    config = MultiSeedConfig(
        rule_seeds=(0, 1),
        n_sim_seeds=3,
        out_dir=tmp_path / "multi_seed",
        steps=8,
        halt_window=3,
    )
    run_multi_seed_robustness(config)
    output_path = tmp_path / "multi_seed" / "logs" / "multi_seed_results.parquet"
    assert output_path.exists()
    table = pq.read_table(output_path)
    expected_cols = {
        "rule_seed",
        "sim_seed",
        "survived",
        "neighbor_mutual_information",
        "mi_shuffle_null",
        "mi_excess",
    }
    assert expected_cols.issubset(table.column_names)
    assert table.num_rows == 2 * 3  # 2 rules x 3 seeds


def test_run_search_main_rejects_density_sweep_and_experiment_together(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--density-sweep",
                "--experiment",
                "--grid-sizes",
                "5x5",
                "--agent-counts",
                "3",
                "--n-rules",
                "1",
                "--out-dir",
                str(tmp_path),
            ]
        )
