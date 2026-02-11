import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.metrics import action_entropy
from src.rules import ObservationPhase
from src.run_search import (
    ExperimentConfig,
    SearchConfig,
    _entropy_from_action_counts,
    main,
    run_batch_search,
    run_experiment,
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
    actions = [0, 0, 1, 8, 8, 8]
    counts = [0] * 9
    for action in actions:
        counts[action] += 1

    assert _entropy_from_action_counts(counts, len(actions)) == pytest.approx(
        action_entropy(actions)
    )
