import csv
import json
from pathlib import Path

import pyarrow as pa

import scripts.viability_filter_ablation as viability_ablation_module
from objectless_alife.config import ExperimentConfig
from objectless_alife.rules import ObservationPhase
from objectless_alife.run_search import run_experiment
from scripts.no_filter_analysis import main as no_filter_main
from scripts.phenotype_taxonomy import main as taxonomy_main
from scripts.pr26_followups_manifest_paths import collect_manifest_output_paths
from scripts.ranking_stability import main as ranking_main
from scripts.render_pr26_followups_tex import main as render_tex_main
from scripts.run_pr26_followups import main as run_all_followups_main
from scripts.synchronous_ablation import main as sync_main
from scripts.te_null_analysis import main as te_main
from scripts.viability_filter_ablation import main as viability_ablation_main


def _make_small_dataset(data_dir: Path, sim_seed_start: int = 0) -> None:
    run_experiment(
        ExperimentConfig(
            phases=(
                ObservationPhase.PHASE1_DENSITY,
                ObservationPhase.PHASE2_PROFILE,
                ObservationPhase.CONTROL_DENSITY_CLOCK,
            ),
            n_rules=2,
            n_seed_batches=1,
            out_dir=data_dir,
            steps=5,
            rule_seed_start=0,
            sim_seed_start=sim_seed_start,
        )
    )


def test_no_filter_analysis_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "nf"
    no_filter_main(
        [
            "--out-dir",
            str(out_dir),
            "--n-rules",
            "1",
            "--steps",
            "4",
            "--seed-batches",
            "1",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    assert "phase_pairwise" in payload
    assert (out_dir / "summary.csv").exists()


def test_synchronous_ablation_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "sync"
    sync_main(
        [
            "--out-dir",
            str(out_dir),
            "--n-rules",
            "1",
            "--steps",
            "4",
            "--seed-batches",
            "1",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    assert "phase_pairwise" in payload
    assert (out_dir / "summary.csv").exists()


def test_ranking_stability_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "stability"
    ranking_main(
        [
            "--out-dir",
            str(out_dir),
            "--n-rules",
            "2",
            "--steps",
            "4",
            "--n-seed-batches",
            "2",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    assert "pairwise_kendall_tau" in payload
    assert payload["alignment_key"] == "rule_seed"
    assert (out_dir / "summary.csv").exists()


def test_ranking_stability_rule_id_alignment_can_be_disjoint(tmp_path: Path) -> None:
    out_dir = tmp_path / "stability_rule_id"
    ranking_main(
        [
            "--out-dir",
            str(out_dir),
            "--n-rules",
            "2",
            "--steps",
            "4",
            "--n-seed-batches",
            "2",
            "--alignment-key",
            "rule_id",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    phase2_rows = payload["pairwise_kendall_tau"]["2"]
    assert len(phase2_rows) == 1
    assert phase2_rows[0]["alignment_key"] == "rule_id"
    assert phase2_rows[0]["n_rules"] == 0
    with (out_dir / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    phase2_csv_rows = [row for row in rows if row["phase"] == "2"]
    assert len(phase2_csv_rows) == 1
    assert phase2_csv_rows[0]["kendall_tau"] == ""


def test_te_null_analysis_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _make_small_dataset(data_dir)
    out_dir = tmp_path / "te"
    te_main(
        [
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--top-k",
            "2",
            "--n-shuffles",
            "10",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    assert "conditions" in payload
    assert (out_dir / "summary.csv").exists()


def test_te_null_analysis_uses_experiment_run_mapping(tmp_path: Path) -> None:
    data_dir = tmp_path / "mapped_data"
    _make_small_dataset(data_dir, sim_seed_start=100)
    out_dir = tmp_path / "te_mapped"
    te_main(
        [
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--top-k",
            "2",
            "--n-shuffles",
            "10",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    assert payload["conditions"]["phase_2"]["n_rules"] >= 1


def test_phenotype_taxonomy_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_tax"
    _make_small_dataset(data_dir)
    out_dir = tmp_path / "taxonomy"
    taxonomy_main(
        [
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--top-k",
            "2",
        ]
    )
    payload = json.loads((out_dir / "taxonomy.json").read_text())
    assert "rows" in payload
    assert (out_dir / "taxonomy.csv").exists()


def test_phenotype_taxonomy_handles_missing_adjacency_column(tmp_path: Path, monkeypatch) -> None:
    table = pa.table(
        {
            "rule_id": ["r1", "r2"],
            "state_entropy": [0.9, 0.1],
            "predictability_hamming": [0.7, 0.1],
            "delta_mi": [0.2, 0.0],
        }
    )
    monkeypatch.setattr("scripts.phenotype_taxonomy.load_final_step_metrics", lambda _: table)

    out_dir = tmp_path / "taxonomy_missing_adj"
    taxonomy_main(
        [
            "--data-dir",
            str(tmp_path / "unused"),
            "--out-dir",
            str(out_dir),
            "--top-k",
            "2",
        ]
    )
    payload = json.loads((out_dir / "taxonomy.json").read_text())
    assert "rows" in payload
    assert len(payload["rows"]) == 2
    assert (out_dir / "taxonomy.csv").exists()


def test_run_pr26_followups_orchestrator_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "orchestrator_data"
    _make_small_dataset(data_dir)
    out_dir = tmp_path / "orchestrator_out"
    run_all_followups_main(
        [
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--quick",
        ]
    )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert "outputs" in manifest
    assert "git_commit" in manifest
    assert "commands" in manifest
    assert manifest["schema_version"] == "1.0"
    assert "generated_at_utc" in manifest
    assert "command_line" in manifest
    assert "python_version" in manifest
    assert "uv_version" in manifest
    assert "platform" in manifest
    assert "git_branch" in manifest
    assert "zenodo" in manifest
    checksums_path = out_dir / "checksums.sha256"
    assert checksums_path.exists()
    checksum_lines = [line for line in checksums_path.read_text().splitlines() if line.strip()]
    assert any(line.endswith("manifest.json") for line in checksum_lines)
    assert not any("/rules/" in line for line in checksum_lines)
    assert (out_dir / "no_filter" / "summary.json").exists()
    assert (out_dir / "synchronous_ablation" / "summary.json").exists()
    assert (out_dir / "ranking_stability" / "summary.json").exists()
    assert (out_dir / "te_null" / "summary.json").exists()
    assert (out_dir / "phenotypes" / "taxonomy.json").exists()


def test_render_pr26_followups_tex_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "tex_data"
    _make_small_dataset(data_dir)
    out_dir = tmp_path / "tex_out"
    run_all_followups_main(
        [
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--quick",
        ]
    )
    tex_path = tmp_path / "generated" / "pr26_followups.tex"
    render_tex_main(
        [
            "--followup-dir",
            str(out_dir),
            "--output",
            str(tex_path),
        ]
    )
    contents = tex_path.read_text()
    assert "\\newcommand{\\PrTwentySixManifestCommit}" in contents
    assert "\\newcommand{\\PrTwentySixPhaseTwoTeExcessMedian}" in contents
    assert "\\newcommand{\\PrTwentySixManifestDoi}" in contents


def test_collect_manifest_output_paths_resolves_manifest_relative(tmp_path: Path) -> None:
    out_dir = tmp_path / "bundle"
    output_json = out_dir / "no_filter" / "summary.json"
    output_csv = out_dir / "no_filter" / "summary.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text('{"ok": true}\n')
    output_csv.write_text("k,v\nok,1\n")
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "outputs": {
            "no_filter": {
                "json": "no_filter/summary.json",
                "csv": "no_filter/summary.csv",
            }
        }
    }
    manifest_path.write_text(json.dumps(manifest))

    targets, skipped = collect_manifest_output_paths(
        manifest,
        manifest_path,
        base_dir=out_dir,
    )

    assert output_json.resolve() in targets
    assert output_csv.resolve() in targets
    assert skipped["outside_base_dir"] == 0


def test_viability_filter_ablation_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "viability_data"
    _make_small_dataset(data_dir)
    out_dir = tmp_path / "viability_out"
    viability_ablation_main(
        [
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    assert "baseline_filter_shift" in payload
    assert "phase_2" in payload["baseline_filter_shift"]
    assert (out_dir / "summary.parquet").exists()


def test_viability_filter_ablation_with_state_uniform_mode_rerun(tmp_path: Path) -> None:
    data_dir = tmp_path / "viability_data_rerun"
    _make_small_dataset(data_dir)
    out_dir = tmp_path / "viability_out_rerun"
    ablation_out = tmp_path / "viability_mode_runs"
    viability_ablation_main(
        [
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--run-state-uniform-ablation",
            "--ablation-out-dir",
            str(ablation_out),
            "--ablation-n-rules",
            "1",
            "--ablation-steps",
            "4",
            "--ablation-seed-batches",
            "1",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    ablation = payload["state_uniform_mode_ablation"]
    assert "terminal" in ablation
    assert "tag_only" in ablation


def test_viability_filter_ablation_skips_missing_phase_metrics(tmp_path: Path) -> None:
    data_dir = tmp_path / "viability_data_missing"
    _make_small_dataset(data_dir)
    (data_dir / "phase_2" / "logs" / "metrics_summary.parquet").unlink()
    out_dir = tmp_path / "viability_out_missing"
    viability_ablation_main(
        [
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    baseline = payload["baseline_filter_shift"]
    assert "phase_1" in baseline
    assert "phase_2" not in baseline


def test_viability_state_uniform_ablation_uses_explicit_seed_starts(
    monkeypatch, tmp_path: Path
) -> None:
    seen_starts: list[tuple[int, int, str]] = []

    def _fake_run_experiment(config: ExperimentConfig) -> None:
        seen_starts.append(
            (config.rule_seed_start, config.sim_seed_start, config.state_uniform_mode.value)
        )

    monkeypatch.setattr(viability_ablation_module, "run_experiment", _fake_run_experiment)
    monkeypatch.setattr(
        viability_ablation_module,
        "_summarize_dataset",
        lambda _path: {"phase_1": {"n_all": 1, "n_survived": 1}},
    )

    viability_ablation_module._run_state_uniform_mode_ablation(
        out_root=tmp_path / "ablation",
        n_rules=1,
        steps=2,
        seed_batches=1,
        rule_seed_start=123,
        sim_seed_start=456,
    )

    assert seen_starts == [
        (123, 456, "terminal"),
        (123, 456, "tag_only"),
    ]
