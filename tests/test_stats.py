from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.run_search import METRICS_SCHEMA, PHASE_SUMMARY_METRIC_NAMES
from src.stats import (
    _holm_bonferroni,
    bootstrap_median_ci,
    filter_metric_independence,
    load_final_step_metrics,
    pairwise_metric_comparison,
    pairwise_survival_comparison,
    phase_comparison_tests,
    run_statistical_analysis,
    save_results,
    survival_rate_test,
    wilson_score_ci,
)
from src.stats import main as stats_main


def _write_metrics_parquet(path: Path, rows: list[dict]) -> None:
    """Helper to write a metrics_summary.parquet with the canonical schema."""
    table = pa.Table.from_pylist(rows, schema=METRICS_SCHEMA)
    pq.write_table(table, path)


def _make_metric_row(
    rule_id: str,
    step: int,
    *,
    neighbor_mi: float = 0.1,
    state_entropy: float = 1.0,
    mi_shuffle_null: float = 0.05,
) -> dict:
    """Build a single metrics row with sensible defaults."""
    return {
        "rule_id": rule_id,
        "step": step,
        "state_entropy": state_entropy,
        "compression_ratio": 0.5,
        "predictability_hamming": 0.1 if step > 0 else None,
        "morans_i": 0.0,
        "cluster_count": 10,
        "quasi_periodicity_peaks": 0,
        "phase_transition_max_delta": 0.01,
        "neighbor_mutual_information": neighbor_mi,
        "action_entropy_mean": 0.5,
        "action_entropy_variance": 0.01,
        "block_ncd": 0.3 if step >= 20 else None,
        "mi_shuffle_null": mi_shuffle_null,
    }


class TestLoadFinalStepMetrics:
    def test_returns_one_row_per_rule(self, tmp_path: Path) -> None:
        rows = [
            _make_metric_row("rule_a", 0),
            _make_metric_row("rule_a", 1),
            _make_metric_row("rule_a", 2),
            _make_metric_row("rule_b", 0),
            _make_metric_row("rule_b", 1),
        ]
        path = tmp_path / "metrics_summary.parquet"
        _write_metrics_parquet(path, rows)

        result = load_final_step_metrics(path)
        assert result.num_rows == 2

    def test_selects_max_step_per_rule(self, tmp_path: Path) -> None:
        rows = [
            _make_metric_row("rule_a", 0, neighbor_mi=0.1),
            _make_metric_row("rule_a", 5, neighbor_mi=0.9),
            _make_metric_row("rule_a", 3, neighbor_mi=0.5),
        ]
        path = tmp_path / "metrics_summary.parquet"
        _write_metrics_parquet(path, rows)

        result = load_final_step_metrics(path)
        assert result.num_rows == 1
        nmi_values = result.column("neighbor_mutual_information").to_pylist()
        assert nmi_values[0] == pytest.approx(0.9)


class TestPhaseComparisonTests:
    def _make_table(self, values: list[float], metric: str = "neighbor_mutual_information"):
        """Build a single-column pyarrow table for testing."""
        data = {m: [None] * len(values) for m in PHASE_SUMMARY_METRIC_NAMES}
        data[metric] = values
        return pa.table(data)

    def test_returns_dict_with_expected_keys(self) -> None:
        p1 = self._make_table([0.1, 0.2, 0.3])
        p2 = self._make_table([0.4, 0.5, 0.6])
        result = phase_comparison_tests(p1, p2, ["neighbor_mutual_information"])
        assert "neighbor_mutual_information" in result

    def test_result_contains_required_fields(self) -> None:
        p1 = self._make_table([0.1, 0.2, 0.3])
        p2 = self._make_table([0.4, 0.5, 0.6])
        result = phase_comparison_tests(p1, p2, ["neighbor_mutual_information"])
        entry = result["neighbor_mutual_information"]
        assert "u_statistic" in entry
        assert "p_value" in entry
        assert "p_value_corrected" in entry
        assert "effect_size_r" in entry
        assert "n_phase1" in entry
        assert "n_phase2" in entry
        assert "phase1_median" in entry
        assert "phase2_median" in entry

    def test_p_value_in_valid_range(self) -> None:
        p1 = self._make_table([0.1, 0.2, 0.3, 0.15, 0.25])
        p2 = self._make_table([0.4, 0.5, 0.6, 0.45, 0.55])
        result = phase_comparison_tests(p1, p2, ["neighbor_mutual_information"])
        entry = result["neighbor_mutual_information"]
        assert 0.0 <= entry["p_value"] <= 1.0

    def test_effect_size_in_valid_range(self) -> None:
        p1 = self._make_table([0.1, 0.2, 0.3, 0.15, 0.25])
        p2 = self._make_table([0.4, 0.5, 0.6, 0.45, 0.55])
        result = phase_comparison_tests(p1, p2, ["neighbor_mutual_information"])
        entry = result["neighbor_mutual_information"]
        assert -1.0 <= entry["effect_size_r"] <= 1.0

    def test_clearly_different_distributions_yield_small_p(self) -> None:
        p1 = self._make_table([float(x) for x in range(100)])
        p2 = self._make_table([float(x + 1000) for x in range(100)])
        result = phase_comparison_tests(p1, p2, ["neighbor_mutual_information"])
        assert result["neighbor_mutual_information"]["p_value"] < 0.05

    def test_identical_distributions_yield_large_p(self) -> None:
        values = [float(x) for x in range(50)]
        p1 = self._make_table(values)
        p2 = self._make_table(values)
        result = phase_comparison_tests(p1, p2, ["neighbor_mutual_information"])
        assert result["neighbor_mutual_information"]["p_value"] > 0.05

    def test_skips_metrics_with_all_nulls(self) -> None:
        p1 = self._make_table([None] * 10, metric="block_ncd")
        p2 = self._make_table([None] * 10, metric="block_ncd")
        result = phase_comparison_tests(p1, p2, ["block_ncd"])
        assert "block_ncd" not in result

    def test_multiple_metrics(self) -> None:
        data1 = {m: [0.1, 0.2, 0.3] for m in PHASE_SUMMARY_METRIC_NAMES}
        data2 = {m: [0.4, 0.5, 0.6] for m in PHASE_SUMMARY_METRIC_NAMES}
        p1 = pa.table(data1)
        p2 = pa.table(data2)
        result = phase_comparison_tests(p1, p2, PHASE_SUMMARY_METRIC_NAMES)
        assert len(result) == len(PHASE_SUMMARY_METRIC_NAMES)


class TestHolmBonferroni:
    def test_single_p_value_unchanged(self) -> None:
        assert _holm_bonferroni([0.03]) == [0.03]

    def test_corrected_values_are_at_least_as_large_as_raw(self) -> None:
        raw = [0.01, 0.04, 0.03]
        corrected = _holm_bonferroni(raw)
        for r, c in zip(raw, corrected, strict=True):
            assert c >= r

    def test_corrected_values_capped_at_one(self) -> None:
        raw = [0.5, 0.6, 0.7]
        corrected = _holm_bonferroni(raw)
        for c in corrected:
            assert c <= 1.0

    def test_known_correction(self) -> None:
        # 3 tests: sorted p-values are 0.01, 0.03, 0.04
        # Holm: 0.01*3=0.03, max(0.03, 0.03*2)=0.06, max(0.06, 0.04*1)=0.06
        raw = [0.01, 0.04, 0.03]
        corrected = _holm_bonferroni(raw)
        assert corrected[0] == pytest.approx(0.03)  # 0.01 * 3
        assert corrected[2] == pytest.approx(0.06)  # max(0.03, 0.03*2)
        assert corrected[1] == pytest.approx(0.06)  # max(0.06, 0.04*1)

    def test_empty_returns_empty(self) -> None:
        assert _holm_bonferroni([]) == []


class TestSurvivalRateTest:
    def test_returns_expected_fields(self) -> None:
        runs = pa.table(
            {
                "phase": [1, 1, 1, 2, 2, 2],
                "survived": [True, True, False, True, False, False],
            }
        )
        result = survival_rate_test(runs)
        assert "chi2" in result
        assert "p_value" in result
        assert "phase1_survived" in result
        assert "phase1_total" in result
        assert "phase2_survived" in result
        assert "phase2_total" in result

    def test_p_value_in_valid_range(self) -> None:
        runs = pa.table(
            {
                "phase": [1] * 100 + [2] * 100,
                "survived": [True] * 80 + [False] * 20 + [True] * 30 + [False] * 70,
            }
        )
        result = survival_rate_test(runs)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_rejects_single_phase(self) -> None:
        runs = pa.table(
            {
                "phase": [1, 1, 1],
                "survived": [True, True, False],
            }
        )
        with pytest.raises(ValueError, match="Expected exactly 2 phases"):
            survival_rate_test(runs)

    def test_very_different_rates_yield_small_p(self) -> None:
        runs = pa.table(
            {
                "phase": [1] * 200 + [2] * 200,
                "survived": [True] * 190 + [False] * 10 + [True] * 50 + [False] * 150,
            }
        )
        result = survival_rate_test(runs)
        assert result["p_value"] < 0.05


class TestSaveResults:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        results = {
            "generated_at": "2026-01-01T00:00:00",
            "metric_tests": {},
            "survival_test": {},
        }
        out = tmp_path / "statistical_tests.json"
        save_results(results, out)
        loaded = json.loads(out.read_text())
        assert loaded["generated_at"] == "2026-01-01T00:00:00"


class TestRunStatisticalAnalysis:
    def test_end_to_end_with_synthetic_data(self, tmp_path: Path) -> None:
        """Build a minimal two-phase dataset and verify full pipeline."""
        # Create directory structure matching run_experiment output
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        p1_logs = tmp_path / "phase_1" / "logs"
        p2_logs = tmp_path / "phase_2" / "logs"
        p1_logs.mkdir(parents=True)
        p2_logs.mkdir(parents=True)

        # experiment_runs.parquet
        n = 20
        runs_rows = []
        for i in range(n):
            runs_rows.append(
                {
                    "schema_version": 1,
                    "rule_id": f"phase1_rs{i}_ss{i}",
                    "phase": 1,
                    "seed_batch": 0,
                    "rule_seed": i,
                    "sim_seed": i,
                    "survived": i < 18,
                    "termination_reason": None if i < 18 else "halt",
                    "terminated_at": None if i < 18 else 100,
                }
            )
            runs_rows.append(
                {
                    "schema_version": 1,
                    "rule_id": f"phase2_rs{i}_ss{i}",
                    "phase": 2,
                    "seed_batch": 0,
                    "rule_seed": i,
                    "sim_seed": i,
                    "survived": i < 15,
                    "termination_reason": None if i < 15 else "halt",
                    "terminated_at": None if i < 15 else 50,
                }
            )
        pq.write_table(pa.Table.from_pylist(runs_rows), logs_dir / "experiment_runs.parquet")

        # phase_1 metrics_summary.parquet (low neighbor MI)
        p1_metric_rows = []
        for i in range(n):
            rule_id = f"phase1_rs{i}_ss{i}"
            max_step = 100 if i >= 18 else 199
            for step in range(max_step + 1):
                p1_metric_rows.append(_make_metric_row(rule_id, step, neighbor_mi=0.1 + i * 0.005))
        _write_metrics_parquet(p1_logs / "metrics_summary.parquet", p1_metric_rows)

        # phase_2 metrics_summary.parquet (high neighbor MI)
        p2_metric_rows = []
        for i in range(n):
            rule_id = f"phase2_rs{i}_ss{i}"
            max_step = 50 if i >= 15 else 199
            for step in range(max_step + 1):
                p2_metric_rows.append(_make_metric_row(rule_id, step, neighbor_mi=0.5 + i * 0.01))
        _write_metrics_parquet(p2_logs / "metrics_summary.parquet", p2_metric_rows)

        result = run_statistical_analysis(tmp_path)

        assert "generated_at" in result
        assert "metric_tests" in result
        assert "survival_test" in result
        assert "neighbor_mutual_information" in result["metric_tests"]

        nmi = result["metric_tests"]["neighbor_mutual_information"]
        assert nmi["p_value"] < 0.05
        assert nmi["n_phase1"] == n
        assert nmi["n_phase2"] == n

    def test_saves_json_output(self, tmp_path: Path) -> None:
        """Verify run_statistical_analysis with save produces valid JSON."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        p1_logs = tmp_path / "phase_1" / "logs"
        p2_logs = tmp_path / "phase_2" / "logs"
        p1_logs.mkdir(parents=True)
        p2_logs.mkdir(parents=True)

        n = 10
        runs_rows = []
        for i in range(n):
            for phase in [1, 2]:
                runs_rows.append(
                    {
                        "schema_version": 1,
                        "rule_id": f"phase{phase}_rs{i}_ss{i}",
                        "phase": phase,
                        "seed_batch": 0,
                        "rule_seed": i,
                        "sim_seed": i,
                        "survived": True,
                        "termination_reason": None,
                        "terminated_at": None,
                    }
                )
        pq.write_table(pa.Table.from_pylist(runs_rows), logs_dir / "experiment_runs.parquet")

        for phase, phase_dir, mi_base in [(1, p1_logs, 0.1), (2, p2_logs, 0.5)]:
            rows = []
            for i in range(n):
                rule_id = f"phase{phase}_rs{i}_ss{i}"
                for step in range(5):
                    rows.append(_make_metric_row(rule_id, step, neighbor_mi=mi_base + i * 0.01))
            _write_metrics_parquet(phase_dir / "metrics_summary.parquet", rows)

        result = run_statistical_analysis(tmp_path)
        out_path = logs_dir / "statistical_tests.json"
        save_results(result, out_path)

        loaded = json.loads(out_path.read_text())
        assert "metric_tests" in loaded
        assert "survival_test" in loaded


class TestPairwiseMetricComparison:
    def test_returns_results_with_expected_keys(self, tmp_path: Path) -> None:
        path_a = tmp_path / "metrics_a.parquet"
        path_b = tmp_path / "metrics_b.parquet"
        rows_a = [
            _make_metric_row(f"rule_{i}", step=5, neighbor_mi=0.1 + i * 0.01) for i in range(20)
        ]
        rows_b = [
            _make_metric_row(f"rule_{i}", step=5, neighbor_mi=0.5 + i * 0.01) for i in range(20)
        ]
        _write_metrics_parquet(path_a, rows_a)
        _write_metrics_parquet(path_b, rows_b)

        result = pairwise_metric_comparison(path_a, path_b, ["neighbor_mutual_information"])
        assert "neighbor_mutual_information" in result
        entry = result["neighbor_mutual_information"]
        assert "u_statistic" in entry
        assert "p_value" in entry
        assert "p_value_corrected" in entry
        assert "effect_size_r" in entry

    def test_clearly_different_distributions_yield_small_p(self, tmp_path: Path) -> None:
        path_a = tmp_path / "metrics_a.parquet"
        path_b = tmp_path / "metrics_b.parquet"
        rows_a = [_make_metric_row(f"rule_{i}", step=5, neighbor_mi=float(i)) for i in range(50)]
        rows_b = [
            _make_metric_row(f"rule_{i}", step=5, neighbor_mi=float(i + 1000)) for i in range(50)
        ]
        _write_metrics_parquet(path_a, rows_a)
        _write_metrics_parquet(path_b, rows_b)

        result = pairwise_metric_comparison(path_a, path_b, ["neighbor_mutual_information"])
        assert result["neighbor_mutual_information"]["p_value"] < 0.05


class TestPairwiseSurvivalComparison:
    def _write_rule_json(self, rules_dir: Path, rule_id: str, survived: bool) -> None:
        payload = {
            "rule_id": rule_id,
            "table": [0] * 20,
            "survived": survived,
            "filter_results": {},
            "metadata": {},
        }
        (rules_dir / f"{rule_id}.json").write_text(json.dumps(payload))

    def test_returns_expected_fields(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a" / "rules"
        dir_b = tmp_path / "b" / "rules"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        for i in range(10):
            self._write_rule_json(dir_a, f"rule_a_{i}", survived=i < 8)
            self._write_rule_json(dir_b, f"rule_b_{i}", survived=i < 4)

        result = pairwise_survival_comparison(dir_a, dir_b)
        assert "chi2" in result
        assert "p_value" in result
        assert "a_survived" in result
        assert "a_total" in result
        assert "b_survived" in result
        assert "b_total" in result

    def test_very_different_rates_yield_small_p(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a" / "rules"
        dir_b = tmp_path / "b" / "rules"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        for i in range(100):
            self._write_rule_json(dir_a, f"rule_a_{i}", survived=i < 90)
            self._write_rule_json(dir_b, f"rule_b_{i}", survived=i < 30)

        result = pairwise_survival_comparison(dir_a, dir_b)
        assert result["p_value"] < 0.05

    def test_all_survived_returns_nan(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a" / "rules"
        dir_b = tmp_path / "b" / "rules"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        for i in range(5):
            self._write_rule_json(dir_a, f"rule_a_{i}", survived=True)
            self._write_rule_json(dir_b, f"rule_b_{i}", survived=True)

        result = pairwise_survival_comparison(dir_a, dir_b)
        assert math.isnan(result["chi2"])
        assert math.isnan(result["p_value"])

    def test_empty_directory_returns_nan(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a" / "rules"
        dir_b = tmp_path / "b" / "rules"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        # dir_a is empty, dir_b has rules
        for i in range(5):
            self._write_rule_json(dir_b, f"rule_b_{i}", survived=True)

        result = pairwise_survival_comparison(dir_a, dir_b)
        assert result["a_total"] == 0
        assert math.isnan(result["chi2"])
        assert math.isnan(result["p_value"])


class TestBootstrapMedianCi:
    def test_contains_true_median_diff(self) -> None:
        """CI should cover the known median difference for clearly separated groups."""
        rng = random.Random(42)
        vals1 = [float(i) for i in range(100)]
        vals2 = [float(i + 50) for i in range(100)]
        lo, hi = bootstrap_median_ci(vals1, vals2, n_bootstrap=5000, rng=rng)
        true_diff = 50.0  # median(vals2) - median(vals1)
        assert lo <= true_diff <= hi

    def test_deterministic_with_seed(self) -> None:
        vals1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        vals2 = [6.0, 7.0, 8.0, 9.0, 10.0]
        lo1, hi1 = bootstrap_median_ci(vals1, vals2, n_bootstrap=1000, rng=random.Random(7))
        lo2, hi2 = bootstrap_median_ci(vals1, vals2, n_bootstrap=1000, rng=random.Random(7))
        assert lo1 == lo2
        assert hi1 == hi2

    def test_empty_inputs_return_nan(self) -> None:
        lo, hi = bootstrap_median_ci([], [1.0, 2.0])
        assert math.isnan(lo) and math.isnan(hi)
        lo, hi = bootstrap_median_ci([1.0, 2.0], [])
        assert math.isnan(lo) and math.isnan(hi)

    def test_single_element_inputs(self) -> None:
        lo, hi = bootstrap_median_ci([1.0], [5.0], n_bootstrap=1000, rng=random.Random(0))
        assert lo == hi == 4.0  # only one possible resample

    def test_narrows_with_more_samples(self) -> None:
        """More samples from the same distribution → narrower CI."""
        gen = random.Random(0)
        base = [gen.gauss(0, 1) for _ in range(500)]
        small = base[:20]
        large = base[:200]
        lo_s, hi_s = bootstrap_median_ci(
            small, [x + 5 for x in small], n_bootstrap=2000, rng=random.Random(1)
        )
        lo_l, hi_l = bootstrap_median_ci(
            large, [x + 5 for x in large], n_bootstrap=2000, rng=random.Random(1)
        )
        width_small = hi_s - lo_s
        width_large = hi_l - lo_l
        assert width_large < width_small


class TestPhaseComparisonCiAndCliffsDelta:
    def _make_table(self, values: list[float], metric: str = "neighbor_mutual_information"):
        data = {m: [None] * len(values) for m in PHASE_SUMMARY_METRIC_NAMES}
        data[metric] = values
        return pa.table(data)

    def test_phase_comparison_includes_ci_and_cliffs_delta(self) -> None:
        """New fields present in phase_comparison_tests output."""
        p1 = self._make_table([0.1, 0.2, 0.3, 0.15, 0.25])
        p2 = self._make_table([0.4, 0.5, 0.6, 0.45, 0.55])
        result = phase_comparison_tests(p1, p2, ["neighbor_mutual_information"])
        entry = result["neighbor_mutual_information"]
        assert "cliffs_delta" in entry
        assert "median_diff_ci_lower" in entry
        assert "median_diff_ci_upper" in entry
        assert entry["median_diff_ci_lower"] <= entry["median_diff_ci_upper"]


class TestStatsMainPairwise:
    def _setup_dir(self, base: Path, label: str, n: int, mi_base: float, surv_frac: float) -> Path:
        """Create a minimal data directory with metrics parquet and rule JSONs."""
        d = base / label
        logs = d / "logs"
        rules = d / "rules"
        logs.mkdir(parents=True)
        rules.mkdir(parents=True)
        rows = [
            _make_metric_row(f"rule_{i}", step=5, neighbor_mi=mi_base + i * 0.01) for i in range(n)
        ]
        _write_metrics_parquet(logs / "metrics_summary.parquet", rows)
        for i in range(n):
            payload = {
                "rule_id": f"rule_{i}",
                "table": [0] * 20,
                "survived": i < int(n * surv_frac),
                "filter_results": {},
                "metadata": {},
            }
            (rules / f"rule_{i}.json").write_text(json.dumps(payload))
        return d

    def test_pairwise_cli_produces_output_json(self, tmp_path: Path) -> None:
        dir_a = self._setup_dir(tmp_path, "a", 20, 0.1, 0.8)
        dir_b = self._setup_dir(tmp_path, "b", 20, 0.5, 0.4)
        out = tmp_path / "result.json"
        stats_main(
            [
                "--pairwise",
                "--dir-a",
                str(dir_a),
                "--dir-b",
                str(dir_b),
                "--output",
                str(out),
            ]
        )
        loaded = json.loads(out.read_text())
        assert "metric_tests" in loaded
        assert "survival_test" in loaded
        assert loaded["label_a"] == "a"
        assert loaded["label_b"] == "b"


class TestWilsonScoreCi:
    def test_known_values(self) -> None:
        """Wilson CI for 50/100 at 95% should be roughly (0.40, 0.60)."""
        lo, hi = wilson_score_ci(50, 100)
        assert 0.39 < lo < 0.42
        assert 0.58 < hi < 0.61

    def test_zero_successes(self) -> None:
        """0/100: lower bound is 0, upper bound positive."""
        lo, hi = wilson_score_ci(0, 100)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert hi > 0.0

    def test_all_successes(self) -> None:
        """100/100: lower bound < 1, upper bound is 1."""
        lo, hi = wilson_score_ci(100, 100)
        assert lo < 1.0
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_bounds_ordered(self) -> None:
        lo, hi = wilson_score_ci(30, 200)
        assert lo < hi

    def test_bounds_in_unit_interval(self) -> None:
        lo, hi = wilson_score_ci(7, 10)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_narrows_with_more_observations(self) -> None:
        lo1, hi1 = wilson_score_ci(5, 10)
        lo2, hi2 = wilson_score_ci(50, 100)
        assert (hi2 - lo2) < (hi1 - lo1)

    def test_zero_total_returns_nan(self) -> None:
        lo, hi = wilson_score_ci(0, 0)
        assert math.isnan(lo) and math.isnan(hi)

    def test_successes_greater_than_total_raises(self) -> None:
        with pytest.raises(ValueError):
            wilson_score_ci(10, 5)

    def test_negative_successes_raises(self) -> None:
        with pytest.raises(ValueError):
            wilson_score_ci(-1, 10)

    def test_negative_total_raises(self) -> None:
        with pytest.raises(ValueError):
            wilson_score_ci(0, -1)


class TestFilterMetricIndependence:
    def _setup_data(
        self,
        tmp_path: Path,
        n_rules: int,
        mi_survived: float,
        mi_terminated: float,
        surv_frac: float,
    ) -> tuple[Path, Path]:
        """Create metrics parquet and rule JSONs with controlled MI/survival."""
        logs = tmp_path / "logs"
        rules = tmp_path / "rules"
        logs.mkdir(parents=True)
        rules.mkdir(parents=True)

        rows = []
        for i in range(n_rules):
            survived = i < int(n_rules * surv_frac)
            mi = mi_survived if survived else mi_terminated
            rows.append(_make_metric_row(f"rule_{i}", step=5, neighbor_mi=mi))
            payload = {
                "rule_id": f"rule_{i}",
                "table": [0] * 20,
                "survived": survived,
                "filter_results": {},
                "metadata": {},
            }
            (rules / f"rule_{i}.json").write_text(json.dumps(payload))

        _write_metrics_parquet(logs / "metrics_summary.parquet", rows)
        return logs / "metrics_summary.parquet", rules

    def test_returns_expected_fields(self, tmp_path: Path) -> None:
        metrics_path, rules_dir = self._setup_data(tmp_path, 20, 0.5, 0.1, 0.5)
        result = filter_metric_independence(metrics_path, rules_dir)
        assert "correlation" in result
        assert "p_value" in result
        assert "survived_median_mi" in result
        assert "terminated_median_mi" in result

    def test_high_correlation_with_correlated_data(self, tmp_path: Path) -> None:
        """When survived rules have high MI and terminated have low, correlation is strong."""
        metrics_path, rules_dir = self._setup_data(tmp_path, 100, 0.9, 0.01, 0.5)
        result = filter_metric_independence(metrics_path, rules_dir)
        assert abs(result["correlation"]) > 0.5
        assert result["p_value"] < 0.05

    def test_low_correlation_with_independent_data(self, tmp_path: Path) -> None:
        """When MI is unrelated to survival, correlation is near zero."""
        # Use varied MI values that are independent of survival status
        logs = tmp_path / "indep" / "logs"
        rules = tmp_path / "indep" / "rules"
        logs.mkdir(parents=True)
        rules.mkdir(parents=True)

        rng = random.Random(42)
        n = 100
        rows = []
        for i in range(n):
            mi = rng.uniform(0.0, 1.0)
            survived = i % 2 == 0  # alternating, uncorrelated with MI
            rows.append(_make_metric_row(f"rule_{i}", step=5, neighbor_mi=mi))
            payload = {
                "rule_id": f"rule_{i}",
                "table": [0] * 20,
                "survived": survived,
                "filter_results": {},
                "metadata": {},
            }
            (rules / f"rule_{i}.json").write_text(json.dumps(payload))

        _write_metrics_parquet(logs / "metrics_summary.parquet", rows)
        result = filter_metric_independence(logs / "metrics_summary.parquet", rules)
        assert abs(result["correlation"]) < 0.3
