"""Phase summary and comparison builders for experiment aggregation.

Functions here build the per-phase metric summaries and cross-phase
comparisons that are persisted as Parquet/JSON artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from objectless_alife.config.types import SimulationResult
from objectless_alife.domain.rules import ObservationPhase
from objectless_alife.io.schemas import AGGREGATE_SCHEMA_VERSION, PHASE_SUMMARY_METRIC_NAMES


def _percentile_pre_sorted(sorted_values: list[float], q: float) -> float | None:
    """Compute percentile in [0, 1] with linear interpolation on pre-sorted values."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    fraction = pos - lo
    return sorted_values[lo] * (1.0 - fraction) + sorted_values[hi] * fraction


def _to_float_list(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        numeric = float(value)
        if numeric != numeric:
            continue
        values.append(numeric)
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _build_phase_summary(
    phase: ObservationPhase,
    run_rows: list[dict[str, Any]],
    final_metric_rows: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    """Build a per-phase metric summary from run rows and final-step metrics."""
    rules_evaluated = len(run_rows)
    survived_count = sum(1 for row in run_rows if bool(row["survived"]))
    terminated_at_values = [
        int(row["terminated_at"]) for row in run_rows if row.get("terminated_at") is not None
    ]

    summary: dict[str, int | float | None] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "phase": phase.value,
        "rules_evaluated": rules_evaluated,
        "survival_rate": (survived_count / rules_evaluated) if rules_evaluated else 0.0,
        "termination_rate": ((rules_evaluated - survived_count) / rules_evaluated)
        if rules_evaluated
        else 0.0,
        "mean_terminated_at": _mean([float(v) for v in terminated_at_values]),
    }

    # Derive delta_mi per rule before summarizing (avoid mutating caller's list)
    enriched_rows = []
    for row in final_metric_rows:
        new_row = row.copy()
        mi = new_row.get("neighbor_mutual_information")
        null = new_row.get("mi_shuffle_null")
        new_row["delta_mi"] = (
            float(mi) - float(null)
            if mi is not None and null is not None and mi == mi and null == null
            else None
        )
        enriched_rows.append(new_row)

    for metric_name in PHASE_SUMMARY_METRIC_NAMES:
        values = sorted(_to_float_list(enriched_rows, metric_name))
        summary[f"{metric_name}_mean"] = _mean(values)
        summary[f"{metric_name}_p25"] = _percentile_pre_sorted(values, 0.25)
        summary[f"{metric_name}_p50"] = _percentile_pre_sorted(values, 0.50)
        summary[f"{metric_name}_p75"] = _percentile_pre_sorted(values, 0.75)

    return summary


def _build_phase_comparison(phase_summaries: list[dict[str, int | float | None]]) -> dict[str, Any]:
    """Build cross-phase comparison payload from a list of phase summary rows."""

    def _phase_value(row: dict[str, int | float | None]) -> int:
        phase_value = row.get("phase")
        if not isinstance(phase_value, int):
            raise ValueError("phase summary row missing integer 'phase'")
        return phase_value

    sorted_rows = sorted(phase_summaries, key=_phase_value)
    payload: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "phases": [_phase_value(row) for row in sorted_rows],
        "deltas": {},
        "deltas_base_phase": None,
        "deltas_target_phase": None,
        "pairwise_deltas": [],
    }
    if len(sorted_rows) < 2:
        return payload

    def _row_delta(
        base: dict[str, int | float | None],
        target: dict[str, int | float | None],
    ) -> dict[str, dict[str, float | None]]:
        deltas: dict[str, dict[str, float | None]] = {}
        for key, target_value in target.items():
            if key in {"phase", "schema_version"}:
                continue
            base_value = base.get(key)
            if not isinstance(base_value, (int, float)) or not isinstance(
                target_value, (int, float)
            ):
                continue
            delta_abs = float(target_value) - float(base_value)
            delta_rel = None if float(base_value) == 0.0 else delta_abs / float(base_value)
            deltas[key] = {"absolute": delta_abs, "relative": delta_rel}
        return deltas

    for i in range(len(sorted_rows)):
        for j in range(i + 1, len(sorted_rows)):
            base = sorted_rows[i]
            target = sorted_rows[j]
            payload["pairwise_deltas"].append(
                {
                    "base_phase": _phase_value(base),
                    "target_phase": _phase_value(target),
                    "deltas": _row_delta(base, target),
                }
            )

    # Backward-compatible primary delta payload.
    if len(sorted_rows) >= 2:
        payload["deltas_base_phase"] = payload["pairwise_deltas"][0]["base_phase"]
        payload["deltas_target_phase"] = payload["pairwise_deltas"][0]["target_phase"]
        payload["deltas"] = payload["pairwise_deltas"][0]["deltas"]

    return payload


def collect_final_metric_rows(
    metrics_path: Path,
    metric_columns: list[str],
    phase_results: list[SimulationResult],
    default_final_step: int,
) -> list[dict[str, Any]]:
    """Collect final-step metric rows per rule from parquet in batches."""
    final_steps = {
        result.rule_id: (
            result.terminated_at if result.terminated_at is not None else default_final_step
        )
        for result in phase_results
    }
    final_rows: list[dict[str, Any]] = []
    metrics_file = pq.ParquetFile(metrics_path)
    available_columns = set(metrics_file.schema_arrow.names)
    for required_col in ("rule_id", "step"):
        if required_col not in available_columns:
            raise ValueError(f"metrics parquet missing required column: {required_col}")
    required = ["rule_id", "step"]
    present_columns = required + [
        col for col in metric_columns if col in available_columns and col not in {"rule_id", "step"}
    ]

    for batch in metrics_file.iter_batches(columns=present_columns, batch_size=8192):
        batch_dict = batch.to_pydict()
        rule_ids = batch_dict["rule_id"]
        steps = batch_dict["step"]
        for idx, rule_id in enumerate(rule_ids):
            expected_step = final_steps.get(str(rule_id))
            if expected_step is None or int(steps[idx]) != expected_step:
                continue
            row: dict[str, Any] = {}
            for name in metric_columns:
                if name in batch_dict:
                    row[name] = batch_dict[name][idx]
                else:
                    row[name] = None
            final_rows.append(row)
    return final_rows
