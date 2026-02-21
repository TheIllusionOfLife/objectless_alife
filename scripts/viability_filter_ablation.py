"""Analyze viability-filter confounds for PR26-style experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from objectless_alife.config import ExperimentConfig, StateUniformMode
from objectless_alife.rules import ObservationPhase
from objectless_alife.run_search import run_experiment
from objectless_alife.stats import load_final_step_metrics

PHASE_LABELS = {
    1: "phase_1",
    2: "phase_2",
    3: "control",
}


def _delta_mi_from_table(table: pa.Table) -> dict[str, float]:
    excess = pc.cast(table.column("delta_mi"), pa.float64(), safe=False)
    return {
        str(rule_id): float(value)
        for rule_id, value in zip(
            table.column("rule_id").to_pylist(), excess.to_pylist(), strict=True
        )
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    sorted_vals = sorted(values)

    def _pick(q: float) -> float:
        # Use nearest-lower index truncation (no interpolation); this is a quick,
        # deterministic summary for exploratory comparison.
        idx = int((len(sorted_vals) - 1) * q)
        return float(sorted_vals[idx])

    return {"median": _pick(0.5), "p25": _pick(0.25), "p75": _pick(0.75)}


def _phase_survival_map(runs_path: Path, phase: int) -> dict[str, bool]:
    rows = pq.read_table(
        runs_path,
        filters=[("phase", "=", phase)],
        columns=["rule_id", "survived"],
    ).to_pylist()
    return {str(row["rule_id"]): bool(row["survived"]) for row in rows}


def _summarize_dataset(data_dir: Path) -> dict[str, dict[str, float | int]]:
    runs_path = data_dir / "logs" / "experiment_runs.parquet"
    summary: dict[str, dict[str, float | int]] = {}
    for phase, label in PHASE_LABELS.items():
        metrics_path = data_dir / f"phase_{phase}" / "logs" / "metrics_summary.parquet"
        if not metrics_path.exists():
            print(
                f"Skipping {label}: missing metrics file at {metrics_path}",
                file=sys.stderr,
            )
            continue
        table = load_final_step_metrics(metrics_path)
        delta_mi_map = _delta_mi_from_table(table)
        surv_map = _phase_survival_map(runs_path, phase)
        all_vals = list(delta_mi_map.values())
        survived_vals = [value for rid, value in delta_mi_map.items() if surv_map.get(rid, False)]
        all_q = _quantiles(all_vals)
        surv_q = _quantiles(survived_vals)
        summary[label] = {
            "n_all": len(all_vals),
            "n_survived": len(survived_vals),
            "all_median": all_q["median"],
            "all_p25": all_q["p25"],
            "all_p75": all_q["p75"],
            "survived_median": surv_q["median"],
            "survived_p25": surv_q["p25"],
            "survived_p75": surv_q["p75"],
            "median_shift_survived_minus_all": float(surv_q["median"] - all_q["median"]),
        }
    return summary


def _run_state_uniform_mode_ablation(
    out_root: Path,
    n_rules: int,
    steps: int,
    seed_batches: int,
    rule_seed_start: int,
    sim_seed_start: int,
) -> dict[str, object]:
    results: dict[str, object] = {}
    phases = (
        ObservationPhase.PHASE1_DENSITY,
        ObservationPhase.PHASE2_PROFILE,
        ObservationPhase.CONTROL_DENSITY_CLOCK,
    )
    for mode in (StateUniformMode.TERMINAL, StateUniformMode.TAG_ONLY):
        mode_dir = out_root / f"state_uniform_{mode.value}"
        run_experiment(
            ExperimentConfig(
                phases=phases,
                n_rules=n_rules,
                n_seed_batches=seed_batches,
                out_dir=mode_dir,
                steps=steps,
                rule_seed_start=rule_seed_start,
                sim_seed_start=sim_seed_start,
                state_uniform_mode=mode,
            )
        )
        mode_summary = _summarize_dataset(mode_dir)
        for _phase_label, phase_stats in mode_summary.items():
            n_all = int(phase_stats["n_all"])
            n_survived = int(phase_stats["n_survived"])
            phase_stats["survival_rate"] = (n_survived / n_all) if n_all else float("nan")
        results[mode.value] = mode_summary
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze viability-filter confounds")
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage_d"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/viability_ablation"))
    parser.add_argument(
        "--run-state-uniform-ablation",
        action="store_true",
        help="Run terminal vs tag_only state-uniform experiment ablation",
    )
    parser.add_argument(
        "--ablation-out-dir",
        type=Path,
        default=Path("data/post_hoc/viability_mode_ablation"),
    )
    parser.add_argument("--ablation-n-rules", type=int, default=300)
    parser.add_argument("--ablation-steps", type=int, default=200)
    parser.add_argument("--ablation-seed-batches", type=int, default=1)
    parser.add_argument("--ablation-rule-seed-start", type=int, default=0)
    parser.add_argument("--ablation-sim-seed-start", type=int, default=0)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = _summarize_dataset(args.data_dir)
    payload: dict[str, object] = {
        "data_dir": str(args.data_dir),
        "baseline_filter_shift": baseline_summary,
    }

    if args.run_state_uniform_ablation:
        payload["state_uniform_mode_ablation"] = _run_state_uniform_mode_ablation(
            out_root=args.ablation_out_dir,
            n_rules=args.ablation_n_rules,
            steps=args.ablation_steps,
            seed_batches=args.ablation_seed_batches,
            rule_seed_start=args.ablation_rule_seed_start,
            sim_seed_start=args.ablation_sim_seed_start,
        )

    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    rows: list[dict[str, object]] = []
    for phase_label, stats in baseline_summary.items():
        rows.append({"phase": phase_label, **stats})
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_dir / "summary.parquet")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
