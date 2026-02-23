"""Build a lightweight phenotype taxonomy for high-MI Phase 2 rules.

Usage:
    uv run python scripts/phenotype_taxonomy.py \
      --data-dir data/stage_d \
      --out-dir data/post_hoc/phenotypes

    uv run python scripts/phenotype_taxonomy.py --all-survivors \
      --data-dir data/stage_d
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from objectless_alife.stats import load_final_step_metrics

# Thresholds for deterministic phenotype taxonomy.
MI_DELTA_POLARIZED = 0.12
ADJACENCY_POLARIZED = 0.60
ENTROPY_FROZEN = 0.30
PREDICTABILITY_FROZEN = 0.20
MI_DELTA_MIXED = 0.05
ENTROPY_MIXED = 0.60
PREDICTABILITY_MIXED = 0.40


def _classify_row(row: dict[str, float | str]) -> str:
    delta_mi = float(row["delta_mi"])
    entropy = float(row["state_entropy"])
    adjacency = float(row["same_state_adjacency_fraction"])
    predictability = float(row["predictability_hamming"])
    if delta_mi >= MI_DELTA_POLARIZED and adjacency >= ADJACENCY_POLARIZED:
        return "polarized_cluster"
    if entropy <= ENTROPY_FROZEN and predictability <= PREDICTABILITY_FROZEN:
        return "frozen_patch"
    if (
        delta_mi >= MI_DELTA_MIXED
        and entropy >= ENTROPY_MIXED
        and predictability >= PREDICTABILITY_MIXED
    ):
        return "mixed_turbulent"
    return "low_signal"


MAX_STEP = 199

ARCHETYPE_NAMES = ("polarized_cluster", "frozen_patch", "mixed_turbulent", "low_signal")

ALL_SURVIVORS_OUT_DIR = Path("data/post_hoc/archetype_full")


def _load_and_prepare_table(metrics_path: Path) -> pa.Table:
    """Load final-step metrics and fill missing columns with defaults."""
    table = load_final_step_metrics(metrics_path)
    if "same_state_adjacency_fraction" not in table.column_names:
        table = table.append_column(
            "same_state_adjacency_fraction",
            pa.array([0.0] * table.num_rows, type=pa.float64()),
        )
    mi_col = pc.cast(table.column("delta_mi"), pa.float64(), safe=False)
    mi_filled = pc.if_else(pc.is_valid(mi_col), mi_col, pa.scalar(0.0))
    table = table.set_column(
        table.column_names.index("delta_mi"),
        "delta_mi",
        mi_filled,
    )
    return table


def _normalize_row(row: dict) -> dict[str, float | str]:
    """Normalize a pylist row dict to safe float values."""
    return {
        "rule_id": str(row["rule_id"]),
        "state_entropy": float(row["state_entropy"]) if row["state_entropy"] is not None else 0.0,
        "predictability_hamming": (
            float(row["predictability_hamming"])
            if row["predictability_hamming"] is not None
            else 0.0
        ),
        "same_state_adjacency_fraction": (
            float(row["same_state_adjacency_fraction"])
            if row["same_state_adjacency_fraction"] is not None
            else 0.0
        ),
        "delta_mi": float(row["delta_mi"]),
    }


def run_all_survivors_analysis(metrics_path: Path, out_dir: Path | None = None) -> dict:
    """Classify all Phase 2 survivors by phenotype and compute quartile breakdown.

    Returns summary dict and saves to out_dir/summary.json.
    """
    table = _load_and_prepare_table(metrics_path)

    # Filter to survivors (step == MAX_STEP)
    step_col = table.column("step")
    survivor_mask = pc.equal(step_col, MAX_STEP)
    survivors = table.filter(survivor_mask)

    n_survivors = survivors.num_rows
    if n_survivors == 0:
        return {"n_survivors": 0}

    # Extract delta_mi values for quartile computation
    delta_mi_vals = pc.cast(survivors.column("delta_mi"), pa.float64(), safe=False).to_pylist()
    q25, q50, q75 = (
        float(np.percentile(delta_mi_vals, 25)),
        float(np.percentile(delta_mi_vals, 50)),
        float(np.percentile(delta_mi_vals, 75)),
    )

    # State entropy and predictability fraction
    entropy_vals = [
        float(v) for v in survivors.column("state_entropy").to_pylist() if v is not None
    ]
    median_state_entropy = statistics.median(entropy_vals) if entropy_vals else float("nan")
    median_delta_mi = q50
    predictability_fraction_median = (
        median_delta_mi / median_state_entropy if median_state_entropy != 0 else float("nan")
    )

    # Classify all surviving rows
    selected_cols = [
        "rule_id",
        "state_entropy",
        "predictability_hamming",
        "same_state_adjacency_fraction",
        "delta_mi",
    ]
    rows = survivors.select(selected_cols).to_pylist()

    overall_counts: dict[str, int] = {name: 0 for name in ARCHETYPE_NAMES}
    quartile_rows: dict[str, list[dict]] = {
        "q1_0_25": [],
        "q2_25_50": [],
        "q3_50_75": [],
        "q4_75_100": [],
    }

    for row in rows:
        normalized = _normalize_row(row)
        phenotype = _classify_row(normalized)
        overall_counts[phenotype] = overall_counts.get(phenotype, 0) + 1

        dmi = normalized["delta_mi"]
        if dmi <= q25:
            quartile_rows["q1_0_25"].append(normalized)
        elif dmi <= q50:
            quartile_rows["q2_25_50"].append(normalized)
        elif dmi <= q75:
            quartile_rows["q3_50_75"].append(normalized)
        else:
            quartile_rows["q4_75_100"].append(normalized)

    # Build quartile archetype counts
    delta_mi_min = min(delta_mi_vals)
    delta_mi_max = max(delta_mi_vals)
    quartile_ranges = {
        "q1_0_25": [delta_mi_min, q25],
        "q2_25_50": [q25, q50],
        "q3_50_75": [q50, q75],
        "q4_75_100": [q75, delta_mi_max],
    }

    quartile_archetype_counts: dict[str, dict] = {}
    for qkey, qrows in quartile_rows.items():
        counts: dict[str, int] = {name: 0 for name in ARCHETYPE_NAMES}
        for row in qrows:
            phenotype = _classify_row(row)
            counts[phenotype] = counts.get(phenotype, 0) + 1
        quartile_archetype_counts[qkey] = {
            "delta_mi_range": quartile_ranges[qkey],
            **counts,
        }

    summary = {
        "n_survivors": n_survivors,
        "median_state_entropy": median_state_entropy,
        "median_delta_mi": median_delta_mi,
        "predictability_fraction_median": predictability_fraction_median,
        "overall_counts": overall_counts,
        "quartile_archetype_counts": quartile_archetype_counts,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phenotype taxonomy for high-MI Phase 2 rules")
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage_d"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/phenotypes"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Run a small sanity-sized preset")
    parser.add_argument(
        "--all-survivors",
        action="store_true",
        help="Classify ALL Phase 2 survivors by quartile (ignores --top-k)",
    )
    args = parser.parse_args(argv)

    if args.all_survivors:
        metrics_path = Path(args.data_dir) / "phase_2" / "logs" / "metrics_summary.parquet"
        summary = run_all_survivors_analysis(metrics_path, out_dir=ALL_SURVIVORS_OUT_DIR)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.quick:
        args.top_k = 10

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.data_dir) / "phase_2" / "logs" / "metrics_summary.parquet"
    table = _load_and_prepare_table(metrics_path)

    rows = table.select(
        [
            "rule_id",
            "state_entropy",
            "predictability_hamming",
            "same_state_adjacency_fraction",
            "delta_mi",
        ]
    ).to_pylist()
    rows.sort(key=lambda row: float(row["delta_mi"]), reverse=True)
    top_rows = rows[: args.top_k]

    classified: list[dict[str, float | str]] = []
    for row in top_rows:
        normalized = _normalize_row(row)
        normalized["phenotype"] = _classify_row(normalized)
        classified.append(normalized)

    counts: dict[str, int] = {}
    representatives: dict[str, str] = {}
    for row in classified:
        phenotype = str(row["phenotype"])
        counts[phenotype] = counts.get(phenotype, 0) + 1
        if phenotype not in representatives:
            representatives[phenotype] = str(row["rule_id"])

    payload = {
        "top_k": args.top_k,
        "counts": counts,
        "representative_rule_ids": representatives,
        "rows": classified,
    }
    (out_dir / "taxonomy.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    with (out_dir / "taxonomy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rule_id",
                "state_entropy",
                "predictability_hamming",
                "same_state_adjacency_fraction",
                "delta_mi",
                "phenotype",
            ],
        )
        writer.writeheader()
        writer.writerows(classified)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
