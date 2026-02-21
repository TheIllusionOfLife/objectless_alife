"""Build a lightweight phenotype taxonomy for high-MI Phase 2 rules.

Usage:
    uv run python scripts/phenotype_taxonomy.py \
      --data-dir data/stage_d \
      --out-dir data/post_hoc/phenotypes
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phenotype taxonomy for high-MI Phase 2 rules")
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage_d"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/phenotypes"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Run a small sanity-sized preset")
    args = parser.parse_args(argv)
    if args.quick:
        args.top_k = 10

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.data_dir) / "phase_2" / "logs" / "metrics_summary.parquet"
    table = load_final_step_metrics(metrics_path)
    if "same_state_adjacency_fraction" not in table.column_names:
        table = table.append_column(
            "same_state_adjacency_fraction",
            pa.array([0.0] * table.num_rows, type=pa.float64()),
        )

    mi_col = pc.cast(table.column("delta_mi"), pa.float64(), safe=False)
    mi_filled = pc.if_else(pc.is_valid(mi_col), mi_col, pa.scalar(0.0))
    enriched = table.set_column(
        table.column_names.index("delta_mi"),
        "delta_mi",
        mi_filled,
    )

    rows = enriched.select(
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
        normalized = {
            "rule_id": str(row["rule_id"]),
            "state_entropy": (
                float(row["state_entropy"]) if row["state_entropy"] is not None else 0.0
            ),
            "predictability_hamming": float(row["predictability_hamming"])
            if row["predictability_hamming"] is not None
            else 0.0,
            "same_state_adjacency_fraction": float(row["same_state_adjacency_fraction"])
            if row["same_state_adjacency_fraction"] is not None
            else 0.0,
            "delta_mi": float(row["delta_mi"]),
        }
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
