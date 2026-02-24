"""Rule selection utilities: select top-K rules by delta_mi."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


def _load_final_step_metrics(metrics_path: Path) -> dict[str, dict[str, float]]:
    """Load final step metrics for each rule from parquet."""
    metrics_file = pq.ParquetFile(metrics_path)
    max_steps: dict[str, int] = {}
    rule_metrics: dict[str, dict[str, float]] = {}

    for batch in metrics_file.iter_batches(
        columns=["rule_id", "step", "neighbor_mutual_information", "mi_shuffle_null"],
        batch_size=8192,
    ):
        batch_dict = batch.to_pydict()
        for idx, rid in enumerate(batch_dict["rule_id"]):
            rid = str(rid)
            step = int(batch_dict["step"][idx])
            if rid not in max_steps or step > max_steps[rid]:
                max_steps[rid] = step
                mi = batch_dict["neighbor_mutual_information"][idx]
                null = batch_dict["mi_shuffle_null"][idx]
                if mi is not None and null is not None and mi == mi and null == null:
                    rule_metrics[rid] = {"mi": float(mi), "null": float(null)}
                else:
                    rule_metrics.pop(rid, None)
    return rule_metrics


def _load_survived_rule_seeds(rules_dir: Path) -> dict[str, int]:
    """Load rule seeds for all surviving rules from parquet or JSON."""
    experiment_parquet = rules_dir.parent / "logs" / "experiment_runs.parquet"
    survived_seeds: dict[str, int] = {}

    if experiment_parquet.exists():
        table = pq.read_table(
            experiment_parquet,
            columns=["rule_id", "rule_seed"],
            filters=[("survived", "=", True)],
        )
        pydict = table.to_pydict()
        for rid, seed_val in zip(pydict["rule_id"], pydict["rule_seed"], strict=False):
            if seed_val is not None:
                survived_seeds[str(rid)] = int(seed_val)
    else:
        for path in sorted(rules_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                if not data.get("survived", False):
                    continue
                rid = str(data["rule_id"])
                seed = data["metadata"]["rule_seed"]
                survived_seeds[rid] = int(seed)
            except (json.JSONDecodeError, KeyError, OSError):
                continue

    return survived_seeds


def select_top_rules_by_delta_mi(
    metrics_path: Path,
    rules_dir: Path,
    top_k: int = 50,
) -> list[int]:
    """Select top-K rule seeds by delta_mi from existing experiment data.

    Returns a list of rule seeds sorted by descending delta_mi.
    Only includes surviving rules.
    """
    rule_metrics = _load_final_step_metrics(metrics_path)
    survived_seeds = _load_survived_rule_seeds(rules_dir)

    candidate_seeds: list[tuple[int, float]] = []

    for rid, seed in survived_seeds.items():
        if rid in rule_metrics:
            m = rule_metrics[rid]
            delta = m["mi"] - m["null"]
            candidate_seeds.append((seed, delta))

    candidate_seeds.sort(key=lambda x: x[1], reverse=True)
    return [seed for seed, _ in candidate_seeds[:top_k]]
