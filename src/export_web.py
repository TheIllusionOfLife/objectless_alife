"""Export simulation data as compact JSON for the web visualization app.

Functions:
    export_single  — one rule's trajectory
    export_paired  — Phase 2 + Control side-by-side (matched sim_seed)
    export_batch   — top-N rules by MI for a phase
    export_gallery — diverse set of rules (spread across MI range)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_GRID_WIDTH = 20
DEFAULT_GRID_HEIGHT = 20


def _load_rule_json(data_dir: Path, rule_id: str) -> dict:
    """Load a rule JSON file and return its metadata."""
    rule_path = data_dir / "rules" / f"{rule_id}.json"
    if not rule_path.exists():
        raise ValueError(f"Rule JSON not found: {rule_path}")
    return json.loads(rule_path.read_text())


def _build_single_payload(data_dir: Path, rule_id: str) -> dict:
    """Build the single-rule JSON payload from Parquet data."""
    sim_path = data_dir / "logs" / "simulation_log.parquet"
    metrics_path = data_dir / "logs" / "metrics_summary.parquet"

    sim_table = pq.read_table(sim_path, filters=[("rule_id", "=", rule_id)])
    if sim_table.num_rows == 0:
        raise ValueError(f"No simulation rows found for rule_id={rule_id}")

    metric_table = pq.read_table(metrics_path, filters=[("rule_id", "=", rule_id)])

    rule_payload = _load_rule_json(data_dir, rule_id)
    raw_metadata = rule_payload.get("metadata", {})

    sim_rows = sim_table.to_pydict()
    steps_col = sim_rows["step"]

    grid_width = raw_metadata.get("grid_width", DEFAULT_GRID_WIDTH)
    grid_height = raw_metadata.get("grid_height", DEFAULT_GRID_HEIGHT)

    # Pre-group rows by step for O(N) frame construction
    step_groups: dict[int, list[tuple[int, list]]] = {}
    for i in range(len(steps_col)):
        step = int(steps_col[i])
        agent_entry = (
            int(sim_rows["agent_id"][i]),
            [int(sim_rows["x"][i]), int(sim_rows["y"][i]), int(sim_rows["state"][i])],
        )
        step_groups.setdefault(step, []).append(agent_entry)

    unique_steps = sorted(step_groups)
    first_step_agents = step_groups[unique_steps[0]]
    num_agents = len({aid for aid, _ in first_step_agents})

    # Build frames
    frames = []
    for step in unique_steps:
        entries = step_groups[step]
        entries.sort(key=lambda e: e[0])
        sorted_agents = [agent for _, agent in entries]
        frames.append({"step": step, "agents": sorted_agents})

    # Build metrics series
    metric_dict = metric_table.to_pydict()
    metric_steps = metric_dict["step"]
    step_order = sorted(range(len(metric_steps)), key=lambda i: metric_steps[i])

    mi_values = metric_dict.get("neighbor_mutual_information", [])
    mi_series = []
    for i in step_order:
        val = mi_values[i] if i < len(mi_values) else 0.0
        mi_series.append(
            float(val)
            if val is not None and not (isinstance(val, float) and math.isnan(val))
            else 0.0
        )

    phase_name = _phase_name_from_rule_id(rule_id)

    return {
        "meta": {
            "rule_id": rule_id,
            "phase": phase_name,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "num_agents": num_agents,
            "steps": len(unique_steps),
        },
        "frames": frames,
        "metrics": {
            "neighbor_mutual_information": mi_series,
        },
    }


def _phase_name_from_rule_id(rule_id: str) -> str:
    """Extract phase name from rule_id format 'phase{N}_rs{X}_ss{Y}'."""
    if rule_id.startswith("phase1_"):
        return "phase_1"
    if rule_id.startswith("phase2_"):
        return "phase_2"
    if rule_id.startswith("phase3_"):
        return "control"
    if rule_id.startswith("phase4_"):
        return "random_walk"
    if rule_id.startswith("phase5_"):
        return "phase1_capacity_matched"
    if rule_id.startswith("phase6_"):
        return "phase2_random_encoding"
    return "unknown"


def _find_rule_id_by_sim_seed(data_dir: Path, sim_seed: int) -> str | None:
    """Find a rule_id matching the given sim_seed from rule JSON files."""
    rules_dir = data_dir / "rules"
    if not rules_dir.exists():
        return None
    for rule_path in sorted(rules_dir.glob("*.json")):
        payload = json.loads(rule_path.read_text())
        metadata = payload.get("metadata", {})
        if metadata.get("sim_seed") == sim_seed:
            return payload["rule_id"]
    return None


def _get_surviving_rules_ranked_by_mi(data_dir: Path) -> list[tuple[str, float]]:
    """Return (rule_id, final_mi) pairs for surviving rules, sorted by MI desc."""
    rules_dir = data_dir / "rules"
    metrics_path = data_dir / "logs" / "metrics_summary.parquet"

    if not rules_dir.exists() or not metrics_path.exists():
        return []

    # Collect survived rule IDs
    survived_ids: set[str] = set()
    for rule_path in sorted(rules_dir.glob("*.json")):
        payload = json.loads(rule_path.read_text())
        if payload.get("survived", False):
            survived_ids.add(payload["rule_id"])

    if not survived_ids:
        return []

    # Read final-step MI per rule
    metrics_file = pq.ParquetFile(metrics_path)
    max_steps: dict[str, int] = {}
    rule_mi: dict[str, float] = {}

    for batch in metrics_file.iter_batches(
        columns=["rule_id", "step", "neighbor_mutual_information"],
        batch_size=8192,
    ):
        batch_dict = batch.to_pydict()
        for idx, rid in enumerate(batch_dict["rule_id"]):
            if rid not in survived_ids:
                continue
            step = int(batch_dict["step"][idx])
            if rid not in max_steps or step > max_steps[rid]:
                max_steps[rid] = step
                mi = batch_dict["neighbor_mutual_information"][idx]
                if mi is not None and not (isinstance(mi, float) and math.isnan(mi)):
                    rule_mi[rid] = float(mi)

    ranked = sorted(rule_mi.items(), key=lambda pair: pair[1], reverse=True)
    return ranked


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def export_single(data_dir: Path, rule_id: str, output: Path) -> None:
    """Export one rule's trajectory as web-ready JSON."""
    data_dir = Path(data_dir)
    output = Path(output)
    payload = _build_single_payload(data_dir, rule_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def export_paired(
    phase2_dir: Path,
    control_dir: Path,
    sim_seed: int,
    output: Path,
) -> None:
    """Export Phase 2 + Control pair (matched sim_seed) as web-ready JSON."""
    phase2_dir = Path(phase2_dir)
    control_dir = Path(control_dir)
    output = Path(output)

    p2_rule_id = _find_rule_id_by_sim_seed(phase2_dir, sim_seed)
    if p2_rule_id is None:
        raise ValueError(f"No Phase 2 rule found with sim_seed={sim_seed}")

    ctrl_rule_id = _find_rule_id_by_sim_seed(control_dir, sim_seed)
    if ctrl_rule_id is None:
        raise ValueError(f"No Control rule found with sim_seed={sim_seed}")

    left = _build_single_payload(phase2_dir, p2_rule_id)
    right = _build_single_payload(control_dir, ctrl_rule_id)

    paired = {
        "left": left,
        "right": right,
        "meta": {
            "left_phase": "phase_2",
            "right_phase": "control",
            "sim_seed": sim_seed,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(paired, ensure_ascii=False, separators=(",", ":")))


def export_batch(data_dir: Path, top_n: int, output_dir: Path) -> None:
    """Export top-N rules by final MI as individual web-ready JSON files."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = _get_surviving_rules_ranked_by_mi(data_dir)
    for rule_id, _ in ranked[:top_n]:
        payload = _build_single_payload(data_dir, rule_id)
        out_file = output_dir / f"{rule_id}.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def export_gallery(data_dir: Path, count: int, output_dir: Path) -> None:
    """Export a diverse set of rules spread across the MI range.

    Selects rules at evenly-spaced quantile positions from the ranked
    MI list to showcase pattern diversity.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = _get_surviving_rules_ranked_by_mi(data_dir)
    if not ranked or count <= 0:
        return

    n_available = len(ranked)
    actual_count = min(count, n_available)

    if actual_count == 1:
        indices = [0]
    else:
        indices = [int(i * (n_available - 1) / (actual_count - 1)) for i in range(actual_count)]

    selected = [ranked[i] for i in indices]
    for rule_id, _ in selected:
        payload = _build_single_payload(data_dir, rule_id)
        out_file = output_dir / f"{rule_id}.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for web data export."""
    parser = argparse.ArgumentParser(description="Export simulation data as web-ready JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    # single
    p_single = sub.add_parser("single", help="Export one rule's trajectory")
    p_single.add_argument("--data-dir", type=Path, required=True)
    p_single.add_argument("--rule-id", type=str, required=True)
    p_single.add_argument("--output", type=Path, required=True)

    # paired
    p_paired = sub.add_parser("paired", help="Export Phase 2 + Control pair")
    p_paired.add_argument("--phase2-dir", type=Path, required=True)
    p_paired.add_argument("--control-dir", type=Path, required=True)
    p_paired.add_argument("--sim-seed", type=int, required=True)
    p_paired.add_argument("--output", type=Path, required=True)

    # batch
    p_batch = sub.add_parser("batch", help="Export top-N rules by MI")
    p_batch.add_argument("--data-dir", type=Path, required=True)
    p_batch.add_argument("--top-n", type=int, default=5)
    p_batch.add_argument("--output-dir", type=Path, required=True)

    # gallery
    p_gallery = sub.add_parser("gallery", help="Export diverse set of rules")
    p_gallery.add_argument("--data-dir", type=Path, required=True)
    p_gallery.add_argument("--count", type=int, default=9)
    p_gallery.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "single":
        export_single(data_dir=args.data_dir, rule_id=args.rule_id, output=args.output)
    elif args.command == "paired":
        export_paired(
            phase2_dir=args.phase2_dir,
            control_dir=args.control_dir,
            sim_seed=args.sim_seed,
            output=args.output,
        )
    elif args.command == "batch":
        export_batch(data_dir=args.data_dir, top_n=args.top_n, output_dir=args.output_dir)
    elif args.command == "gallery":
        export_gallery(data_dir=args.data_dir, count=args.count, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
