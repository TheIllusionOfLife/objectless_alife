"""Shared utilities for analysis scripts."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq


def load_final_snapshots(
    sim_log_path: Path,
    rule_ids: set[str] | None = None,
) -> dict[str, tuple[tuple[int, int, int, int], ...]]:
    """Load final-step snapshots per rule_id from a simulation log.

    Parameters
    ----------
    sim_log_path:
        Path to a ``simulation_log.parquet`` file.
    rule_ids:
        Optional set of rule IDs to load.  When provided, only matching rows
        are read (filter pushed to the storage layer).  When *None*, all rules
        present in the file are loaded.

    Returns
    -------
    dict mapping rule_id to a tuple of ``(agent_id, x, y, state)`` tuples for
    the final simulation step of that rule.
    """
    filters = [("rule_id", "in", list(rule_ids))] if rule_ids is not None else None
    table = pq.read_table(
        sim_log_path,
        columns=["rule_id", "step", "agent_id", "x", "y", "state"],
        filters=filters,
    )
    rows = table.to_pylist()

    max_steps: dict[str, int] = {}
    for row in rows:
        rid = row["rule_id"]
        step = int(row["step"])
        if rid not in max_steps or step > max_steps[rid]:
            max_steps[rid] = step

    snapshots: dict[str, list[tuple[int, int, int, int]]] = {}
    for row in rows:
        rid = row["rule_id"]
        if int(row["step"]) != max_steps.get(rid, -1):
            continue
        snapshots.setdefault(rid, []).append(
            (int(row["agent_id"]), int(row["x"]), int(row["y"]), int(row["state"]))
        )

    return {rid: tuple(agents) for rid, agents in snapshots.items() if agents}
