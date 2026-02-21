"""Shared utilities for analysis scripts."""

from __future__ import annotations

from pathlib import Path

import pyarrow.compute as pc
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

    if len(table) == 0:
        return {}

    # Arrow-native: aggregate max step per rule, join back, filter to final rows.
    max_step_table = table.group_by("rule_id").aggregate([("step", "max")])
    joined = table.join(max_step_table, "rule_id")
    mask = pc.equal(joined.column("step"), joined.column("step_max"))
    final_table = joined.filter(mask)

    # Build snapshots from the filtered column arrays.
    rid_col = final_table.column("rule_id").to_pylist()
    agent_id_col = final_table.column("agent_id").to_pylist()
    x_col = final_table.column("x").to_pylist()
    y_col = final_table.column("y").to_pylist()
    state_col = final_table.column("state").to_pylist()

    snapshots: dict[str, list[tuple[int, int, int, int]]] = {}
    for rid, aid, x, y, s in zip(rid_col, agent_id_col, x_col, y_col, state_col, strict=True):
        snapshots.setdefault(rid, []).append((int(aid), int(x), int(y), int(s)))

    return {rid: tuple(agents) for rid, agents in snapshots.items() if agents}
