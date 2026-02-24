"""Parquet persistence helpers for simulation log and metrics streams."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from objectless_alife.io.schemas import SIMULATION_SCHEMA


def flush_sim_columns(
    sim_columns: dict[str, list[int | str]],
    simulation_log_path: Path,
    sim_writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter | None:
    """Write accumulated simulation rows to Parquet and clear in-memory buffers."""
    if not sim_columns["rule_id"]:
        return sim_writer
    sim_table = pa.Table.from_pydict(sim_columns, schema=SIMULATION_SCHEMA)
    if sim_writer is None:
        sim_writer = pq.ParquetWriter(simulation_log_path, SIMULATION_SCHEMA)
    sim_writer.write_table(sim_table)
    for values in sim_columns.values():
        values.clear()
    return sim_writer


# Keep private alias for backward compatibility with internal callers
_flush_sim_columns = flush_sim_columns
