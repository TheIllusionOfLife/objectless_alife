"""Parquet schema definitions and metric name constants for simulation artifacts.

All Arrow schemas used for persisting simulation logs, metrics, experiment runs,
density sweeps, multi-seed evaluations, and halt-window sweeps are centralised
here so that every module works against the same column contracts.
"""

from __future__ import annotations

import pyarrow as pa

# ---------------------------------------------------------------------------
# Schema version constants
# ---------------------------------------------------------------------------

AGGREGATE_SCHEMA_VERSION = 1
DENSITY_SWEEP_SCHEMA_VERSION = 1
RULE_PAYLOAD_SCHEMA_VERSION = 4

# ---------------------------------------------------------------------------
# Core simulation & metrics schemas
# ---------------------------------------------------------------------------

SIMULATION_SCHEMA = pa.schema(
    [
        ("rule_id", pa.string()),
        ("step", pa.int64()),
        ("agent_id", pa.int64()),
        ("x", pa.int64()),
        ("y", pa.int64()),
        ("state", pa.int64()),
        ("action", pa.int64()),
    ]
)

METRICS_SCHEMA = pa.schema(
    [
        ("rule_id", pa.string()),
        ("step", pa.int64()),
        ("state_entropy", pa.float64()),
        ("compression_ratio", pa.float64()),
        ("predictability_hamming", pa.float64()),
        ("morans_i", pa.float64()),
        ("cluster_count", pa.int64()),
        ("quasi_periodicity_peaks", pa.int64()),
        ("phase_transition_max_delta", pa.float64()),
        ("same_state_adjacency_fraction", pa.float64()),
        ("neighbor_mutual_information", pa.float64()),
        ("action_entropy_mean", pa.float64()),
        ("action_entropy_variance", pa.float64()),
        ("block_ncd", pa.float64()),
        ("mi_shuffle_null", pa.float64()),
    ]
)

PHASE_SUMMARY_METRIC_NAMES = [
    "state_entropy",
    "compression_ratio",
    "predictability_hamming",
    "morans_i",
    "cluster_count",
    "same_state_adjacency_fraction",
    "neighbor_mutual_information",
    "quasi_periodicity_peaks",
    "phase_transition_max_delta",
    "action_entropy_mean",
    "action_entropy_variance",
    "block_ncd",
    "mi_shuffle_null",
    "delta_mi",
]

# ---------------------------------------------------------------------------
# Density sweep schemas
# ---------------------------------------------------------------------------

DENSITY_SWEEP_RUNS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int64()),
        ("rule_id", pa.string()),
        ("phase", pa.int64()),
        ("grid_width", pa.int64()),
        ("grid_height", pa.int64()),
        ("num_agents", pa.int64()),
        ("density_ratio", pa.float64()),
        ("seed_batch", pa.int64()),
        ("rule_seed", pa.int64()),
        ("sim_seed", pa.int64()),
        ("survived", pa.bool_()),
        ("termination_reason", pa.string()),
        ("terminated_at", pa.int64()),
    ]
)

DENSITY_PHASE_SUMMARY_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int64()),
        ("phase", pa.int64()),
        ("grid_width", pa.int64()),
        ("grid_height", pa.int64()),
        ("num_agents", pa.int64()),
        ("density_ratio", pa.float64()),
        ("rules_evaluated", pa.int64()),
        ("survival_rate", pa.float64()),
        ("termination_rate", pa.float64()),
        ("mean_terminated_at", pa.float64()),
    ]
    + [
        (f"{metric}_{suffix}", pa.float64())
        for metric in PHASE_SUMMARY_METRIC_NAMES
        for suffix in ("mean", "p25", "p50", "p75")
    ]
)

DENSITY_PHASE_COMPARISON_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int64()),
        ("base_phase", pa.int64()),
        ("target_phase", pa.int64()),
        ("grid_width", pa.int64()),
        ("grid_height", pa.int64()),
        ("num_agents", pa.int64()),
        ("density_ratio", pa.float64()),
        ("metric", pa.string()),
        ("delta_absolute", pa.float64()),
        ("delta_relative", pa.float64()),
    ]
)

# ---------------------------------------------------------------------------
# Multi-seed & halt-window sweep schemas
# ---------------------------------------------------------------------------

MULTI_SEED_SCHEMA = pa.schema(
    [
        ("rule_seed", pa.int64()),
        ("sim_seed", pa.int64()),
        ("survived", pa.bool_()),
        ("termination_reason", pa.string()),
        ("neighbor_mutual_information", pa.float64()),
        ("mi_shuffle_null", pa.float64()),
        ("delta_mi", pa.float64()),
        ("n_pairs", pa.int64()),
        ("same_state_adjacency_fraction", pa.float64()),
        ("update_mode", pa.string()),
        ("state_uniform_mode", pa.string()),
        ("enable_viability_filters", pa.bool_()),
    ]
)

HALT_WINDOW_SWEEP_SCHEMA = pa.schema(
    [
        ("rule_seed", pa.int64()),
        ("halt_window", pa.int64()),
        ("survived", pa.bool_()),
        ("termination_reason", pa.string()),
        ("neighbor_mutual_information", pa.float64()),
        ("mi_shuffle_null", pa.float64()),
        ("delta_mi", pa.float64()),
        ("n_pairs", pa.int64()),
        ("update_mode", pa.string()),
        ("state_uniform_mode", pa.string()),
        ("enable_viability_filters", pa.bool_()),
    ]
)
