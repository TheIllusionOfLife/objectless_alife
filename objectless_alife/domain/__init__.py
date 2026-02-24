"""Domain layer: world model, rules, filters, and typed snapshots."""

from objectless_alife.domain.filters import (
    HaltDetector,
    LowActivityDetector,
    ShortPeriodDetector,
    StateUniformDetector,
    TerminationReason,
)
from objectless_alife.domain.rules import (
    CLOCK_PERIOD,
    ObservationPhase,
    compute_capacity_matched_index,
    compute_control_index,
    compute_phase1_index,
    compute_phase2_index,
    dominant_neighbor_state,
    generate_rule_table,
    rule_table_size,
)
from objectless_alife.domain.snapshot import AgentState, Snapshot
from objectless_alife.domain.world import Agent, World, WorldConfig

__all__ = [
    "Agent",
    "AgentState",
    "CLOCK_PERIOD",
    "HaltDetector",
    "LowActivityDetector",
    "ObservationPhase",
    "ShortPeriodDetector",
    "Snapshot",
    "StateUniformDetector",
    "TerminationReason",
    "World",
    "WorldConfig",
    "compute_capacity_matched_index",
    "compute_control_index",
    "compute_phase1_index",
    "compute_phase2_index",
    "dominant_neighbor_state",
    "generate_rule_table",
    "rule_table_size",
]
