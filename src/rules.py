from __future__ import annotations

from collections import Counter
from enum import Enum
from random import Random
from typing import Sequence


class ObservationPhase(Enum):
    """Observation table variants used to index rule actions."""

    PHASE1_DENSITY = 1
    PHASE2_PROFILE = 2
    CONTROL_DENSITY_CLOCK = 3


def rule_table_size(phase: ObservationPhase) -> int:
    """Return rule table length for the selected observation phase."""
    if phase == ObservationPhase.PHASE1_DENSITY:
        return 20
    if phase in (ObservationPhase.PHASE2_PROFILE, ObservationPhase.CONTROL_DENSITY_CLOCK):
        return 100
    raise ValueError(f"Unsupported observation phase: {phase}")


def dominant_neighbor_state(neighbor_states: Sequence[int]) -> int:
    """Return dominant neighbor state with deterministic tie-break.

    Returns 4 when there are no occupied neighbors, matching the phase-2
    "none" sentinel slot.
    """
    if not neighbor_states:
        return 4

    counts = Counter(neighbor_states)
    max_count = max(counts.values())
    candidates = [state for state, count in counts.items() if count == max_count]
    return min(candidates)


def compute_phase1_index(self_state: int, neighbor_count: int) -> int:
    """Compute phase-1 rule table index from self state and neighbor density."""
    if not 0 <= self_state <= 3:
        raise ValueError("self_state must be in [0, 3]")
    if not 0 <= neighbor_count <= 4:
        raise ValueError("neighbor_count must be in [0, 4]")
    return self_state * 5 + neighbor_count


def compute_phase2_index(self_state: int, neighbor_count: int, dominant_state: int) -> int:
    """Compute phase-2 rule table index from state, density, and dominant state."""
    if not 0 <= self_state <= 3:
        raise ValueError("self_state must be in [0, 3]")
    if not 0 <= neighbor_count <= 4:
        raise ValueError("neighbor_count must be in [0, 4]")
    if not 0 <= dominant_state <= 4:
        raise ValueError("dominant_state must be in [0, 4]")
    return self_state * 25 + neighbor_count * 5 + dominant_state


CLOCK_PERIOD = 5
"""Number of distinct step-clock values for the control phase."""


def compute_control_index(self_state: int, neighbor_count: int, step_mod: int) -> int:
    """Compute control rule table index from state, density, and step clock.

    Uses step_mod (step_number % CLOCK_PERIOD) as a non-informative third
    dimension, producing a 100-entry table comparable in size to phase 2
    but without neighbor state information.
    """
    if not 0 <= self_state <= 3:
        raise ValueError("self_state must be in [0, 3]")
    if not 0 <= neighbor_count <= 4:
        raise ValueError("neighbor_count must be in [0, 4]")
    if not 0 <= step_mod <= CLOCK_PERIOD - 1:
        raise ValueError(f"step_mod must be in [0, {CLOCK_PERIOD - 1}]")
    return self_state * 25 + neighbor_count * CLOCK_PERIOD + step_mod


def generate_rule_table(phase: ObservationPhase, seed: int) -> list[int]:
    """Generate a seeded rule table with action IDs in [0, 8]."""
    rng = Random(seed)
    size = rule_table_size(phase)
    return [rng.randint(0, 8) for _ in range(size)]
