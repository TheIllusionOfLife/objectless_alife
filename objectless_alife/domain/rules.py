"""Observation phase enums, rule-table indexing, and seeded rule generation.

This module is the canonical source for ``ObservationPhase`` and all
rule-table sizing / indexing helpers.  ``CLOCK_PERIOD`` is defined here
for historical reasons; prefer importing it from
``objectless_alife.config.constants`` in new code.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from random import Random
from typing import Sequence

from objectless_alife.config.constants import CLOCK_PERIOD


class ObservationPhase(Enum):
    """Observation table variants used to index rule actions."""

    PHASE1_DENSITY = 1
    PHASE2_PROFILE = 2
    CONTROL_DENSITY_CLOCK = 3
    RANDOM_WALK = 4
    PHASE1_CAPACITY_MATCHED = 5
    PHASE2_RANDOM_ENCODING = 6


def rule_table_size(phase: ObservationPhase) -> int:
    """Return rule table length for the selected observation phase."""
    if phase == ObservationPhase.PHASE1_DENSITY:
        return 20
    if phase in (
        ObservationPhase.PHASE2_PROFILE,
        ObservationPhase.CONTROL_DENSITY_CLOCK,
        ObservationPhase.PHASE1_CAPACITY_MATCHED,
        ObservationPhase.PHASE2_RANDOM_ENCODING,
    ):
        return 100
    if phase == ObservationPhase.RANDOM_WALK:
        return 1
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


def compute_capacity_matched_index(
    self_state: int, neighbor_count: int, dominant_state: int
) -> int:
    """Compute index for capacity-matched Phase-1 control.

    The index math intentionally matches ``compute_phase2_index``.
    Capacity-matching aliasing (20 effective observation bins tiled into
    100 table entries) is implemented at rule-table generation time in
    ``generate_rule_table(ObservationPhase.PHASE1_CAPACITY_MATCHED, ...)``.
    """
    return compute_phase2_index(self_state, neighbor_count, dominant_state)


def generate_rule_table(phase: ObservationPhase, seed: int) -> list[int]:
    """Generate a seeded rule table with action IDs in [0, 8]."""
    rng = Random(seed)
    size = rule_table_size(phase)

    if phase == ObservationPhase.PHASE1_CAPACITY_MATCHED:
        # Generate 20 base actions (Phase-1 observations), then tile to 100
        base = [rng.randint(0, 8) for _ in range(20)]
        table = [0] * 100
        for s in range(4):
            for n in range(5):
                action = base[s * 5 + n]
                for d in range(5):
                    table[s * 25 + n * 5 + d] = action
        return table

    if phase == ObservationPhase.PHASE2_RANDOM_ENCODING:
        # Same action values as Phase 2, but permute entries to destroy
        # the structured observation-to-action mapping.
        base_table = [rng.randint(0, 8) for _ in range(100)]
        perm = list(range(100))
        rng.shuffle(perm)
        return [base_table[perm[i]] for i in range(100)]

    return [rng.randint(0, 8) for _ in range(size)]
