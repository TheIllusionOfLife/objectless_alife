from __future__ import annotations

from enum import Enum
from random import Random
from typing import Sequence


class ObservationPhase(Enum):
    PHASE1_DENSITY = 1
    PHASE2_PROFILE = 2


def rule_table_size(phase: ObservationPhase) -> int:
    if phase == ObservationPhase.PHASE1_DENSITY:
        return 20
    return 100


def dominant_neighbor_state(neighbor_states: Sequence[int]) -> int:
    if not neighbor_states:
        return 4

    counts: dict[int, int] = {}
    for state in neighbor_states:
        counts[state] = counts.get(state, 0) + 1

    max_count = max(counts.values())
    candidates = [state for state, count in counts.items() if count == max_count]
    return min(candidates)


def compute_phase1_index(self_state: int, neighbor_count: int) -> int:
    if not 0 <= self_state <= 3:
        raise ValueError("self_state must be in [0, 3]")
    if not 0 <= neighbor_count <= 4:
        raise ValueError("neighbor_count must be in [0, 4]")
    return self_state * 5 + neighbor_count


def compute_phase2_index(self_state: int, neighbor_count: int, dominant_state: int) -> int:
    if not 0 <= self_state <= 3:
        raise ValueError("self_state must be in [0, 3]")
    if not 0 <= neighbor_count <= 4:
        raise ValueError("neighbor_count must be in [0, 4]")
    if not 0 <= dominant_state <= 4:
        raise ValueError("dominant_state must be in [0, 4]")
    return self_state * 25 + neighbor_count * 5 + dominant_state


def generate_rule_table(phase: ObservationPhase, seed: int) -> list[int]:
    rng = Random(seed)
    size = rule_table_size(phase)
    return [rng.randint(0, 8) for _ in range(size)]
