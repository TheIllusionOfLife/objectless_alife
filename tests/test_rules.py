import pytest

from src.rules import (
    ObservationPhase,
    compute_control_index,
    compute_phase1_index,
    compute_phase2_index,
    dominant_neighbor_state,
    generate_rule_table,
    rule_table_size,
)


def test_rule_table_size_per_phase() -> None:
    assert rule_table_size(ObservationPhase.PHASE1_DENSITY) == 20
    assert rule_table_size(ObservationPhase.PHASE2_PROFILE) == 100


def test_compute_phase1_index() -> None:
    assert compute_phase1_index(self_state=2, neighbor_count=3) == 13


def test_dominant_neighbor_state_tie_uses_lowest_state() -> None:
    # tie between 1 and 2 -> choose 1
    assert dominant_neighbor_state([1, 2, 2, 1]) == 1


def test_dominant_neighbor_state_empty_returns_none_slot() -> None:
    assert dominant_neighbor_state([]) == 4


def test_compute_phase2_index() -> None:
    assert compute_phase2_index(self_state=3, neighbor_count=4, dominant_state=2) == 97


def test_generate_rule_table_is_seeded_and_action_bounded() -> None:
    table_a = generate_rule_table(ObservationPhase.PHASE1_DENSITY, seed=10)
    table_b = generate_rule_table(ObservationPhase.PHASE1_DENSITY, seed=10)
    assert table_a == table_b
    assert len(table_a) == 20
    assert all(0 <= action <= 8 for action in table_a)


def test_compute_control_index() -> None:
    # 2*25 + 3*5 + 4 = 50 + 15 + 4 = 69
    assert compute_control_index(self_state=2, neighbor_count=3, step_mod=4) == 69


def test_compute_control_index_bounds() -> None:
    with pytest.raises(ValueError):
        compute_control_index(self_state=4, neighbor_count=0, step_mod=0)
    with pytest.raises(ValueError):
        compute_control_index(self_state=0, neighbor_count=5, step_mod=0)
    with pytest.raises(ValueError):
        compute_control_index(self_state=0, neighbor_count=0, step_mod=5)
    with pytest.raises(ValueError):
        compute_control_index(self_state=-1, neighbor_count=0, step_mod=0)


def test_rule_table_size_control() -> None:
    assert rule_table_size(ObservationPhase.CONTROL_DENSITY_CLOCK) == 100


def test_generate_rule_table_control() -> None:
    table = generate_rule_table(ObservationPhase.CONTROL_DENSITY_CLOCK, seed=42)
    assert len(table) == 100
    assert all(0 <= action <= 8 for action in table)


def test_rule_table_size_random_walk() -> None:
    assert rule_table_size(ObservationPhase.RANDOM_WALK) == 1


def test_generate_rule_table_random_walk() -> None:
    table = generate_rule_table(ObservationPhase.RANDOM_WALK, seed=0)
    assert len(table) == 1
    assert all(0 <= action <= 8 for action in table)
