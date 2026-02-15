import pytest

from src.rules import (
    ObservationPhase,
    compute_capacity_matched_index,
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


# --- PHASE1_CAPACITY_MATCHED tests ---


def test_rule_table_size_capacity_matched() -> None:
    assert rule_table_size(ObservationPhase.PHASE1_CAPACITY_MATCHED) == 100


def test_capacity_matched_index_aliases_phase1() -> None:
    """Capacity-matched indices should map to the same observation as Phase 1."""
    table = generate_rule_table(ObservationPhase.PHASE1_CAPACITY_MATCHED, seed=42)
    for s in range(4):
        for n in range(5):
            for d in range(5):
                cm_idx = compute_capacity_matched_index(s, n, d)
                assert 0 <= cm_idx < 100
                # Different d values for same (s, n) should map to same action
                cm_idx2 = compute_capacity_matched_index(s, n, 0)
                assert table[cm_idx] == table[cm_idx2]


def test_generate_rule_table_capacity_matched() -> None:
    table = generate_rule_table(ObservationPhase.PHASE1_CAPACITY_MATCHED, seed=42)
    assert len(table) == 100
    assert all(0 <= action <= 8 for action in table)


# --- PHASE2_RANDOM_ENCODING tests ---


def test_rule_table_size_random_encoding() -> None:
    assert rule_table_size(ObservationPhase.PHASE2_RANDOM_ENCODING) == 100


def test_random_encoding_deterministic_with_seed() -> None:
    """Same seed → same table."""
    t1 = generate_rule_table(ObservationPhase.PHASE2_RANDOM_ENCODING, seed=42)
    t2 = generate_rule_table(ObservationPhase.PHASE2_RANDOM_ENCODING, seed=42)
    assert t1 == t2


def test_random_encoding_correct_size() -> None:
    table = generate_rule_table(ObservationPhase.PHASE2_RANDOM_ENCODING, seed=99)
    assert len(table) == 100
    assert all(0 <= action <= 8 for action in table)


def test_random_encoding_differs_from_phase2() -> None:
    """Random encoding table should be a permutation of Phase 2 table (same seed)."""
    p2 = generate_rule_table(ObservationPhase.PHASE2_PROFILE, seed=42)
    re = generate_rule_table(ObservationPhase.PHASE2_RANDOM_ENCODING, seed=42)
    assert p2 != re  # Same base entries, but permuted
    assert sorted(p2) == sorted(re)  # Same multiset of values


def test_capacity_matched_index_rejects_out_of_range() -> None:
    """Boundary validation for compute_capacity_matched_index."""
    with pytest.raises(ValueError):
        compute_capacity_matched_index(self_state=4, neighbor_count=0, dominant_state=0)
    with pytest.raises(ValueError):
        compute_capacity_matched_index(self_state=0, neighbor_count=5, dominant_state=0)
    with pytest.raises(ValueError):
        compute_capacity_matched_index(self_state=0, neighbor_count=0, dominant_state=5)
    with pytest.raises(ValueError):
        compute_capacity_matched_index(self_state=-1, neighbor_count=0, dominant_state=0)
