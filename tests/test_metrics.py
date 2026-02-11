import pytest

from src.metrics import (
    action_entropy,
    action_entropy_variance,
    block_ncd,
    cluster_count_by_state,
    compression_ratio,
    morans_i_occupied,
    neighbor_mutual_information,
    normalized_hamming_distance,
    phase_transition_max_delta,
    quasi_periodicity_peak_count,
    serialize_snapshot,
    state_entropy,
)


def test_state_entropy_binary_balanced() -> None:
    assert state_entropy([0, 0, 1, 1]) == 1.0


def test_normalized_hamming_distance() -> None:
    assert normalized_hamming_distance([0, 1, 2], [0, 2, 2]) == 1 / 3


def test_compression_ratio_positive() -> None:
    payload = b"aaaaaaaaaabbbbbbbbbbcccccccccc"
    assert compression_ratio(payload) > 0


def test_serialize_snapshot_raises_when_coordinates_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        serialize_snapshot(((0, 5, 0, 1),), grid_width=5, grid_height=5)


def test_morans_i_occupied_returns_nan_when_variance_zero() -> None:
    snapshot = ((0, 0, 0, 1), (1, 1, 0, 1))
    value = morans_i_occupied(snapshot, grid_width=5, grid_height=5)
    assert value != value  # NaN


def test_cluster_count_by_state_counts_connected_components() -> None:
    snapshot = (
        (0, 0, 0, 1),
        (1, 1, 0, 1),
        (2, 4, 4, 1),
        (3, 2, 2, 2),
        (4, 2, 3, 2),
    )
    assert cluster_count_by_state(snapshot, grid_width=5, grid_height=5) == 3


def test_quasi_periodicity_peak_count_detects_repeating_signal() -> None:
    series = [0, 1, 0, -1] * 8
    assert quasi_periodicity_peak_count(series) >= 1


def test_phase_transition_max_delta() -> None:
    assert phase_transition_max_delta([0.1, 0.3, 0.9, 0.4]) == pytest.approx(0.6)


def test_action_entropy_variance_positive_for_diverged_agents() -> None:
    per_agent_actions = [[0, 0, 0, 0], [0, 1, 2, 3], [8, 8, 8, 8]]
    assert action_entropy([0, 1, 2, 3]) == 2.0
    assert action_entropy_variance(per_agent_actions) > 0.0


def test_neighbor_mutual_information_zero_when_independent() -> None:
    snapshot = (
        (0, 0, 0, 0),
        (1, 2, 2, 1),
    )
    assert neighbor_mutual_information(snapshot, grid_width=5, grid_height=5) == pytest.approx(0.0)


def test_block_ncd_is_bounded() -> None:
    a = b"ABCD" * 32
    b = b"ABCE" * 32
    ncd = block_ncd(a, b)
    assert 0.0 <= ncd <= 1.0
