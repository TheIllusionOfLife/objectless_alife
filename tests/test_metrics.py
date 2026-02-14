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


def _naive_mi(snapshot: tuple[tuple[int, int, int, int], ...], grid_width: int) -> float:
    """Compute naive plug-in MI (no bias correction) for test comparison."""
    import math as _math
    from collections import Counter

    occupied = {(x, y): state for _, x, y, state in snapshot}
    pairs: list[tuple[int, int]] = []
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for x, y in occupied:
        for nx, ny in (((x + 1) % grid_width, y), (x, (y + 1) % grid_width)):
            if (nx, ny) not in occupied:
                continue
            a, b = (x, y), (nx, ny)
            edge = (a, b) if a < b else (b, a)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            pairs.append((occupied[a], occupied[b]))

    joint = Counter(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    n = len(pairs)
    mi = 0.0
    for (sl, sr), c in joint.items():
        p_ab = c / n
        mi += p_ab * _math.log2(p_ab / (left[sl] / n * (right[sr] / n)))
    return mi


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


def test_mi_miller_madow_reduces_bias() -> None:
    """Corrected MI should be lower than naive MI for a small-sample case."""
    # 4 agents in a 3x3 grid — very few pairs, so bias correction matters
    snapshot = (
        (0, 0, 0, 0),
        (1, 1, 0, 1),
        (2, 0, 1, 0),
        (3, 1, 1, 1),
    )
    mi = neighbor_mutual_information(snapshot, grid_width=3, grid_height=3)
    assert mi >= 0.0
    assert mi < float("inf")
    naive_mi = _naive_mi(snapshot, grid_width=3)
    assert mi < naive_mi


def test_mi_miller_madow_zero_for_independent() -> None:
    """MI should be 0 for independent (non-neighboring) agents."""
    snapshot = (
        (0, 0, 0, 0),
        (1, 2, 2, 1),
    )
    mi = neighbor_mutual_information(snapshot, grid_width=5, grid_height=5)
    assert mi == pytest.approx(0.0)


def test_mi_miller_madow_matches_naive_for_large_sample() -> None:
    """With enough pairs, the correction is negligible."""
    # Build a dense 10x10 grid with alternating states — many pairs
    snapshot = tuple((i * 10 + j, i, j, (i + j) % 4) for i in range(10) for j in range(10))
    mi = neighbor_mutual_information(snapshot, grid_width=10, grid_height=10)
    naive_mi = _naive_mi(snapshot, grid_width=10)

    # With 200 pairs and only a few bins, correction should be tiny
    assert abs(mi - naive_mi) < 0.05


def test_block_ncd_is_bounded() -> None:
    a = b"ABCD" * 32
    b = b"ABCE" * 32
    ncd = block_ncd(a, b)
    assert 0.0 <= ncd <= 1.0
