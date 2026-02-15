import random

import pytest

from src.metrics import (
    action_entropy,
    action_entropy_variance,
    block_ncd,
    block_shuffle_null_mi,
    cluster_count_by_state,
    compression_ratio,
    fixed_marginal_null_mi,
    morans_i_occupied,
    neighbor_mutual_information,
    neighbor_transfer_entropy,
    normalized_hamming_distance,
    phase_transition_max_delta,
    quasi_periodicity_peak_count,
    same_state_adjacency_fraction,
    serialize_snapshot,
    shuffle_null_mi,
    spatial_scramble_mi,
    state_entropy,
)


def _naive_mi(
    snapshot: tuple[tuple[int, int, int, int], ...],
    grid_width: int,
    grid_height: int,
) -> float:
    """Compute naive plug-in MI (no bias correction) for test comparison."""
    import math as _math
    from collections import Counter

    occupied = {(x, y): state for _, x, y, state in snapshot}
    pairs: list[tuple[int, int]] = []
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for x, y in occupied:
        for nx, ny in (((x + 1) % grid_width, y), (x, (y + 1) % grid_height)):
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
    # 8 agents filling a 4x2 grid with states that produce all 4 joint bins
    # (K_joint=4, K_left=2, K_right=2) so correction = 1/(2n ln2) > 0
    snapshot = (
        (0, 0, 0, 0),
        (1, 1, 0, 0),
        (2, 2, 0, 1),
        (3, 3, 0, 1),
        (4, 0, 1, 1),
        (5, 1, 1, 1),
        (6, 2, 1, 0),
        (7, 3, 1, 0),
    )
    mi = neighbor_mutual_information(snapshot, grid_width=4, grid_height=2)
    assert mi >= 0.0
    assert mi < float("inf")
    naive_mi = _naive_mi(snapshot, grid_width=4, grid_height=2)
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
    naive_mi = _naive_mi(snapshot, grid_width=10, grid_height=10)

    # With 200 pairs and only a few bins, correction should be tiny
    assert abs(mi - naive_mi) < 0.05


def test_shuffle_null_mi_deterministic_with_seed() -> None:
    """Same seed produces identical shuffle null MI."""
    snapshot = tuple((i * 10 + j, i, j, (i + j) % 4) for i in range(10) for j in range(10))
    result1 = shuffle_null_mi(snapshot, 10, 10, n_shuffles=50, rng=random.Random(42))
    result2 = shuffle_null_mi(snapshot, 10, 10, n_shuffles=50, rng=random.Random(42))
    assert result1 == result2


def test_shuffle_null_mi_near_zero_for_independent() -> None:
    """Non-neighboring agents have zero MI regardless of shuffling."""
    snapshot = (
        (0, 0, 0, 0),
        (1, 4, 4, 1),
    )
    result = shuffle_null_mi(snapshot, 10, 10, n_shuffles=50, rng=random.Random(7))
    assert result == pytest.approx(0.0)


def test_shuffle_null_mi_below_observed_for_correlated() -> None:
    """Structured (correlated) states have observed MI > shuffle null mean."""
    # Checkerboard pattern: neighbors always differ → high MI
    snapshot = tuple((i * 10 + j, i, j, (i + j) % 2) for i in range(10) for j in range(10))
    observed_mi = neighbor_mutual_information(snapshot, 10, 10)
    null_mi = shuffle_null_mi(snapshot, 10, 10, n_shuffles=100, rng=random.Random(99))
    assert observed_mi > null_mi


def test_shuffle_null_mi_zero_when_no_pairs() -> None:
    """No neighbor pairs → 0.0."""
    snapshot = (
        (0, 0, 0, 0),
        (1, 5, 5, 1),
    )
    result = shuffle_null_mi(snapshot, 10, 10, n_shuffles=20, rng=random.Random(1))
    assert result == 0.0


def test_same_state_adjacency_fraction_all_same_state() -> None:
    """All occupied cells share the same state → fraction = 1.0."""
    snapshot = (
        (0, 0, 0, 1),
        (1, 1, 0, 1),
        (2, 0, 1, 1),
        (3, 1, 1, 1),
    )
    result = same_state_adjacency_fraction(snapshot, grid_width=5, grid_height=5)
    assert result == pytest.approx(1.0)


def test_same_state_adjacency_fraction_alternating_states() -> None:
    """Checkerboard pattern: no adjacent pair shares the same state → fraction = 0.0."""
    # 2x2 grid fully occupied, checkerboard with 2 states on a torus
    snapshot = (
        (0, 0, 0, 0),
        (1, 1, 0, 1),
        (2, 0, 1, 1),
        (3, 1, 1, 0),
    )
    result = same_state_adjacency_fraction(snapshot, grid_width=2, grid_height=2)
    assert result == pytest.approx(0.0)


def test_same_state_adjacency_fraction_known_small_grid() -> None:
    """Known small grid with specific fraction."""
    # 3 agents on a 5x5 grid:
    # (0,0)=1, (1,0)=1, (2,0)=2
    # Pairs: (0,0)-(1,0) same=1, (1,0)-(2,0) diff=0
    # Fraction = 1/2 = 0.5
    snapshot = (
        (0, 0, 0, 1),
        (1, 1, 0, 1),
        (2, 2, 0, 2),
    )
    result = same_state_adjacency_fraction(snapshot, grid_width=5, grid_height=5)
    assert result == pytest.approx(0.5)


def test_same_state_adjacency_fraction_no_pairs() -> None:
    """No adjacent occupied pairs → NaN."""
    snapshot = (
        (0, 0, 0, 0),
        (1, 4, 4, 1),
    )
    result = same_state_adjacency_fraction(snapshot, grid_width=10, grid_height=10)
    assert result != result  # NaN


def test_same_state_adjacency_fraction_single_agent() -> None:
    """Single agent → no pairs → NaN."""
    snapshot = ((0, 0, 0, 1),)
    result = same_state_adjacency_fraction(snapshot, grid_width=5, grid_height=5)
    assert result != result  # NaN


def test_block_ncd_is_bounded() -> None:
    a = b"ABCD" * 32
    b = b"ABCE" * 32
    ncd = block_ncd(a, b)
    assert 0.0 <= ncd <= 1.0


# --- spatial_scramble_mi tests ---


def test_spatial_scramble_mi_all_same_state() -> None:
    """All agents share a state → scrambling produces identical snapshots → MI unchanged."""
    snapshot = (
        (0, 0, 0, 1),
        (1, 1, 0, 1),
        (2, 0, 1, 1),
        (3, 1, 1, 1),
    )
    result = spatial_scramble_mi(snapshot, 5, 5, n_scrambles=50, rng=random.Random(0))
    observed = neighbor_mutual_information(snapshot, 5, 5)
    # With all-same states, scrambling doesn't change anything
    assert result == pytest.approx(observed, abs=1e-10)


def test_spatial_scramble_mi_empty_snapshot() -> None:
    """Empty snapshot → 0.0."""
    result = spatial_scramble_mi((), 5, 5, n_scrambles=50, rng=random.Random(0))
    assert result == 0.0


def test_spatial_scramble_mi_drops_for_structured() -> None:
    """Structured patterns should have scrambled MI lower than observed MI."""
    # Checkerboard: neighbors always differ
    snapshot = tuple((i * 10 + j, i, j, (i + j) % 2) for i in range(10) for j in range(10))
    observed = neighbor_mutual_information(snapshot, 10, 10)
    scrambled = spatial_scramble_mi(snapshot, 10, 10, n_scrambles=100, rng=random.Random(42))
    assert scrambled < observed


def test_spatial_scramble_mi_deterministic() -> None:
    """Same seed → same result."""
    snapshot = tuple((i * 10 + j, i, j, (i + j) % 4) for i in range(10) for j in range(10))
    r1 = spatial_scramble_mi(snapshot, 10, 10, n_scrambles=50, rng=random.Random(7))
    r2 = spatial_scramble_mi(snapshot, 10, 10, n_scrambles=50, rng=random.Random(7))
    assert r1 == r2


# --- block_shuffle_null_mi tests ---


def test_block_shuffle_null_mi_nonnegative() -> None:
    """Result is non-negative."""
    snapshot = tuple((i * 10 + j, i, j, (i + j) % 4) for i in range(10) for j in range(10))
    result = block_shuffle_null_mi(snapshot, 10, 10, block_size=4, n_shuffles=20, rng=random.Random(0))
    assert result >= 0.0


def test_block_shuffle_null_mi_empty() -> None:
    """Empty snapshot → 0.0."""
    result = block_shuffle_null_mi((), 5, 5, block_size=2, n_shuffles=20, rng=random.Random(0))
    assert result == 0.0


def test_block_shuffle_null_mi_deterministic() -> None:
    """Same seed → same result."""
    snapshot = tuple((i * 10 + j, i, j, (i + j) % 4) for i in range(10) for j in range(10))
    r1 = block_shuffle_null_mi(snapshot, 10, 10, block_size=4, n_shuffles=20, rng=random.Random(3))
    r2 = block_shuffle_null_mi(snapshot, 10, 10, block_size=4, n_shuffles=20, rng=random.Random(3))
    assert r1 == r2


# --- fixed_marginal_null_mi tests ---


def test_fixed_marginal_null_mi_nonnegative() -> None:
    """Result is non-negative."""
    snapshot = tuple((i * 10 + j, i, j, (i + j) % 4) for i in range(10) for j in range(10))
    result = fixed_marginal_null_mi(snapshot, 10, 10, n_samples=20, rng=random.Random(0))
    assert result >= 0.0


def test_fixed_marginal_null_mi_empty() -> None:
    """Empty snapshot → 0.0."""
    result = fixed_marginal_null_mi((), 5, 5, n_samples=20, rng=random.Random(0))
    assert result == 0.0


def test_fixed_marginal_null_mi_nan_on_single_agent() -> None:
    """Single agent → no pairs → 0.0."""
    snapshot = ((0, 2, 2, 1),)
    result = fixed_marginal_null_mi(snapshot, 5, 5, n_samples=20, rng=random.Random(0))
    assert result == 0.0


def test_fixed_marginal_null_mi_deterministic() -> None:
    """Same seed → same result."""
    snapshot = tuple((i * 10 + j, i, j, (i + j) % 4) for i in range(10) for j in range(10))
    r1 = fixed_marginal_null_mi(snapshot, 10, 10, n_samples=20, rng=random.Random(5))
    r2 = fixed_marginal_null_mi(snapshot, 10, 10, n_samples=20, rng=random.Random(5))
    assert r1 == r2


# --- neighbor_transfer_entropy tests ---


def test_transfer_entropy_independent_near_zero() -> None:
    """Independent agents (no neighbor influence) → TE near zero."""
    # Simulate agents with random states, no correlation between neighbors
    rng = random.Random(42)
    sim_log: list[tuple[int, int, int, int, int]] = []  # (step, agent_id, x, y, state)
    n_agents = 10
    for step in range(20):
        for aid in range(n_agents):
            sim_log.append((step, aid, aid, 0, rng.randint(0, 3)))
    te = neighbor_transfer_entropy(sim_log, 10, 1)
    assert te >= 0.0
    assert te < 0.5  # Should be near zero for independent


def test_transfer_entropy_copied_states_high() -> None:
    """When agents copy neighbor states, TE should be elevated."""
    # Construct a scenario where agent state at t+1 = neighbor state at t
    sim_log: list[tuple[int, int, int, int, int]] = []
    # Two agents side by side: agent 0 at (0,0), agent 1 at (1,0)
    # Agent 1's state at t+1 always equals agent 0's state at t
    states_a0 = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    states_a1 = [0] + states_a0[:-1]  # shifted copy of agent 0
    for step in range(20):
        sim_log.append((step, 0, 0, 0, states_a0[step]))
        sim_log.append((step, 1, 1, 0, states_a1[step]))
    te = neighbor_transfer_entropy(sim_log, 5, 5)
    assert te > 0.0


def test_transfer_entropy_empty_log() -> None:
    """Empty simulation log → 0.0."""
    te = neighbor_transfer_entropy([], 5, 5)
    assert te == 0.0
