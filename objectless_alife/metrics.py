from __future__ import annotations

import math
import random
import statistics
import zlib
from collections import Counter
from typing import Sequence


def state_entropy(states: Sequence[int]) -> float:
    """Compute Shannon entropy (base 2) for discrete agent states."""
    if not states:
        return 0.0

    counts = Counter(states)
    n = len(states)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def compression_ratio(payload: bytes) -> float:
    """Return zlib compressed_size / original_size for payload bytes."""
    if not payload:
        return 0.0
    compressed = zlib.compress(payload)
    return len(compressed) / len(payload)


def normalized_hamming_distance(before: Sequence[int], after: Sequence[int]) -> float:
    """Return normalized Hamming distance in [0, 1] between equal-length vectors."""
    if len(before) != len(after):
        raise ValueError("Vectors must have equal length")
    if not before:
        return 0.0
    distance = sum(1 for b, a in zip(before, after, strict=True) if b != a)
    return distance / len(before)


def serialize_snapshot(
    snapshot: tuple[tuple[int, int, int, int], ...], grid_width: int, grid_height: int
) -> bytes:
    """Serialize snapshot into a flat grid byte buffer with 255 for empty cells."""
    data = bytearray([255] * (grid_width * grid_height))
    for _, x, y, state in snapshot:
        if not (0 <= x < grid_width and 0 <= y < grid_height):
            raise ValueError("Snapshot contains out-of-bounds coordinates")
        idx = y * grid_width + x
        data[idx] = state
    return bytes(data)


def same_state_adjacency_fraction(
    snapshot: tuple[tuple[int, int, int, int], ...], grid_width: int, grid_height: int
) -> float:
    """Fraction of occupied neighbor pairs sharing the same state.

    Returns a value in [0, 1].  Returns NaN when no occupied neighbor pairs
    exist (fewer than 2 occupied cells, or none adjacent).
    """
    occupied = {(x, y): state for _, x, y, state in snapshot}
    same = 0
    total = 0
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for (x, y), state in occupied.items():
        neighbors = (
            ((x + 1) % grid_width, y),
            (x, (y + 1) % grid_height),
        )
        for nx, ny in neighbors:
            if (nx, ny) not in occupied:
                continue
            a = (x, y)
            b = (nx, ny)
            edge = (a, b) if a < b else (b, a)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            total += 1
            if state == occupied[(nx, ny)]:
                same += 1
    if total == 0:
        return float("nan")
    return same / total


def morans_i_occupied(
    snapshot: tuple[tuple[int, int, int, int], ...], grid_width: int, grid_height: int
) -> float:
    """Compute Moran's I across occupied cells using torus 4-neighborhood weights."""
    occupied = {(x, y): state for _, x, y, state in snapshot}
    n = len(occupied)
    if n < 2:
        return float("nan")

    mean_state = sum(occupied.values()) / n
    denominator = sum((state - mean_state) ** 2 for state in occupied.values())
    if denominator == 0.0:
        return float("nan")

    numerator = 0.0
    weight_sum = 0
    for (x, y), state in occupied.items():
        neighbors = (
            (x, (y - 1) % grid_height),
            (x, (y + 1) % grid_height),
            ((x - 1) % grid_width, y),
            ((x + 1) % grid_width, y),
        )
        for nx, ny in neighbors:
            neighbor_state = occupied.get((nx, ny))
            if neighbor_state is None:
                continue
            numerator += (state - mean_state) * (neighbor_state - mean_state)
            weight_sum += 1

    if weight_sum == 0:
        return float("nan")

    return (n / weight_sum) * (numerator / denominator)


def cluster_count_by_state(
    snapshot: tuple[tuple[int, int, int, int], ...], grid_width: int, grid_height: int
) -> int:
    """Count same-state connected components among occupied cells."""
    occupied = {(x, y): state for _, x, y, state in snapshot}
    seen: set[tuple[int, int]] = set()
    clusters = 0

    for start in occupied:
        if start in seen:
            continue
        clusters += 1
        target_state = occupied[start]
        stack = [start]
        seen.add(start)
        while stack:
            x, y = stack.pop()
            neighbors = (
                (x, (y - 1) % grid_height),
                (x, (y + 1) % grid_height),
                ((x - 1) % grid_width, y),
                ((x + 1) % grid_width, y),
            )
            for nx, ny in neighbors:
                if (nx, ny) in seen:
                    continue
                if occupied.get((nx, ny)) != target_state:
                    continue
                seen.add((nx, ny))
                stack.append((nx, ny))

    return clusters


def quasi_periodicity_peak_count(series: Sequence[float]) -> int:
    """Count local maxima in positive-lag autocorrelation."""
    n = len(series)
    if n < 4:
        return 0
    mean = sum(series) / n
    centered = [v - mean for v in series]
    denom = sum(v * v for v in centered)
    if denom == 0.0:
        return 0

    ac = []
    for lag in range(1, (n // 2) + 1):
        num = sum(centered[t] * centered[t - lag] for t in range(lag, n))
        ac.append(num / denom)

    if len(ac) < 3:
        return 0

    threshold = 0.1
    peaks = 0
    for i in range(1, len(ac) - 1):
        if ac[i] > threshold and ac[i] > ac[i - 1] and ac[i] > ac[i + 1]:
            peaks += 1
    return peaks


def phase_transition_max_delta(series: Sequence[float]) -> float:
    """Compute max absolute first-difference in a time series."""
    if len(series) < 2:
        return 0.0
    return max(abs(curr - prev) for prev, curr in zip(series, series[1:], strict=False))


def action_entropy(actions: Sequence[int]) -> float:
    """Compute Shannon entropy for an action sequence."""
    if not actions:
        return 0.0
    counts = Counter(actions)
    n = len(actions)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def action_entropy_variance(per_agent_actions: Sequence[Sequence[int]]) -> float:
    """Compute variance of per-agent action entropy values."""
    if not per_agent_actions:
        return 0.0
    entropies = [action_entropy(actions) for actions in per_agent_actions]
    if len(entropies) < 2:
        return 0.0
    return statistics.pvariance(entropies)


def neighbor_pair_count(
    snapshot: tuple[tuple[int, int, int, int], ...], grid_width: int, grid_height: int
) -> int:
    """Return the number of distinct occupied adjacent pairs in the snapshot."""
    occupied = {(x, y) for _, x, y, _ in snapshot}
    n = 0
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for x, y in occupied:
        for nx, ny in (((x + 1) % grid_width, y), (x, (y + 1) % grid_height)):
            if (nx, ny) not in occupied:
                continue
            edge = ((x, y), (nx, ny)) if (x, y) < (nx, ny) else ((nx, ny), (x, y))
            if edge in seen:
                continue
            seen.add(edge)
            n += 1
    return n


def neighbor_mutual_information(
    snapshot: tuple[tuple[int, int, int, int], ...], grid_width: int, grid_height: int
) -> float:
    """Compute mutual information between occupied neighboring state pairs."""
    occupied = {(x, y): state for _, x, y, state in snapshot}
    pairs: list[tuple[int, int]] = []
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for x, y in occupied:
        neighbors = (
            ((x + 1) % grid_width, y),
            (x, (y + 1) % grid_height),
        )
        for nx, ny in neighbors:
            if (nx, ny) not in occupied:
                continue
            a = (x, y)
            b = (nx, ny)
            edge = (a, b) if a < b else (b, a)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            pairs.append((occupied[a], occupied[b]))

    if not pairs:
        return 0.0

    joint = Counter(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    n = len(pairs)
    mi = 0.0
    for (state_left, state_right), c in joint.items():
        p_ab = c / n
        p_a = left[state_left] / n
        p_b = right[state_right] / n
        mi += p_ab * math.log2(p_ab / (p_a * p_b))
    K_joint = len(joint)
    K_left = len(left)
    K_right = len(right)
    correction = (K_joint - K_left - K_right + 1) / (2 * n * math.log(2))
    return max(mi - correction, 0.0)


def shuffle_null_mi(
    snapshot: tuple[tuple[int, int, int, int], ...],
    grid_width: int,
    grid_height: int,
    n_shuffles: int = 200,
    rng: random.Random | None = None,
) -> float:
    """Compute mean MI under a state-shuffle null (positions fixed, states permuted).

    For each of *n_shuffles* iterations, randomly reassign states among occupied
    positions and compute ``neighbor_mutual_information`` on the shuffled
    snapshot.  Returns the mean MI across all shuffles.  Returns 0.0 when the
    snapshot is empty or *n_shuffles* <= 0.
    """
    if rng is None:
        rng = random.Random()

    positions = [(agent_id, x, y) for agent_id, x, y, _ in snapshot]
    states = [state for _, _, _, state in snapshot]

    if not positions or n_shuffles <= 0:
        return 0.0

    mi_sum = 0.0
    for _ in range(n_shuffles):
        shuffled_states = states.copy()
        rng.shuffle(shuffled_states)
        shuffled_snapshot = tuple(
            (agent_id, x, y, s)
            for (agent_id, x, y), s in zip(positions, shuffled_states, strict=True)
        )
        mi_sum += neighbor_mutual_information(shuffled_snapshot, grid_width, grid_height)

    return mi_sum / n_shuffles


def spatial_scramble_mi(
    snapshot: tuple[tuple[int, int, int, int], ...],
    grid_width: int,
    grid_height: int,
    n_scrambles: int = 200,
    rng: random.Random | None = None,
) -> float:
    """Mean MI after randomly reassigning agent positions (keep states, shuffle positions).

    Tests whether MI depends on genuine local coordination vs incidental
    position arrangement.  Returns 0.0 for empty snapshots.
    """
    if not snapshot or n_scrambles <= 0:
        return 0.0
    if rng is None:
        rng = random.Random()

    agent_ids = [agent_id for agent_id, _, _, _ in snapshot]
    positions = [(x, y) for _, x, y, _ in snapshot]
    states = [state for _, _, _, state in snapshot]

    mi_sum = 0.0
    for _ in range(n_scrambles):
        shuffled_positions = positions.copy()
        rng.shuffle(shuffled_positions)
        scrambled = tuple(
            (aid, x, y, s)
            for aid, (x, y), s in zip(agent_ids, shuffled_positions, states, strict=True)
        )
        mi_sum += neighbor_mutual_information(scrambled, grid_width, grid_height)

    return mi_sum / n_scrambles


def block_shuffle_null_mi(
    snapshot: tuple[tuple[int, int, int, int], ...],
    grid_width: int,
    grid_height: int,
    block_size: int = 4,
    n_shuffles: int = 200,
    rng: random.Random | None = None,
) -> float:
    """Mean MI after shuffling states in spatial blocks.

    Divides the grid into blocks of *block_size* x *block_size* and shuffles
    states among agents within each block, preserving local autocorrelation
    structure at the block level.  Returns 0.0 for empty snapshots.
    """
    if not snapshot or n_shuffles <= 0:
        return 0.0
    if rng is None:
        rng = random.Random()

    occupied = {(x, y): (agent_id, state) for agent_id, x, y, state in snapshot}
    # Assign occupied positions to blocks
    blocks: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x, y in occupied:
        bx = x // block_size
        by = y // block_size
        blocks.setdefault((bx, by), []).append((x, y))

    mi_sum = 0.0
    for _ in range(n_shuffles):
        new_snapshot_dict: dict[tuple[int, int], tuple[int, int]] = {}
        for positions in blocks.values():
            block_states = [occupied[pos][1] for pos in positions]
            rng.shuffle(block_states)
            for pos, s in zip(positions, block_states, strict=True):
                new_snapshot_dict[pos] = (occupied[pos][0], s)
        shuffled = tuple((aid, x, y, s) for (x, y), (aid, s) in new_snapshot_dict.items())
        mi_sum += neighbor_mutual_information(shuffled, grid_width, grid_height)

    return mi_sum / n_shuffles


def fixed_marginal_null_mi(
    snapshot: tuple[tuple[int, int, int, int], ...],
    grid_width: int,
    grid_height: int,
    n_samples: int = 200,
    rng: random.Random | None = None,
) -> float:
    """Mean MI for synthetic snapshots with identical marginal distributions.

    Generates snapshots where each occupied position is independently assigned
    a state drawn from the observed marginal distribution.  This preserves
    marginal frequencies but destroys all spatial dependence.
    Returns 0.0 for empty snapshots.
    """
    if not snapshot or n_samples <= 0:
        return 0.0
    if rng is None:
        rng = random.Random()

    positions = [(agent_id, x, y) for agent_id, x, y, _ in snapshot]
    states = [state for _, _, _, state in snapshot]
    state_counts = Counter(states)
    state_pool = list(state_counts.keys())
    weights = [state_counts[s] for s in state_pool]

    mi_sum = 0.0
    for _ in range(n_samples):
        sampled_states = rng.choices(state_pool, weights=weights, k=len(positions))
        synthetic = tuple(
            (aid, x, y, s) for (aid, x, y), s in zip(positions, sampled_states, strict=True)
        )
        mi_sum += neighbor_mutual_information(synthetic, grid_width, grid_height)

    return mi_sum / n_samples


def neighbor_transfer_entropy(
    sim_log: Sequence[tuple[int, int, int, int, int]],
    grid_width: int,
    grid_height: int,
) -> float:
    """Transfer entropy from neighbor states to agent next-state.

    TE = I(neighbor_state_t ; agent_state_{t+1} | agent_state_t)

    *sim_log* is a sequence of (step, agent_id, x, y, state) tuples.
    Returns 0.0 for empty or insufficient data.
    """
    if not sim_log:
        return 0.0

    # Organize by step
    by_step: dict[int, dict[int, tuple[int, int, int]]] = {}
    for step, agent_id, x, y, state in sim_log:
        by_step.setdefault(step, {})[agent_id] = (x, y, state)

    sorted_steps = sorted(by_step.keys())
    if len(sorted_steps) < 2:
        return 0.0

    # Collect triplets: (agent_state_t, neighbor_state_t, agent_state_{t+1})
    triplets: list[tuple[int, int, int]] = []
    for i in range(len(sorted_steps) - 1):
        t = sorted_steps[i]
        t1 = sorted_steps[i + 1]
        agents_t = by_step[t]
        agents_t1 = by_step[t1]

        pos_to_state_t = {(ax, ay): astate for _, (ax, ay, astate) in agents_t.items()}

        for agent_id, (x, y, state_t) in agents_t.items():
            if agent_id not in agents_t1:
                continue
            state_t1 = agents_t1[agent_id][2]
            # Check von Neumann neighbors at time t
            for nx, ny in (
                ((x + 1) % grid_width, y),
                ((x - 1) % grid_width, y),
                (x, (y + 1) % grid_height),
                (x, (y - 1) % grid_height),
            ):
                neighbor_state = pos_to_state_t.get((nx, ny))
                if neighbor_state is not None and (nx, ny) != (x, y):
                    triplets.append((state_t, neighbor_state, state_t1))

    if not triplets:
        return 0.0

    n = len(triplets)
    # Count distributions for TE = H(Y|X) - H(Y|X,Z)
    # where X=agent_state_t, Z=neighbor_state_t, Y=agent_state_{t+1}
    joint_xyz = Counter(triplets)
    joint_xy = Counter((x, y_val) for x, _, y_val in triplets)
    joint_xz = Counter((x, z) for x, z, _ in triplets)
    count_x = Counter(x for x, _, _ in triplets)

    # TE = sum p(x,z,y) * log2( p(y|x,z) / p(y|x) )
    #    = sum p(x,z,y) * log2( p(x,z,y) * p(x) / (p(x,z) * p(x,y)) )
    te = 0.0
    for (x, z, y_val), c_xyz in joint_xyz.items():
        p_xyz = c_xyz / n
        p_xz = joint_xz[(x, z)] / n
        p_xy = joint_xy[(x, y_val)] / n
        p_x = count_x[x] / n
        if p_xz > 0 and p_xy > 0 and p_x > 0:
            ratio = (p_xyz * p_x) / (p_xz * p_xy)
            if ratio > 0:
                te += p_xyz * math.log2(ratio)

    # Miller-Madow correction
    k_xyz = len(joint_xyz)
    k_xz = len(joint_xz)
    k_xy = len(joint_xy)
    k_x = len(count_x)
    correction = (k_xyz - k_xz - k_xy + k_x) / (2 * n * math.log(2))
    return max(te - correction, 0.0)


def transfer_entropy_shuffle_null(
    sim_log: Sequence[tuple[int, int, int, int, int]],
    grid_width: int,
    grid_height: int,
    n_shuffles: int = 200,
    rng: random.Random | None = None,
) -> float:
    """Mean TE under a null that shuffles next-step states within each step pair."""
    if not sim_log or n_shuffles <= 0:
        return 0.0
    if rng is None:
        rng = random.Random()

    by_step: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for row in sim_log:
        by_step.setdefault(int(row[0]), []).append(row)
    sorted_steps = sorted(by_step.keys())
    if len(sorted_steps) < 2:
        return 0.0

    base_rows: list[tuple[int, int, int, int, int]] = [
        (int(step), int(agent_id), int(x), int(y), int(state))
        for step, agent_id, x, y, state in sim_log
    ]
    # Pre-index target rows once to avoid rebuilding key maps per shuffle.
    next_step_to_row_indices: dict[int, list[int]] = {}
    next_step_to_states: dict[int, list[int]] = {}
    for step in sorted_steps[1:]:
        indices = [idx for idx, row in enumerate(base_rows) if row[0] == step]
        next_step_to_row_indices[step] = indices
        next_step_to_states[step] = [base_rows[idx][4] for idx in indices]

    te_sum = 0.0
    for _ in range(n_shuffles):
        shuffled_rows = list(base_rows)
        for t1 in sorted_steps[1:]:
            shuffled_states = next_step_to_states[t1].copy()
            rng.shuffle(shuffled_states)
            for pos, row_idx in enumerate(next_step_to_row_indices[t1]):
                step, agent_id, x, y, _state = shuffled_rows[row_idx]
                shuffled_rows[row_idx] = (step, agent_id, x, y, shuffled_states[pos])
        te_sum += neighbor_transfer_entropy(shuffled_rows, grid_width, grid_height)
    return te_sum / n_shuffles


def transfer_entropy_excess(
    sim_log: Sequence[tuple[int, int, int, int, int]],
    grid_width: int,
    grid_height: int,
    n_shuffles: int = 200,
    rng: random.Random | None = None,
) -> float:
    """Non-negative TE excess: observed TE minus shuffle-null TE."""
    observed = neighbor_transfer_entropy(sim_log, grid_width, grid_height)
    null = transfer_entropy_shuffle_null(
        sim_log,
        grid_width,
        grid_height,
        n_shuffles=n_shuffles,
        rng=rng,
    )
    return max(observed - null, 0.0)


def block_ncd(left: bytes, right: bytes) -> float:
    """Compute normalized compression distance for two byte blocks."""
    if not left and not right:
        return 0.0

    c_left = len(zlib.compress(left))
    c_right = len(zlib.compress(right))
    c_join = len(zlib.compress(left + right))
    denom = max(c_left, c_right)
    if denom == 0:
        return 0.0
    value = (c_join - min(c_left, c_right)) / denom
    return max(0.0, min(1.0, value))
