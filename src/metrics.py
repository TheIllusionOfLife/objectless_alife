from __future__ import annotations

import math
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
    K = len(joint)  # number of non-zero bins in joint distribution
    correction = (K - 1) / (2 * n * math.log(2))
    return max(mi - correction, 0.0)


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
