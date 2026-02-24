"""Spatial metrics: Moran's I, clustering, adjacency, neighbor-pair counting."""

from __future__ import annotations


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
