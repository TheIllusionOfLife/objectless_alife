from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Sequence


def state_entropy(states: Sequence[int]) -> float:
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
    if not payload:
        return 0.0
    compressed = zlib.compress(payload)
    return len(compressed) / len(payload)


def normalized_hamming_distance(before: Sequence[int], after: Sequence[int]) -> float:
    if len(before) != len(after):
        raise ValueError("Vectors must have equal length")
    if not before:
        return 0.0
    distance = sum(1 for b, a in zip(before, after, strict=False) if b != a)
    return distance / len(before)


def serialize_snapshot(
    snapshot: tuple[tuple[int, int, int, int], ...], grid_width: int, grid_height: int
) -> bytes:
    data = bytearray([255] * (grid_width * grid_height))
    for _, x, y, state in snapshot:
        idx = y * grid_width + x
        data[idx] = state
    return bytes(data)
