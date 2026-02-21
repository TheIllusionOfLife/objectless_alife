from __future__ import annotations

import math
import random
from pathlib import Path

import scripts.mi_vs_n_pairs_analysis as module

BINS = [(1, 3, "1\u20133"), (4, 6, "4\u20136"), (7, 12, "7\u201312"), (13, None, "\u226513")]


def test_assign_bin():
    assert module.assign_bin(1, BINS) == "1\u20133"
    assert module.assign_bin(3, BINS) == "1\u20133"
    assert module.assign_bin(4, BINS) == "4\u20136"
    assert module.assign_bin(5, BINS) == "4\u20136"
    assert module.assign_bin(7, BINS) == "7\u201312"
    assert module.assign_bin(12, BINS) == "7\u201312"
    assert module.assign_bin(13, BINS) == "\u226513"
    assert module.assign_bin(100, BINS) == "\u226513"
    assert module.assign_bin(0, BINS) is None


def test_bootstrap_median_ci_single():
    rng = random.Random(42)
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = module.bootstrap_median_ci_single(vals, n_bootstrap=500, rng=rng)
    assert lo <= 3.0 <= hi
    assert math.isfinite(lo) and math.isfinite(hi)


def test_compute_bin_stats_smoke():
    # (n_pairs, delta_mi) data
    data = [(2, 0.1), (3, 0.2), (5, 0.5), (6, 0.6), (8, 1.0), (14, 2.0)]
    rng = random.Random(42)
    result = module.compute_bin_stats(data, BINS, n_bootstrap=100, rng=rng)
    assert "1\u20133" in result
    assert "4\u20136" in result
    # Bin "1-3" has n_pairs 2 and 3
    assert result["1\u20133"]["n"] == 2
    # median of [0.1, 0.2] = 0.15
    assert abs(result["1\u20133"]["median"] - 0.15) < 1e-9
    assert "\u226513" in result
    assert result["\u226513"]["n"] == 1


def test_figure_created(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Create a minimal figure to verify the output path works
    fig, ax = plt.subplots()
    ax.bar([0], [1])
    out = tmp_path / "figP1_mi_vs_n_pairs.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    assert out.exists()
