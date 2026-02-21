"""Tests for scripts/shuffle_null_convergence.py — all synthetic, no Parquet data."""

from __future__ import annotations

import math
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import scripts.shuffle_null_convergence as module  # noqa: E402


def _make_synthetic_snapshots(
    n_snapshots: int = 3, n_agents: int = 5, grid: int = 5, seed: int = 0
) -> dict[str, tuple[tuple[int, int, int, int], ...]]:
    rng = random.Random(seed)
    result: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
    for i in range(n_snapshots):
        positions: set[tuple[int, int]] = set()
        agents: list[tuple[int, int, int, int]] = []
        while len(agents) < n_agents:
            x, y = rng.randrange(grid), rng.randrange(grid)
            if (x, y) not in positions:
                positions.add((x, y))
                agents.append((i * 100 + len(agents), x, y, rng.randrange(4)))
        result[f"rule_{i}"] = tuple(agents)
    return result


def test_convergence_analysis_smoke():
    snapshots = _make_synthetic_snapshots()
    result = module.run_convergence_analysis(
        snapshots, n_values=[5, 10, 20], seed=42, grid_width=5, grid_height=5
    )
    assert set(result.keys()) == {5, 10, 20}
    for _n, stats in result.items():
        assert "mean" in stats and "std" in stats
        assert math.isfinite(stats["mean"])
        assert math.isfinite(stats["std"])


def test_convergence_stability():
    snapshots = _make_synthetic_snapshots(n_snapshots=5)
    result = module.run_convergence_analysis(
        snapshots, n_values=[5, 10, 20, 50], seed=42, grid_width=5, grid_height=5
    )
    means = [result[n]["mean"] for n in [5, 10, 20, 50]]
    assert statistics.stdev(means) < 0.1, f"std of means {statistics.stdev(means):.4f} >= 0.1"


def test_convergence_analysis_empty_snapshots():
    """run_convergence_analysis returns nan for empty snapshots (no data to report)."""
    result = module.run_convergence_analysis(
        {}, n_values=[5, 10], seed=42, grid_width=5, grid_height=5
    )
    assert set(result.keys()) == {5, 10}
    for stats in result.values():
        assert math.isnan(stats["mean"])
        assert math.isnan(stats["std"])


def test_convergence_analysis_single_snapshot():
    """run_convergence_analysis: single snapshot has valid mean, nan std."""
    snapshots = _make_synthetic_snapshots(n_snapshots=1)
    result = module.run_convergence_analysis(
        snapshots, n_values=[5, 10], seed=42, grid_width=5, grid_height=5
    )
    assert set(result.keys()) == {5, 10}
    for stats in result.values():
        assert math.isfinite(stats["mean"])
        assert math.isnan(stats["std"])


def test_plot_convergence(tmp_path: Path):
    snapshots = _make_synthetic_snapshots()
    n_vals = [5, 10]
    result = module.run_convergence_analysis(
        snapshots, n_values=n_vals, seed=42, grid_width=5, grid_height=5
    )
    out = tmp_path / "test_fig.pdf"
    module.plot_convergence(result, out)
    assert out.exists()
    assert out.stat().st_size > 0
