"""Tests for experiments/density_sweep.py: config validation and path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from objectless_alife.config.types import DensitySweepConfig
from objectless_alife.experiments.density_sweep import _validate_density_sweep_config


def _make_config(**overrides: object) -> DensitySweepConfig:
    defaults: dict[str, object] = {
        "grid_sizes": ((20, 20),),
        "agent_counts": (30,),
        "n_rules": 1,
        "n_seed_batches": 1,
        "steps": 10,
        "out_dir": Path("data"),
    }
    defaults.update(overrides)
    return DensitySweepConfig(**defaults)  # type: ignore[arg-type]


def test_valid_config_passes() -> None:
    _validate_density_sweep_config(_make_config())


def test_zero_rules_raises() -> None:
    with pytest.raises(ValueError, match="n_rules"):
        _validate_density_sweep_config(_make_config(n_rules=0))


def test_zero_seed_batches_raises() -> None:
    with pytest.raises(ValueError, match="n_seed_batches"):
        _validate_density_sweep_config(_make_config(n_seed_batches=0))


def test_zero_steps_raises() -> None:
    with pytest.raises(ValueError, match="steps"):
        _validate_density_sweep_config(_make_config(steps=0))


def test_empty_grid_sizes_raises() -> None:
    with pytest.raises(ValueError, match="grid_sizes"):
        _validate_density_sweep_config(_make_config(grid_sizes=()))


def test_empty_agent_counts_raises() -> None:
    with pytest.raises(ValueError, match="agent_counts"):
        _validate_density_sweep_config(_make_config(agent_counts=()))


def test_excessive_workload_raises() -> None:
    # Large combination that exceeds MAX_EXPERIMENT_WORK_UNITS
    with pytest.raises(ValueError, match="workload exceeds"):
        _validate_density_sweep_config(
            _make_config(
                grid_sizes=tuple((w, w) for w in range(10, 110, 10)),  # 10 sizes
                agent_counts=tuple(range(10, 60, 10)),  # 5 counts
                n_rules=10_000,
                n_seed_batches=10,
                steps=1_000,
            )
        )
