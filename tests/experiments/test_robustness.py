"""Tests for experiments/robustness.py: config construction and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from objectless_alife.config.types import HaltWindowSweepConfig, MultiSeedConfig
from objectless_alife.domain.rules import ObservationPhase


def test_multi_seed_config_defaults() -> None:
    cfg = MultiSeedConfig(rule_seeds=(1, 2, 3))
    assert cfg.n_sim_seeds == 20
    assert cfg.steps == 200
    assert cfg.halt_window == 10
    assert cfg.phase == ObservationPhase.PHASE2_PROFILE


def test_multi_seed_config_custom_phase() -> None:
    cfg = MultiSeedConfig(
        rule_seeds=(42,),
        phase=ObservationPhase.PHASE1_DENSITY,
        n_sim_seeds=5,
    )
    assert cfg.phase == ObservationPhase.PHASE1_DENSITY
    assert cfg.n_sim_seeds == 5


def test_multi_seed_config_is_frozen() -> None:
    cfg = MultiSeedConfig(rule_seeds=(1,))
    with pytest.raises((AttributeError, TypeError)):
        cfg.n_sim_seeds = 99  # type: ignore[misc]


def test_halt_window_sweep_config_defaults() -> None:
    cfg = HaltWindowSweepConfig(rule_seeds=(1, 2))
    assert cfg.halt_windows == (5, 10, 20)
    assert cfg.steps == 200
    assert cfg.phase == ObservationPhase.PHASE2_PROFILE


def test_halt_window_sweep_config_is_frozen() -> None:
    cfg = HaltWindowSweepConfig(rule_seeds=(1,))
    with pytest.raises((AttributeError, TypeError)):
        cfg.halt_windows = (3,)  # type: ignore[misc]


def test_run_multi_seed_robustness_writes_parquet(tmp_path: Path) -> None:
    """run_multi_seed_robustness returns the path to the output Parquet file."""
    import pyarrow.parquet as pq

    from objectless_alife.experiments.robustness import run_multi_seed_robustness

    cfg = MultiSeedConfig(
        rule_seeds=(0,),
        n_sim_seeds=2,
        out_dir=tmp_path,
        steps=5,
        halt_window=2,
        phase=ObservationPhase.PHASE1_DENSITY,
        shuffle_null_n_shuffles=5,
    )
    out_path = run_multi_seed_robustness(cfg)
    assert isinstance(out_path, Path)
    assert out_path.exists()
    table = pq.read_table(out_path)
    assert "rule_seed" in table.schema.names
