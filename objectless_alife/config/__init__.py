"""Configuration layer: constants and typed config dataclasses.

This package re-exports all symbols that were previously defined in the flat
``objectless_alife/config.py`` module, preserving backward compatibility for
all existing import sites.
"""

from objectless_alife.config.constants import (
    ACTION_SPACE_SIZE,
    BLOCK_NCD_WINDOW,
    CLOCK_PERIOD,
    FLUSH_THRESHOLD,
    GRID_HEIGHT,
    GRID_WIDTH,
    HALT_WINDOW,
    MAX_EXPERIMENT_WORK_UNITS,
    NUM_AGENTS,
    NUM_STATES,
    NUM_STEPS,
    SHUFFLE_NULL_N,
)
from objectless_alife.config.types import (
    DensitySweepConfig,
    ExperimentConfig,
    FilterConfig,
    HaltWindowSweepConfig,
    MetricComputeConfig,
    MultiSeedConfig,
    RuntimeConfig,
    SearchConfig,
    SimulationResult,
    StateUniformMode,
    UpdateMode,
)

__all__ = [
    "ACTION_SPACE_SIZE",
    "BLOCK_NCD_WINDOW",
    "CLOCK_PERIOD",
    "DensitySweepConfig",
    "ExperimentConfig",
    "FilterConfig",
    "FLUSH_THRESHOLD",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "HALT_WINDOW",
    "HaltWindowSweepConfig",
    "MAX_EXPERIMENT_WORK_UNITS",
    "MetricComputeConfig",
    "MultiSeedConfig",
    "NUM_AGENTS",
    "NUM_STATES",
    "NUM_STEPS",
    "RuntimeConfig",
    "SHUFFLE_NULL_N",
    "SearchConfig",
    "SimulationResult",
    "StateUniformMode",
    "UpdateMode",
]
