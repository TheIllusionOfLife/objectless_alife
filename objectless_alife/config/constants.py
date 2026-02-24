"""Centralized domain constants for simulation experiments.

All magic numbers that appear across multiple modules are defined here.
Consuming modules should import from this module rather than defining
their own inline literals.
"""

from __future__ import annotations

GRID_WIDTH = 20
"""Default grid width in cells."""

GRID_HEIGHT = 20
"""Default grid height in cells."""

NUM_AGENTS = 30
"""Default number of agents per simulation."""

NUM_STEPS = 200
"""Default number of simulation steps."""

NUM_STATES = 4
"""Number of discrete agent internal states."""

HALT_WINDOW = 10
"""Default halt-detector window (consecutive unchanged snapshots)."""

SHUFFLE_NULL_N = 200
"""Default number of shuffles for the MI shuffle-null model."""

BLOCK_NCD_WINDOW = 10
"""Default block window size for NCD metric computation."""

CLOCK_PERIOD = 5
"""Number of distinct step-clock values for the control phase."""

ACTION_SPACE_SIZE = 9
"""Total action count: 4 movement + 4 state-change + 1 no-op."""

FLUSH_THRESHOLD = 8_192
"""Flush simulation log rows to Parquet once this in-memory row count is reached."""

MAX_EXPERIMENT_WORK_UNITS = 100_000_000
"""Safety cap on total simulation steps across all phases/rules/seeds."""
