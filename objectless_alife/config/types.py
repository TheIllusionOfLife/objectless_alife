"""Configuration dataclasses and safety constants for simulation experiments.

All frozen dataclasses that parameterise search, experiment, density-sweep,
multi-seed, and halt-window-sweep runs live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from objectless_alife.config.constants import MAX_EXPERIMENT_WORK_UNITS
from objectless_alife.domain.rules import ObservationPhase

# Re-export the safety constant so callers can import from config or config.types
__all__ = [
    "MAX_EXPERIMENT_WORK_UNITS",
    "SimulationResult",
    "UpdateMode",
    "StateUniformMode",
    "RuntimeConfig",
    "FilterConfig",
    "MetricComputeConfig",
    "SearchConfig",
    "ExperimentConfig",
    "DensitySweepConfig",
    "MultiSeedConfig",
    "HaltWindowSweepConfig",
]

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """Top-level result for one evaluated rule table."""

    rule_id: str
    survived: bool
    terminated_at: int | None
    termination_reason: str | None


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


class UpdateMode(Enum):
    """Agent update semantics for one simulation step."""

    SEQUENTIAL = "sequential"
    SYNCHRONOUS = "synchronous"


class StateUniformMode(Enum):
    """Handling policy when all agents share the same state."""

    TERMINAL = "terminal"
    TAG_ONLY = "tag_only"


@dataclass(frozen=True)
class RuntimeConfig:
    """Core runtime knobs shared by search/experiment/sweep runs."""

    steps: int = 200
    halt_window: int = 10
    enable_viability_filters: bool = True
    update_mode: UpdateMode = UpdateMode.SEQUENTIAL
    state_uniform_mode: StateUniformMode = StateUniformMode.TERMINAL

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.halt_window < 1:
            raise ValueError("halt_window must be >= 1")


@dataclass(frozen=True)
class FilterConfig:
    """Optional dynamic-filter settings."""

    filter_short_period: bool = False
    short_period_max_period: int = 2
    short_period_history_size: int = 8
    filter_low_activity: bool = False
    low_activity_window: int = 5
    low_activity_min_unique_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.short_period_max_period < 1:
            raise ValueError("short_period_max_period must be >= 1")
        if self.short_period_history_size < 1:
            raise ValueError("short_period_history_size must be >= 1")
        if self.short_period_history_size < self.short_period_max_period * 2:
            raise ValueError("short_period_history_size must be >= 2 * short_period_max_period")
        if self.low_activity_window < 1:
            raise ValueError("low_activity_window must be >= 1")
        if not 0.0 <= self.low_activity_min_unique_ratio <= 1.0:
            raise ValueError("low_activity_min_unique_ratio must be in [0.0, 1.0]")


@dataclass(frozen=True)
class MetricComputeConfig:
    """Expensive metric-computation controls."""

    block_ncd_window: int = 10
    shuffle_null_n_shuffles: int = 200
    skip_null_models: bool = False

    def __post_init__(self) -> None:
        if self.block_ncd_window < 0:
            raise ValueError("block_ncd_window must be >= 0")
        if self.shuffle_null_n_shuffles < 1:
            raise ValueError("shuffle_null_n_shuffles must be >= 1")


@dataclass(frozen=True)
class SearchConfig:
    """Batch-search runtime parameters including optional dynamic filters."""

    steps: int = 200
    halt_window: int = 10
    enable_viability_filters: bool = True
    update_mode: UpdateMode = UpdateMode.SEQUENTIAL
    state_uniform_mode: StateUniformMode = StateUniformMode.TERMINAL
    filter_short_period: bool = False
    short_period_max_period: int = 2
    short_period_history_size: int = 8
    filter_low_activity: bool = False
    low_activity_window: int = 5
    low_activity_min_unique_ratio: float = 0.2
    block_ncd_window: int = 10
    shuffle_null_n_shuffles: int = 200
    skip_null_models: bool = False

    def __post_init__(self) -> None:
        RuntimeConfig(
            steps=self.steps,
            halt_window=self.halt_window,
            enable_viability_filters=self.enable_viability_filters,
            update_mode=self.update_mode,
            state_uniform_mode=self.state_uniform_mode,
        )
        FilterConfig(
            filter_short_period=self.filter_short_period,
            short_period_max_period=self.short_period_max_period,
            short_period_history_size=self.short_period_history_size,
            filter_low_activity=self.filter_low_activity,
            low_activity_window=self.low_activity_window,
            low_activity_min_unique_ratio=self.low_activity_min_unique_ratio,
        )
        MetricComputeConfig(
            block_ncd_window=self.block_ncd_window,
            shuffle_null_n_shuffles=self.shuffle_null_n_shuffles,
            skip_null_models=self.skip_null_models,
        )

    @classmethod
    def from_components(
        cls,
        runtime: RuntimeConfig | None = None,
        filters: FilterConfig | None = None,
        metrics: MetricComputeConfig | None = None,
    ) -> "SearchConfig":
        """Compose SearchConfig from reusable sub-config components."""
        runtime = runtime or RuntimeConfig()
        filters = filters or FilterConfig()
        metrics = metrics or MetricComputeConfig()
        return cls(
            steps=runtime.steps,
            halt_window=runtime.halt_window,
            enable_viability_filters=runtime.enable_viability_filters,
            update_mode=runtime.update_mode,
            state_uniform_mode=runtime.state_uniform_mode,
            filter_short_period=filters.filter_short_period,
            short_period_max_period=filters.short_period_max_period,
            short_period_history_size=filters.short_period_history_size,
            filter_low_activity=filters.filter_low_activity,
            low_activity_window=filters.low_activity_window,
            low_activity_min_unique_ratio=filters.low_activity_min_unique_ratio,
            block_ncd_window=metrics.block_ncd_window,
            shuffle_null_n_shuffles=metrics.shuffle_null_n_shuffles,
            skip_null_models=metrics.skip_null_models,
        )

    def to_components(self) -> tuple[RuntimeConfig, FilterConfig, MetricComputeConfig]:
        """Decompose SearchConfig into reusable sub-config components."""
        return (
            RuntimeConfig(
                steps=self.steps,
                halt_window=self.halt_window,
                enable_viability_filters=self.enable_viability_filters,
                update_mode=self.update_mode,
                state_uniform_mode=self.state_uniform_mode,
            ),
            FilterConfig(
                filter_short_period=self.filter_short_period,
                short_period_max_period=self.short_period_max_period,
                short_period_history_size=self.short_period_history_size,
                filter_low_activity=self.filter_low_activity,
                low_activity_window=self.low_activity_window,
                low_activity_min_unique_ratio=self.low_activity_min_unique_ratio,
            ),
            MetricComputeConfig(
                block_ncd_window=self.block_ncd_window,
                shuffle_null_n_shuffles=self.shuffle_null_n_shuffles,
                skip_null_models=self.skip_null_models,
            ),
        )


def _search_config_from_legacy_fields(
    *,
    steps: int,
    halt_window: int,
    enable_viability_filters: bool,
    update_mode: UpdateMode,
    state_uniform_mode: StateUniformMode,
    filter_short_period: bool,
    short_period_max_period: int,
    short_period_history_size: int,
    filter_low_activity: bool,
    low_activity_window: int,
    low_activity_min_unique_ratio: float,
    block_ncd_window: int,
    skip_null_models: bool,
) -> SearchConfig:
    """Build SearchConfig from legacy flattened dataclass fields."""
    return SearchConfig(
        steps=steps,
        halt_window=halt_window,
        enable_viability_filters=enable_viability_filters,
        update_mode=update_mode,
        state_uniform_mode=state_uniform_mode,
        filter_short_period=filter_short_period,
        short_period_max_period=short_period_max_period,
        short_period_history_size=short_period_history_size,
        filter_low_activity=filter_low_activity,
        low_activity_window=low_activity_window,
        low_activity_min_unique_ratio=low_activity_min_unique_ratio,
        block_ncd_window=block_ncd_window,
        skip_null_models=skip_null_models,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment-scale runtime settings for multi-phase, multi-seed evaluation."""

    phases: tuple[ObservationPhase, ...] = (
        ObservationPhase.PHASE1_DENSITY,
        ObservationPhase.PHASE2_PROFILE,
    )
    n_rules: int = 100
    n_seed_batches: int = 1
    out_dir: Path = Path("data")
    steps: int = 200
    halt_window: int = 10
    enable_viability_filters: bool = True
    update_mode: UpdateMode = UpdateMode.SEQUENTIAL
    state_uniform_mode: StateUniformMode = StateUniformMode.TERMINAL
    rule_seed_start: int = 0
    sim_seed_start: int = 0
    filter_short_period: bool = False
    short_period_max_period: int = 2
    short_period_history_size: int = 8
    filter_low_activity: bool = False
    low_activity_window: int = 5
    low_activity_min_unique_ratio: float = 0.2
    block_ncd_window: int = 10
    skip_null_models: bool = False
    search_config: SearchConfig | None = None

    def __post_init__(self) -> None:
        RuntimeConfig(
            steps=self.steps,
            halt_window=self.halt_window,
            enable_viability_filters=self.enable_viability_filters,
            update_mode=self.update_mode,
            state_uniform_mode=self.state_uniform_mode,
        )
        FilterConfig(
            filter_short_period=self.filter_short_period,
            short_period_max_period=self.short_period_max_period,
            short_period_history_size=self.short_period_history_size,
            filter_low_activity=self.filter_low_activity,
            low_activity_window=self.low_activity_window,
            low_activity_min_unique_ratio=self.low_activity_min_unique_ratio,
        )
        MetricComputeConfig(
            block_ncd_window=self.block_ncd_window,
            skip_null_models=self.skip_null_models,
        )
        if self.search_config is None:
            return
        legacy = self._legacy_search_config()
        if legacy != self.search_config:
            raise ValueError(
                "search_config conflicts with ExperimentConfig legacy fields; "
                "use search_config only or keep fields consistent"
            )

    def _legacy_search_config(self) -> SearchConfig:
        return _search_config_from_legacy_fields(
            steps=self.steps,
            halt_window=self.halt_window,
            enable_viability_filters=self.enable_viability_filters,
            update_mode=self.update_mode,
            state_uniform_mode=self.state_uniform_mode,
            filter_short_period=self.filter_short_period,
            short_period_max_period=self.short_period_max_period,
            short_period_history_size=self.short_period_history_size,
            filter_low_activity=self.filter_low_activity,
            low_activity_window=self.low_activity_window,
            low_activity_min_unique_ratio=self.low_activity_min_unique_ratio,
            block_ncd_window=self.block_ncd_window,
            skip_null_models=self.skip_null_models,
        )

    def resolved_search_config(self) -> SearchConfig:
        if self.search_config is not None:
            return self.search_config
        return self._legacy_search_config()


@dataclass(frozen=True)
class DensitySweepConfig:
    """Runtime settings for grid/agent density sweeps across both phases."""

    grid_sizes: tuple[tuple[int, int], ...] = ((20, 20),)
    agent_counts: tuple[int, ...] = (30,)
    n_rules: int = 100
    n_seed_batches: int = 1
    out_dir: Path = Path("data")
    steps: int = 200
    halt_window: int = 10
    enable_viability_filters: bool = True
    update_mode: UpdateMode = UpdateMode.SEQUENTIAL
    state_uniform_mode: StateUniformMode = StateUniformMode.TERMINAL
    rule_seed_start: int = 0
    sim_seed_start: int = 0
    filter_short_period: bool = False
    short_period_max_period: int = 2
    short_period_history_size: int = 8
    filter_low_activity: bool = False
    low_activity_window: int = 5
    low_activity_min_unique_ratio: float = 0.2
    block_ncd_window: int = 10
    skip_null_models: bool = False
    search_config: SearchConfig | None = None

    def __post_init__(self) -> None:
        RuntimeConfig(
            steps=self.steps,
            halt_window=self.halt_window,
            enable_viability_filters=self.enable_viability_filters,
            update_mode=self.update_mode,
            state_uniform_mode=self.state_uniform_mode,
        )
        FilterConfig(
            filter_short_period=self.filter_short_period,
            short_period_max_period=self.short_period_max_period,
            short_period_history_size=self.short_period_history_size,
            filter_low_activity=self.filter_low_activity,
            low_activity_window=self.low_activity_window,
            low_activity_min_unique_ratio=self.low_activity_min_unique_ratio,
        )
        MetricComputeConfig(
            block_ncd_window=self.block_ncd_window,
            skip_null_models=self.skip_null_models,
        )
        if self.search_config is None:
            return
        legacy = self._legacy_search_config()
        if legacy != self.search_config:
            raise ValueError(
                "search_config conflicts with DensitySweepConfig legacy fields; "
                "use search_config only or keep fields consistent"
            )

    def _legacy_search_config(self) -> SearchConfig:
        return _search_config_from_legacy_fields(
            steps=self.steps,
            halt_window=self.halt_window,
            enable_viability_filters=self.enable_viability_filters,
            update_mode=self.update_mode,
            state_uniform_mode=self.state_uniform_mode,
            filter_short_period=self.filter_short_period,
            short_period_max_period=self.short_period_max_period,
            short_period_history_size=self.short_period_history_size,
            filter_low_activity=self.filter_low_activity,
            low_activity_window=self.low_activity_window,
            low_activity_min_unique_ratio=self.low_activity_min_unique_ratio,
            block_ncd_window=self.block_ncd_window,
            skip_null_models=self.skip_null_models,
        )

    def resolved_search_config(self) -> SearchConfig:
        if self.search_config is not None:
            return self.search_config
        return self._legacy_search_config()


@dataclass(frozen=True)
class MultiSeedConfig:
    """Settings for multi-seed robustness evaluation of selected rules."""

    rule_seeds: tuple[int, ...]
    n_sim_seeds: int = 20
    out_dir: Path = Path("data/multi_seed")
    steps: int = 200
    halt_window: int = 10
    enable_viability_filters: bool = True
    update_mode: UpdateMode = UpdateMode.SEQUENTIAL
    state_uniform_mode: StateUniformMode = StateUniformMode.TERMINAL
    phase: ObservationPhase = ObservationPhase.PHASE2_PROFILE
    shuffle_null_n_shuffles: int = 200


@dataclass(frozen=True)
class HaltWindowSweepConfig:
    """Settings for halt-window sensitivity analysis."""

    rule_seeds: tuple[int, ...]
    halt_windows: tuple[int, ...] = (5, 10, 20)
    out_dir: Path = Path("data/halt_window_sweep")
    steps: int = 200
    enable_viability_filters: bool = True
    update_mode: UpdateMode = UpdateMode.SEQUENTIAL
    state_uniform_mode: StateUniformMode = StateUniformMode.TERMINAL
    phase: ObservationPhase = ObservationPhase.PHASE2_PROFILE
    shuffle_null_n_shuffles: int = 200

    def __post_init__(self) -> None:
        if not self.rule_seeds:
            raise ValueError("rule_seeds must not be empty")
        if not self.halt_windows:
            raise ValueError("halt_windows must not be empty")
        if any(w < 1 for w in self.halt_windows):
            raise ValueError("halt_windows values must be >= 1")
