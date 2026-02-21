"""CLI entrypoint for search execution.

After refactoring, this thin module owns only the CLI argument parsing
and mode dispatch.  All domain logic lives in the extracted modules:

- ``objectless_alife.schemas``      – Parquet schemas & metric-name constants
- ``objectless_alife.config``       – configuration dataclasses
- ``objectless_alife.simulation``   – ``run_batch_search`` engine
- ``objectless_alife.aggregation``  – experiment / density-sweep / multi-seed orchestration

For backward compatibility every public symbol is re-exported here so
that existing ``from objectless_alife.run_search import X`` imports keep working.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-exports for backward compatibility
# ---------------------------------------------------------------------------
from objectless_alife.aggregation import (  # noqa: F401
    _build_phase_comparison,
    _build_phase_summary,
    _collect_final_metric_rows,
    run_density_sweep,
    run_experiment,
    run_halt_window_sweep,
    run_multi_seed_robustness,
    select_top_rules_by_delta_mi,
)
from objectless_alife.config import (  # noqa: F401
    MAX_EXPERIMENT_WORK_UNITS,
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
from objectless_alife.rules import ObservationPhase
from objectless_alife.schemas import (  # noqa: F401
    AGGREGATE_SCHEMA_VERSION,
    DENSITY_PHASE_COMPARISON_SCHEMA,
    DENSITY_PHASE_SUMMARY_SCHEMA,
    DENSITY_SWEEP_RUNS_SCHEMA,
    DENSITY_SWEEP_SCHEMA_VERSION,
    HALT_WINDOW_SWEEP_SCHEMA,
    METRICS_SCHEMA,
    MULTI_SEED_SCHEMA,
    PHASE_SUMMARY_METRIC_NAMES,
    SIMULATION_SCHEMA,
)
from objectless_alife.simulation import _entropy_from_action_counts, run_batch_search  # noqa: F401

# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------


def _parse_phase(raw_phase: int) -> ObservationPhase:
    """Parse CLI phase value into ObservationPhase enum."""
    try:
        return ObservationPhase(raw_phase)
    except ValueError as exc:
        valid = ", ".join(str(p.value) for p in ObservationPhase)
        raise ValueError(f"phase must be one of {valid}") from exc


def _parse_phase_list(raw_phases: str) -> tuple[ObservationPhase, ...]:
    """Parse comma-delimited phase list and require at least two entries."""
    parts = [part.strip() for part in raw_phases.split(",") if part.strip()]
    if len(parts) < 2:
        raise ValueError("phases must contain at least two values")

    phases: list[ObservationPhase] = []
    for part in parts:
        try:
            phase_raw = int(part)
        except ValueError as exc:
            valid = ", ".join(str(p.value) for p in ObservationPhase)
            raise ValueError(f"invalid phase value; must be one of {valid}") from exc
        phase = _parse_phase(phase_raw)
        phases.append(phase)

    if len(set(phases)) != len(phases):
        raise ValueError("phases must include distinct values")

    return tuple(phases)


def _parse_grid_sizes(raw_grid_sizes: str) -> tuple[tuple[int, int], ...]:
    """Parse comma-delimited grid sizes formatted as `WxH`."""
    parts = [part.strip() for part in raw_grid_sizes.split(",") if part.strip()]
    if not parts:
        raise ValueError("grid-sizes must not be empty")

    grid_sizes: list[tuple[int, int]] = []
    for part in parts:
        tokens = part.lower().split("x")
        if len(tokens) != 2:
            raise ValueError("grid-sizes entries must use WxH format")
        width_raw, height_raw = tokens
        try:
            width = int(width_raw)
            height = int(height_raw)
        except ValueError as exc:
            raise ValueError("grid-sizes entries must use integer WxH values") from exc
        if width < 1 or height < 1:
            raise ValueError("grid-sizes entries must be >= 1x1")
        grid_sizes.append((width, height))
    return tuple(grid_sizes)


def _parse_positive_int_csv(raw_values: str, label: str) -> tuple[int, ...]:
    """Parse comma-delimited positive integers."""
    parts = [part.strip() for part in raw_values.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{label} must not be empty")

    values: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"{label} must contain integers") from exc
        if value < 1:
            raise ValueError(f"{label} values must be >= 1")
        values.append(value)
    return tuple(values)


def _parse_update_mode(raw_update_mode: str) -> UpdateMode:
    """Parse update mode from CLI/config."""
    try:
        return UpdateMode(raw_update_mode)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in UpdateMode)
        raise ValueError(f"update-mode must be one of {valid}") from exc


def _parse_state_uniform_mode(raw_state_uniform_mode: str) -> StateUniformMode:
    """Parse state-uniform handling mode from CLI/config."""
    try:
        return StateUniformMode(raw_state_uniform_mode)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in StateUniformMode)
        raise ValueError(f"state-uniform-mode must be one of {valid}") from exc


def _coerce_bool(raw: object, key: str) -> bool:
    """Coerce raw value to bool with strict string-check."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{key} must be a boolean value")


def _coerce_int(raw: object, key: str) -> int:
    """Coerce raw value to int; rejects booleans."""
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be an integer value")
    if isinstance(raw, (int, float, str, bytes, bytearray)):
        return int(raw)
    if hasattr(raw, "__int__"):
        return int(raw)
    raise ValueError(f"{key} must be an integer value")


def _coerce_float(raw: object, key: str) -> float:
    """Coerce raw value to float; rejects booleans."""
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be a float value")
    if isinstance(raw, (int, float, str, bytes, bytearray)):
        return float(raw)
    if hasattr(raw, "__float__"):
        return float(raw)
    raise ValueError(f"{key} must be a float value")


def _coerce_str(raw: object, key: str) -> str:
    """Coerce raw value to str; rejects booleans."""
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be a string-coercible value")
    if isinstance(raw, (str, Path, int, float)):
        return str(raw)
    raise ValueError(f"{key} must be a string-coercible value")


def _get_val(cli_val: object, key: str, file_cfg: dict[str, object], default: object) -> object:
    """CLI > file > default resolution."""
    if cli_val is not None:
        return cli_val
    return file_cfg.get(key, default)


def _get_bool(cli_val: bool | None, key: str, file_cfg: dict[str, object], default: bool) -> bool:
    """CLI > file > default resolution for boolean flags."""
    return _coerce_bool(_get_val(cli_val, key, file_cfg, default), key)


def _get_int(cli_val: int | None, key: str, file_cfg: dict[str, object], default: int) -> int:
    """CLI > file > default resolution for integer values."""
    return _coerce_int(_get_val(cli_val, key, file_cfg, default), key)


def _get_float(
    cli_val: float | None, key: str, file_cfg: dict[str, object], default: float
) -> float:
    """CLI > file > default resolution for float values."""
    return _coerce_float(_get_val(cli_val, key, file_cfg, default), key)


def _get_str(cli_val: str | None, key: str, file_cfg: dict[str, object], default: str) -> str:
    """CLI > file > default resolution for string values."""
    return _coerce_str(_get_val(cli_val, key, file_cfg, default), key)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run objective-free ALife search")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file (CLI args override file values)",
    )
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--n-rules", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--halt-window", type=int, default=None)
    parser.add_argument(
        "--update-mode",
        type=str,
        choices=[mode.value for mode in UpdateMode],
        default=None,
    )
    parser.add_argument(
        "--enable-viability-filters",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--state-uniform-mode",
        type=str,
        choices=[mode.value for mode in StateUniformMode],
        default=None,
    )
    parser.add_argument("--rule-seed", type=int, default=None)
    parser.add_argument("--sim-seed", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--density-sweep", action=argparse.BooleanOptionalAction, default=None)
    mode_group.add_argument("--experiment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--grid-sizes", type=str, default=None)
    parser.add_argument("--agent-counts", type=str, default=None)
    parser.add_argument("--seed-batches", type=int, default=None)
    parser.add_argument("--phases", type=str, default=None)
    parser.add_argument(
        "--filter-short-period", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--short-period-max-period", type=int, default=None)
    parser.add_argument("--short-period-history-size", type=int, default=None)
    parser.add_argument(
        "--filter-low-activity", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--low-activity-window", type=int, default=None)
    parser.add_argument("--low-activity-min-unique-ratio", type=float, default=None)
    parser.add_argument("--block-ncd-window", type=int, default=None)
    parser.add_argument(
        "--fast-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip expensive null-model computations (shuffle_null_mi)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for search execution.

    Supports ``--config path/to/config.json`` for experiment reproducibility.
    CLI arguments override config-file values; config-file values override
    built-in defaults.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load config file defaults (CLI overrides file, file overrides built-in)
    file_cfg: dict[str, object] = {}
    if args.config is not None:
        file_cfg = json.loads(Path(args.config).read_text())

    phase_raw = _get_int(args.phase, "phase", file_cfg, 1)
    n_rules = _get_int(args.n_rules, "n_rules", file_cfg, 100)
    steps = _get_int(args.steps, "steps", file_cfg, 200)
    halt_window = _get_int(args.halt_window, "halt_window", file_cfg, 10)
    update_mode_raw = _get_str(
        args.update_mode, "update_mode", file_cfg, UpdateMode.SEQUENTIAL.value
    )
    update_mode = _parse_update_mode(update_mode_raw)
    enable_viability_filters = _get_bool(
        args.enable_viability_filters, "enable_viability_filters", file_cfg, True
    )
    state_uniform_mode_raw = _get_str(
        args.state_uniform_mode, "state_uniform_mode", file_cfg, StateUniformMode.TERMINAL.value
    )
    state_uniform_mode = _parse_state_uniform_mode(state_uniform_mode_raw)
    rule_seed = _get_int(args.rule_seed, "rule_seed", file_cfg, 0)
    sim_seed = _get_int(args.sim_seed, "sim_seed", file_cfg, 0)
    out_dir = Path(_get_str(args.out_dir, "out_dir", file_cfg, "data"))
    grid_sizes_raw = _get_str(args.grid_sizes, "grid_sizes", file_cfg, "20x20")
    agent_counts_raw = _get_str(args.agent_counts, "agent_counts", file_cfg, "30")
    seed_batches = _get_int(args.seed_batches, "seed_batches", file_cfg, 1)
    phases_raw = _get_str(args.phases, "phases", file_cfg, "1,2")
    filter_short_period = _get_bool(
        args.filter_short_period, "filter_short_period", file_cfg, False
    )
    short_period_max_period = _get_int(
        args.short_period_max_period, "short_period_max_period", file_cfg, 2
    )
    short_period_history_size = _get_int(
        args.short_period_history_size, "short_period_history_size", file_cfg, 8
    )
    filter_low_activity = _get_bool(
        args.filter_low_activity, "filter_low_activity", file_cfg, False
    )
    low_activity_window = _get_int(args.low_activity_window, "low_activity_window", file_cfg, 5)
    low_activity_min_unique_ratio = _get_float(
        args.low_activity_min_unique_ratio,
        "low_activity_min_unique_ratio",
        file_cfg,
        0.2,
    )
    block_ncd_window = _get_int(args.block_ncd_window, "block_ncd_window", file_cfg, 10)
    fast_metrics = _get_bool(args.fast_metrics, "fast_metrics", file_cfg, False)
    is_density_sweep = _get_bool(args.density_sweep, "density_sweep", file_cfg, False)
    is_experiment = _get_bool(args.experiment, "experiment", file_cfg, False)

    if is_density_sweep and is_experiment:
        raise ValueError(
            "--density-sweep and --experiment cannot both be enabled; "
            "disable one via CLI (--no-density-sweep / --no-experiment) "
            "or in the config file"
        )

    if is_density_sweep:
        density_sweep_config = DensitySweepConfig(
            grid_sizes=_parse_grid_sizes(grid_sizes_raw),
            agent_counts=_parse_positive_int_csv(agent_counts_raw, "agent-counts"),
            n_rules=n_rules,
            n_seed_batches=seed_batches,
            out_dir=out_dir,
            steps=steps,
            halt_window=halt_window,
            enable_viability_filters=enable_viability_filters,
            update_mode=update_mode,
            state_uniform_mode=state_uniform_mode,
            rule_seed_start=rule_seed,
            sim_seed_start=sim_seed,
            filter_short_period=filter_short_period,
            short_period_max_period=short_period_max_period,
            short_period_history_size=short_period_history_size,
            filter_low_activity=filter_low_activity,
            low_activity_window=low_activity_window,
            low_activity_min_unique_ratio=low_activity_min_unique_ratio,
            block_ncd_window=block_ncd_window,
            skip_null_models=fast_metrics,
        )
        results = run_density_sweep(density_sweep_config)
        summary = {
            "mode": "density_sweep",
            "phases": [1, 2],
            "density_points": len(density_sweep_config.grid_sizes)
            * len(density_sweep_config.agent_counts),
            "total_rules": len(results),
            "survived": sum(1 for r in results if r.survived),
            "terminated": sum(1 for r in results if not r.survived),
        }
    elif is_experiment:
        experiment_config = ExperimentConfig(
            phases=_parse_phase_list(phases_raw),
            n_rules=n_rules,
            n_seed_batches=seed_batches,
            out_dir=out_dir,
            steps=steps,
            halt_window=halt_window,
            enable_viability_filters=enable_viability_filters,
            update_mode=update_mode,
            state_uniform_mode=state_uniform_mode,
            rule_seed_start=rule_seed,
            sim_seed_start=sim_seed,
            filter_short_period=filter_short_period,
            short_period_max_period=short_period_max_period,
            short_period_history_size=short_period_history_size,
            filter_low_activity=filter_low_activity,
            low_activity_window=low_activity_window,
            low_activity_min_unique_ratio=low_activity_min_unique_ratio,
            block_ncd_window=block_ncd_window,
            skip_null_models=fast_metrics,
        )
        results = run_experiment(experiment_config)
        summary = {
            "experiment": True,
            "phases": [phase.value for phase in experiment_config.phases],
            "seed_batches": experiment_config.n_seed_batches,
            "total_rules": len(results),
            "survived": sum(1 for r in results if r.survived),
            "terminated": sum(1 for r in results if not r.survived),
        }
    else:
        phase = _parse_phase(phase_raw)
        search_config = SearchConfig(
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
            skip_null_models=fast_metrics,
        )
        results = run_batch_search(
            n_rules=n_rules,
            phase=phase,
            out_dir=out_dir,
            base_rule_seed=rule_seed,
            base_sim_seed=sim_seed,
            config=search_config,
        )

        summary = {
            "experiment": False,
            "phase": phase.value,
            "total_rules": len(results),
            "survived": sum(1 for r in results if r.survived),
            "terminated": sum(1 for r in results if not r.survived),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
