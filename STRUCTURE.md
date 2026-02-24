# STRUCTURE.md

## Repository Layout

- `spec.md`: canonical implementation specification
- `README.md`: human-facing quickstart and usage
- `AGENTS.md`: agent-specific implementation/etiquette guidance
- `PRODUCT.md`: product intent and user/value framing
- `TECH.md`: stack and constraints
- `STRUCTURE.md`: this document
- `run_search.py`: root-level shim → `objectless_alife.experiments.search.main`
- `visualize.py`: root-level shim → `objectless_alife.viz.cli.main`
- `docs/legacy/`: archived historical proposal/review docs
- `docs/stage_b_results.md`: Stage B experimental results
- `docs/stage_c_results.md`: Stage C experimental results
- `paper/`: ALIFE conference paper draft (LaTeX) and figures
- `objectless_alife/`: application source package (layered subpackages)
- `tests/`: test modules
- `.github/workflows/`: CI and automation workflows

## Source Package Structure

The package is organized into focused subpackages. Old flat modules remain as
backward-compatible re-export shims.

### `objectless_alife/config/`
- `constants.py`: all domain constants (`GRID_WIDTH`, `NUM_AGENTS`, `CLOCK_PERIOD`, `ACTION_SPACE_SIZE`, `FLUSH_THRESHOLD`, etc.)
- `types.py`: frozen configuration dataclasses (`SearchConfig`, `ExperimentConfig`, `WorldConfig`, `UpdateMode`, `StateUniformMode`, etc.)

### `objectless_alife/domain/`
- `snapshot.py`: `AgentState` frozen dataclass + `Snapshot` type alias
- `rules.py`: `ObservationPhase` enum, index computation functions, `generate_rule_table`
- `filters.py`: `HaltDetector`, `StateUniformDetector`, `ShortPeriodDetector`, `LowActivityDetector`, `TerminationReason`
- `world.py`: `World`, `WorldConfig`, `Agent` — toroidal grid simulation mechanics

### `objectless_alife/metrics/`
- `spatial.py`: `morans_i_occupied`, `cluster_count_by_state`, `same_state_adjacency_fraction`, `neighbor_pair_count`
- `temporal.py`: `quasi_periodicity_peak_count`, `phase_transition_max_delta`, `action_entropy`, `action_entropy_variance`
- `information.py`: `state_entropy`, `compression_ratio`, `neighbor_mutual_information`, `shuffle_null_mi`, `block_ncd`, transfer entropy variants, and null-model helpers

### `objectless_alife/io/`
- `schemas.py`: Parquet schemas, schema version constants, `PHASE_SUMMARY_METRIC_NAMES` (single source of truth)
- `paths.py`: path construction helpers for experiment output directories

### `objectless_alife/analysis/`
- `stats.py`: Mann-Whitney U, bootstrap CI, Holm-Bonferroni, point-biserial, `load_final_step_metrics`, `run_statistical_analysis`

### `objectless_alife/experiments/`
- `summaries.py`: `collect_final_metric_rows`, phase summary/comparison builders
- `selection.py`: `select_top_rules_by_delta_mi`
- `experiment.py`: `run_experiment` orchestration
- `density_sweep.py`: `run_density_sweep`, density phase helpers
- `robustness.py`: `run_multi_seed_robustness`, `run_halt_window_sweep`
- `search.py`: CLI argument parsing and `main()` entrypoint

### `objectless_alife/simulation/`
- `engine.py`: `run_batch_search` — orchestrates the simulation loop
- `step.py`: per-step metric computation helpers (`compute_step_metrics`, `entropy_from_action_counts`, `mean_and_pvariance`)
- `persistence.py`: `flush_sim_columns` — Parquet writer helper

### `objectless_alife/viz/`
- `theme.py`: `Theme` frozen dataclass, `DEFAULT_THEME`, `PAPER_THEME`, `get_theme`
- `render.py`: Matplotlib rendering functions (`render_rule_animation`, `render_batch`, `render_snapshot_grid`, `render_metric_distribution`, `render_metric_timeseries`, `render_filmstrip`)
- `cli.py`: CLI argument parsing and subcommand dispatch (`main`, handler functions)
- `export_web.py`: web JSON export (`export_single`, `export_paired`, `export_batch`, `export_gallery`)

### Backward-Compat Shims (flat files)
The following flat files are retained as re-export shims for existing import paths:
`world.py`, `rules.py`, `filters.py`, `metrics.py`, `schemas.py`, `config.py` (→ `config/`),
`stats.py`, `aggregation.py`, `simulation.py`, `run_search.py`,
`viz_cli.py`, `viz_render.py`, `viz_theme.py`, `export_web.py`, `visualize.py`

## Test Organization

- Test files follow `tests/test_<module>.py`
- Each source module has a corresponding test module
- Prefer deterministic tests with explicit seeds and focused behavior assertions

## Naming Conventions

- Files/functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

## Import And Dependency Patterns

- Import from subpackage modules directly (e.g., `from objectless_alife.domain.rules import ObservationPhase`)
- All domain constants live in `config/constants.py` — never define them inline
- Schemas and metric-name lists are the single source of truth in `io/schemas.py`
- Old flat module paths still work via re-export shims; prefer new subpackage paths in new code

## Directory Hygiene Rules

- Root should contain only active project entry docs/configs
- Generated artifacts go to `data/` and `output/` and remain untracked
- Historical/context-only docs belong under `docs/legacy/`
