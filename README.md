# objectless_alife

Objective-free artificial life (ALife) proof-of-concept for exploring emergent structure without optimization targets.

## What This Repository Contains

- Deterministic, seed-driven grid-world simulation with shared rule tables
- Two observation phases for comparative experiments
- Physical inconsistency filters (halt/state-uniform) plus optional dynamic filters for ablations
- Metrics and Parquet/JSON output pipelines
- Animation rendering for inspecting individual rule trajectories

The implementation source of truth is `spec.md`.

## Quick Start

Requirements:
- Python 3.11+
- `uv`

Setup:

```bash
uv venv
uv sync --extra dev
```

Run quality checks:

```bash
uv run ruff check .
uv run ruff format . --check
uv run pytest -q
```

## Common Commands

Run a single-phase batch search:

```bash
uv run python -m src.run_search --phase 1 --n-rules 100 --out-dir data
```

Run a two-phase experiment comparison:

```bash
uv run python -m src.run_search \
  --experiment \
  --phases 1,2 \
  --seed-batches 3 \
  --n-rules 100 \
  --steps 200 \
  --out-dir data
```

Run a density sweep across explicit grid/agent points (both phases):

```bash
uv run python -m src.run_search \
  --density-sweep \
  --grid-sizes 20x20,30x30 \
  --agent-counts 30,60 \
  --seed-batches 2 \
  --n-rules 100 \
  --steps 200 \
  --out-dir data
```

Render an animation from generated artifacts:

```bash
uv run python -m src.visualize \
  --simulation-log data/logs/simulation_log.parquet \
  --metrics-summary data/logs/metrics_summary.parquet \
  --rule-json data/rules/<rule_id>.json \
  --output output/preview.gif \
  --fps 8
```

## Documentation Map

- `spec.md`: canonical implementation spec
- `AGENTS.md`: agent-facing repository instructions
- `PRODUCT.md`: product intent and research goals
- `TECH.md`: stack and technical constraints
- `STRUCTURE.md`: codebase layout and conventions
- `docs/legacy/`: archived context/review docs kept for traceability

## High-Level Architecture

- `src/world.py`: toroidal world model, agent state, collision/movement semantics
- `src/rules.py`: observation phases, indexing logic, seeded rule-table generation
- `src/filters.py`: termination and optional dynamic filter detectors
- `src/metrics.py`: post-step analysis metrics
- `src/run_search.py`: batch/experiment runner + artifact persistence
- `src/visualize.py`: animation renderer from stored artifacts

## Data Outputs

By default, runs produce:
- `data/rules/*.json`: per-rule metadata, filter outcomes, seeds
- `data/logs/simulation_log.parquet`: per-agent per-step state/action logs
- `data/logs/metrics_summary.parquet`: per-step metric summaries
- `data/logs/experiment_runs.parquet`: per-rule aggregate outcomes (experiment mode)
- `data/logs/phase_summary.parquet`: per-phase aggregates (experiment mode)
- `data/logs/phase_comparison.json`: phase delta summary (experiment mode)
- `data/logs/density_sweep_runs.parquet`: per-rule aggregate outcomes (density sweep mode)
- `data/logs/density_phase_summary.parquet`: per-density/per-phase aggregates (density sweep mode)
- `data/logs/density_phase_comparison.parquet`: phase deltas for each density point (density sweep mode)

## Development Workflow

- Branch from `main` before changes (for example `feat/<topic>`, `fix/<topic>`, `chore/<topic>`)
- Keep commits focused and imperative
- Run lint + format-check + tests locally before opening PRs
- Do not treat `docs/legacy/*` as normative when it conflicts with `spec.md`
