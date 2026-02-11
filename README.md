# objectless_alife

Objective-free ALife PoC implementation.

## Setup

Requires Python 3.11+.

```bash
uv venv
uv sync --extra dev
```

## Run tests

```bash
uv run pytest -q
```

## Quality gates

Run the same checks as CI before opening a PR:

```bash
uv run ruff check .
uv run ruff format . --check
uv run pytest -q
```

## Run search

```bash
uv run python -m src.run_search --phase 1 --n-rules 100 --out-dir data
```

Optional dynamic filters (default off):

```bash
uv run python -m src.run_search \
  --phase 2 \
  --n-rules 100 \
  --filter-short-period \
  --short-period-max-period 2 \
  --filter-low-activity \
  --low-activity-window 5 \
  --low-activity-min-unique-ratio 0.2 \
  --out-dir data
```

## Run experiment mode (phase comparison)

Run Phase 1 and Phase 2 across multiple seed batches and generate aggregate comparison files:

```bash
uv run python -m src.run_search \
  --experiment \
  --phases 1,2 \
  --seed-batches 3 \
  --n-rules 100 \
  --steps 200 \
  --out-dir data
```

`--phases` currently accepts exactly two distinct values (for example `1,2`).
For safety, very large workloads are rejected when `phases * n_rules * seed_batches * steps` exceeds an internal threshold.

Experiment outputs:

- `data/phase_1/` and `data/phase_2/`: per-phase rule JSON + simulation/metrics parquet
- `data/logs/experiment_runs.parquet`: per-rule run outcomes across phases
- `data/logs/phase_summary.parquet`: per-phase aggregate statistics
- `data/logs/phase_comparison.json`: phase-to-phase absolute/relative deltas

## Throughput guidance for large runs

For stable long runs, scale up in stages and keep each invocation within a practical work budget.

- Work-unit formula: `len(phases) * n_rules * seed_batches * steps`
- Current safety threshold: `100_000_000` work units (runs above this are rejected)
- Recommended progression:
  - Debug: `--n-rules 100 --seed-batches 1`
  - Exploration: `--n-rules 1000 --seed-batches 1-3`
  - Large sweep: split into multiple invocations by seed ranges instead of one giant run
- Practical tip: keep `--out-dir` per campaign (for example `data/exp_2026_02_11/`) to avoid mixing artifacts from different parameter sets.

## Render animation

First run a search so `data/rules/*.json` and `data/logs/*.parquet` exist.

```bash
uv run python -c "from src.visualize import render_rule_animation; \
from pathlib import Path; \
rule_json = next(Path('data/rules').glob('*.json')); \
render_rule_animation(Path('data/logs/simulation_log.parquet'), Path('data/logs/metrics_summary.parquet'), rule_json, Path('output/preview.gif'), fps=8)"
```

You can force explicit world bounds if needed:

```bash
uv run python -m src.visualize \
  --simulation-log data/logs/simulation_log.parquet \
  --metrics-summary data/logs/metrics_summary.parquet \
  --rule-json data/rules/<rule_id>.json \
  --output output/preview.gif \
  --fps 8 \
  --grid-width 20 \
  --grid-height 20
```

## Spec Coverage Matrix

- `spec.md` section 2-6: `src/world.py`, `src/rules.py`, `src/filters.py`
- `spec.md` section 7: `src/metrics.py`, `src/run_search.py`
- `spec.md` section 9: JSON + Parquet outputs in `src/run_search.py`
- `spec.md` section 10: `src/visualize.py`
