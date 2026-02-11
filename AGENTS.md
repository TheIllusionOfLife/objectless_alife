# AGENTS.md

Agent-facing repository instructions for `objectless_alife`.

## Scope And Priority

- `spec.md` is the authoritative behavior contract.
- `docs/legacy/*` is historical context only.
- If docs conflict, follow `spec.md` and update other docs in the same change.

## Environment And Tooling

- Python ecosystem uses `uv` only.
- Required Python version: 3.11+
- Install dependencies:

```bash
uv venv
uv sync --extra dev
```

- Run commands via `uv run ...`; do not call global tools directly.

## Non-Obvious Commands

- Batch search (phase 1 baseline):

```bash
uv run python -m src.run_search --phase 1 --n-rules 100 --out-dir data
```

- Experiment mode (phase comparison):

```bash
uv run python -m src.run_search --experiment --phases 1,2 --seed-batches 3 --n-rules 100 --steps 200 --out-dir data
```

- Density sweep mode (explicit grid/agent combinations across both phases):

```bash
uv run python -m src.run_search --density-sweep --grid-sizes 20x20,30x30 --agent-counts 30,60 --seed-batches 2 --n-rules 100 --steps 200 --out-dir data
```

- Visualization CLI:

```bash
uv run python -m src.visualize --simulation-log data/logs/simulation_log.parquet --metrics-summary data/logs/metrics_summary.parquet --rule-json data/rules/<rule_id>.json --output output/preview.gif --fps 8
```

## Code Style And Architecture Rules

- 4-space indentation, type hints on public APIs.
- Keep deterministic behavior by explicit seed handling (`rule_seed`, `sim_seed`).
- Keep world dynamics and metrics decoupled:
  - Simulation logic stays in `src/world.py` and `src/run_search.py`.
  - Metric calculations stay in `src/metrics.py`.
- Keep filters as detectors (decision logic), not as scoring functions.
- Preserve observation phase compatibility:
  - Phase 1 table size: 20
  - Phase 2 table size: 100
- Do not introduce implicit objectives into filtering/selection logic.

## Testing And Validation

Preferred runner and checks:

```bash
uv run ruff check .
uv run ruff format . --check
uv run pytest -q
```

- Add or update tests under `tests/` mirroring source modules.
- For behavior changes, include deterministic seed-based tests.
- Validate both normal run and experiment mode when touching `src/run_search.py`.

## Repository Etiquette

- Create a feature branch before making changes.
  - Recommended prefixes: `feat/`, `fix/`, `chore/`, `docs/`
- Never push directly to `main`.
- Keep commit messages short, imperative, and single-purpose.
- PRs should include:
  - Purpose and scope
  - Design decisions tied to `spec.md`
  - Local verification summary (`ruff`, `pytest`)

## CI Expectations

Current CI (`.github/workflows/ci.yml`) validates:
- `uv sync --extra dev`
- `uv run ruff check .`
- `uv run ruff format . --check`
- `uv run pytest -q`

Mirror these checks locally before opening a PR.

## Developer Environment Quirks

- `data/` and `output/` are generated artifact directories and should remain untracked.
- `src.visualize` defaults `--base-dir` to current directory; absolute paths outside that base are rejected unless base-dir is explicitly set.
- Large experiment workloads are bounded in `src/run_search.py` by `MAX_EXPERIMENT_WORK_UNITS`.

## Common Gotchas

- `state_uniform` is an immediate termination condition; this is intentional.
- `action` in simulation logs records intended action, not movement success.
- Sequential random updates mean early agent updates affect later observations in the same step.
