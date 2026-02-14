# TECH.md

## Language And Runtime

- Python 3.11+

## Package And Environment Management

- `uv` for virtual environment and dependency management
- Project metadata in `pyproject.toml`
- Locked dependency graph in `uv.lock`

## Core Libraries

- `pyarrow`: Parquet IO and tabular artifact persistence
- `matplotlib`: animation and visual output rendering
- `scipy`: statistical significance tests (Mann-Whitney U, chi-squared)

## Development Tooling

- `pytest`: test runner
- `ruff`: lint and formatting
- GitHub Actions: CI execution (`.github/workflows/ci.yml`)

## Canonical Commands

- Setup: `uv venv && uv sync --extra dev`
- Lint: `uv run ruff check .`
- Format check: `uv run ruff format . --check`
- Tests: `uv run pytest -q`

## Technical Constraints

- Determinism through explicit seeds and controlled RNG usage
- Rule table sizes are phase-defined and fixed by spec
- Simulation outputs are written as JSON + Parquet
- Filters must not encode hidden optimization objectives
- Experiment mode enforces a maximum work-unit bound for runtime safety

## CI Compatibility

Local development should reproduce CI checks exactly to keep mainline green.
