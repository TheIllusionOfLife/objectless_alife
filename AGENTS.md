# AGENTS.md

Agent-facing repository instructions for `objectless_alife`.

## Scope And Priority

- `spec.md` is the authoritative behavior contract.
- `docs/legacy/*` is historical context only.
- If docs conflict, follow `spec.md` and update other docs in the same change.

## Environment And Tooling

- Python ecosystem uses `uv` only.
- LaTeX compilation uses `tectonic` (not pdflatex/latexmk).
- Overfull `\hbox` warnings are not accepted. Fix column widths, line breaks, or wording until tectonic reports no overfull lines.
- Required Python version: 3.11+
- Install dependencies:

```bash
uv venv
uv sync --extra dev
```

- Run commands via `uv run ...`; do not call global tools directly.
- Compile paper:

```bash
tectonic paper/main.tex
```

## Non-Obvious Commands

- Batch search (phase 1 baseline):

```bash
uv run python -m objectless_alife.run_search --phase 1 --n-rules 100 --out-dir data
```

- Experiment mode (phase comparison):

```bash
uv run python -m objectless_alife.run_search --experiment --phases 1,2 --seed-batches 3 --n-rules 100 --steps 200 --out-dir data
```

- Density sweep mode (explicit grid/agent combinations across both phases):

```bash
uv run python -m objectless_alife.run_search --density-sweep --grid-sizes 20x20,30x30 --agent-counts 30,60 --seed-batches 2 --n-rules 100 --steps 200 --out-dir data
```

- Visualization CLI (subcommands: `single`, `batch`, `figure`, `filmstrip`):

```bash
uv run python -m objectless_alife.visualize single --simulation-log data/logs/simulation_log.parquet --metrics-summary data/logs/metrics_summary.parquet --rule-json data/rules/<rule_id>.json --output output/preview.gif --fps 8
uv run python -m objectless_alife.visualize batch --phase-dir P1=data/stage_b/phase_1 --phase-dir P2=data/stage_b/phase_2 --top-n 5 --output-dir output/batch
uv run python -m objectless_alife.visualize figure --p1-dir data/stage_b/phase_1 --p2-dir data/stage_b/phase_2 --control-dir data/stage_c/control --output-dir output/figures
uv run python -m objectless_alife.visualize filmstrip --simulation-log data/logs/simulation_log.parquet --rule-json data/rules/<rule_id>.json --output output/filmstrip.png --n-frames 8
```

- Statistical significance testing:

```bash
uv run python -m objectless_alife.stats --data-dir data/stage_b
uv run python -m objectless_alife.stats --pairwise --dir-a data/stage_b --dir-b data/stage_c
```

## Code Style And Architecture Rules

- 4-space indentation, type hints on public APIs.
- Keep deterministic behavior by explicit seed handling (`rule_seed`, `sim_seed`).
- Keep world dynamics and metrics decoupled:
  - Simulation logic stays in `objectless_alife/world.py`, `objectless_alife/simulation.py`, and `objectless_alife/run_search.py`.
  - Metric calculations stay in `objectless_alife/metrics.py`.
- Keep filters as detectors (decision logic), not as scoring functions.
- Preserve observation phase compatibility:
  - Phase 1 table size: 20
  - Phase 2 table size: 100
  - Phase 3 (Control) table size: 100
  - Phase 4 (Random Walk) table size: 1
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
- Validate both normal run and experiment mode when touching `objectless_alife/run_search.py`.

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
- `objectless_alife.visualize` defaults `--base-dir` to current directory; absolute paths outside that base are rejected unless base-dir is explicitly set.
- Large experiment workloads are bounded in `objectless_alife/config.py` by `MAX_EXPERIMENT_WORK_UNITS`.

## Common Gotchas

- `state_uniform` is an immediate termination condition; this is intentional.
- `action` in simulation logs records intended action, not movement success.
- Sequential random updates mean early agent updates affect later observations in the same step.
- GIFs in output directory is too big to read. Don't try to read them.
- **Path security** — validate ALL path inputs against `base_dir`; sanitize components from user input or data files with `Path(user_input).name`.
- **Empty collection guards** — always check non-empty before aggregation (`max`, `min`, `median`); provide graceful degradation (NaN, skip).
- **PyArrow performance** — use `pyarrow.compute` functions, not Python loops with `.as_py()`; filter/aggregate on Arrow tables before `.to_pylist()`.
- **No magic numbers** — extract domain-specific values (action space size, state count) to named module-level constants.
- **Explicit phase handling** — handle each `ObservationPhase` variant explicitly; use `raise ValueError` in else, never silent fallthrough.
- **Formula verification** — cross-check statistical formulas against cited references; ensure paper claims match actual data.
- **Non-square grid safety** — use `grid_height` for y-axis and `grid_width` for x-axis separately; never assume square.
- **Defensive test coverage** — always test empty/degenerate inputs and boundary cases for new functions.
