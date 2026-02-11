# Repository Guidelines

## Project Structure & Module Organization
This repository is currently spec-driven. Core documents at the root are:
- `spec.md`: authoritative implementation spec (use this as source of truth)
- `project_onboarding_guide_objective_free_alife.md`: original proposal/context
- `unified_review.md`: consolidated review and rationale for spec changes

When implementing the PoC, follow the module split defined in the spec:
- `world.py`, `rules.py`, `filters.py`, `metrics.py`, `run_search.py`
- Tests should mirror modules under `tests/` (for example, `tests/test_world.py`).

## Build, Test, and Development Commands
Use `uv` for Python environment and tooling.
- `uv venv && uv sync`: create/sync local environment from project metadata
- `uv run pytest -q`: run test suite
- `uv run ruff check .`: lint
- `uv run ruff format .`: format code

If project metadata is not yet present, add it before implementation and keep commands unchanged.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints on public APIs, small pure functions where possible.
- Naming: `snake_case` for files/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep simulation logic deterministic via explicit seeds (`rule_seed`, `sim_seed`) and pass RNG dependencies explicitly.
- Prefer simple data structures first; only add abstractions when repeated patterns emerge.

## Testing Guidelines
Follow TDD: write tests first, confirm failure, implement, then refactor.
- Framework: `pytest`
- Test file names: `test_<module>.py`
- Test names: `test_<behavior>_<condition>`
- Include deterministic seed-based tests and edge cases (halt detection, state uniformity, collision handling).

## Commit & Pull Request Guidelines
Commit messages in this repo are short, imperative, and specific (for example, `Add implementation spec and cross-reference all documents`).
- Keep subject lines concise and action-oriented.
- One logical change per commit.
- PRs should include: purpose, key design decisions, test evidence (`uv run pytest` output summary), and related issue/spec section.
- Do not push to `main`; use feature branches such as `feat/<topic>` or `docs/<topic>`.

## Agent-Specific Notes
Do not treat proposal/review docs as normative when they conflict with `spec.md`. Implement behavior from `spec.md` first, then update docs/tests together.
