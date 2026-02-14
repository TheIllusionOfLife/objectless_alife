# STRUCTURE.md

## Repository Layout

- `spec.md`: canonical implementation specification
- `README.md`: human-facing quickstart and usage
- `AGENTS.md`: agent-specific implementation/etiquette guidance
- `PRODUCT.md`: product intent and user/value framing
- `TECH.md`: stack and constraints
- `STRUCTURE.md`: this document
- `docs/legacy/`: archived historical proposal/review docs
- `docs/stage_b_results.md`: Stage B experimental results
- `docs/stage_c_results.md`: Stage C experimental results
- `paper/`: ALIFE conference paper draft (LaTeX) and figures
- `src/`: application source modules
- `tests/`: test modules mirroring `src/`
- `.github/workflows/`: CI and automation workflows

## Source Module Responsibilities

- `src/world.py`: world model, agents, movement/collision mechanics, step updates
- `src/rules.py`: observation phase enums, table indexing, seeded rule generation
- `src/filters.py`: termination/dynamic filter detectors
- `src/metrics.py`: simulation analysis metrics
- `src/run_search.py`: batch and experiment orchestration, artifact persistence
- `src/stats.py`: statistical significance testing, pairwise comparisons, effect sizes
- `src/visualize.py`: animation rendering from stored artifacts

## Test Organization

- Test files follow `tests/test_<module>.py`
- Each source module has a corresponding test module
- Prefer deterministic tests with explicit seeds and focused behavior assertions

## Naming Conventions

- Files/functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

## Import And Dependency Patterns

- Keep imports explicit by module responsibility
- Avoid cross-module leakage of concerns (for example, metrics logic in world module)
- Keep utility functions near their owning domain unless shared by multiple modules

## Directory Hygiene Rules

- Root should contain only active project entry docs/configs
- Generated artifacts go to `data/` and `output/` and remain untracked
- Historical/context-only docs belong under `docs/legacy/`
