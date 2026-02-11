# objectless_alife

Objective-free ALife PoC implementation.

## Setup

```bash
uv venv
uv sync --group dev
```

## Run tests

```bash
uv run pytest -q
```

## Run search

```bash
uv run python -m src.run_search --phase 1 --n-rules 100 --out-dir data
```
