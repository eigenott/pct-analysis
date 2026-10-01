# AGENTS.md

## Toolchain

- Python 3.14 (pinned in `.python-version`), managed with **uv** — use `uv sync`, `uv run`, `uv add`. Do not use pip or create venvs manually.
- Build backend is `uv_build`; the package uses a `src/` layout (`src/pct_analysis/`).

## Commands

- Run the CLI entry point: `uv run pct-analysis` (wired to `pct_analysis:main` in `pyproject.toml`).
- No test, lint, or typecheck tooling is configured yet — do not assume pytest/ruff/mypy exist. If adding tests, add dev dependencies via `uv add --dev`.

## Notes

- `marimo` is a core dependency — interactive marimo notebooks are likely part of the intended workflow, not just a one-off.
- README is empty; project intent is not documented in-repo.
