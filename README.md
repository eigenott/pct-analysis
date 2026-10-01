# PCT 2026 — Hike Data Analysis

> **TL;DR (placeholder — final overview stats go here).**
> - Trail miles hiked: *TBD*
> - Hiking days / zero days: *TBD*
> - Biggest day: *TBD*
> - Total ascent / descent: *TBD*
> - Average miles per hiking day: *TBD*
> - Start → end: Campo → Rainy Pass (final ~61 mi closed, Ptarmigan fire)

## What this is

Turning my Forerunner 255 recordings from the 2026 PCT thru-hike into
cleaned data and visualizations (marimo + polars). Start with the
interactive notebook (`notebooks/pct.py`); how the data was cleaned lives
in [`docs/methodology.md`](docs/methodology.md).

## Run it

```bash
uv sync
uv run python scripts/split_fit_by_date.py   # needs the Garmin export in rawdata/
uv run python scripts/build_tracks.py
uv run python scripts/flag_jumps.py
uv run python scripts/compare_heuristics.py
uv run python scripts/apply_cuts.py
uv run python scripts/build_route.py         # needs GPX in rawdata/pcta_route/
uv run python scripts/project_tracks.py
uv run marimo edit notebooks/pct.py
```

`rawdata/` (Garmin export, GPX) and `data/` (generated Parquet) are local-only
and gitignored — scripts rebuild everything from raw.

## Research report

- **Interactive (recommended):** `uv run marimo export html notebooks/pct.py -o report.html`
  and host it on the blog — all charts stay interactive.
- **GitHub viewing:** `uv run marimo export ipynb notebooks/pct.py -o pct.ipynb` —
  GitHub renders `.ipynb` with static plots (it does *not* render marimo `.py` files).
- **Print/archive:** `uv run marimo export pdf notebooks/pct.py -o report.pdf`.

## Layout

```
scripts/          pipeline (sort → parse → flag → compare → cut → route → project)
notebooks/pct.py  marimo analysis notebook
docs/             methodology + decisions
rawdata/          (untracked) Garmin export, PCTA GPX
data/             (untracked) generated Parquet tables
```
