# PCT 2026 — Hike Data Analysis

> **TL;DR for hikers (no code needed).** I thru-hiked the Pacific Crest Trail
> in 2026 (Apr 25 – Aug 11) with a Garmin Forerunner 255 recording every day
> in battery-saving UltraTrac mode. That mode saves battery by taking sparse
> GPS fixes — and sometimes the GPS "jumped" miles sideways, inflating daily
> mileage by up to 30%. This project cleans those jumps out of the data and
> visualizes the hike: **2,442.8 trail miles** after cleaning (my own
> estimate was ~2,450 — off by 0.3%). The hike ended at **Rainy Pass**
> because the final ~61 miles were closed by the Ptarmigan fire, with two
> more sections skipped for fire closures and catching up with my tramily.
> Biggest day: **41 miles** on Jul 21 (yes, really). Scroll the notebook
> plots below — or open the interactive version linked under
> *Research report*.

## What this is

A modern Python data project (marimo + polars) that turns 10,515 raw Garmin
`.fit` files into cleaned, visualized thru-hike data. Pipeline:

1. **Sort** — Garmin Takeout zips → `activities/{pre,pct,post,unknown}/` by
   watch-local date (`scripts/split_fit_by_date.py` + `manifest.csv`).
2. **Parse** — 108 PCT activity files → `tracks.parquet` (524,963 GPS/HR
   records) + `daily_summary.parquet` (99 days), same-day files merged
   (`scripts/build_tracks.py`).
3. **Cut teleports** — H5 rule (speed > 26.8 mph *and* segment > 656 ft) drops
   203 bogus segments / 250.1 mi (`scripts/compare_heuristics.py`,
   `scripts/apply_cuts.py` → `daily_corrected.parquet`).
4. **Project onto the trail** — PCTA centerline (2,652.90 mi) + half-mile
   markers → per-fix trail mile + off-route distance
   (`scripts/build_route.py`, `scripts/project_tracks.py` →
   `track_route.parquet`, `daily_route.parquet`). Net total: **2,442.8 mi**.
5. **Explore** — `notebooks/pct.py` (marimo): mileage, elevation profiles,
   corrected-vs-raw, along-trail progress.

Full rationale for every cleaning decision: [`docs/methodology.md`](docs/methodology.md).

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
