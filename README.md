# PCT 2026 — Hike Data Analysis

This project was substantially produced with the help of AI coding tools.  

> **TL;DR.** I thru-hiked the Pacific Crest Trail in 2026 (Apr 25 – Aug 11):
> **2,442.8 trail miles** over **99 hiking days** (+ 10 town/zero days),
> biggest day **40.3 mi** (Jul 15), **427,500 ft** up and **430,400 ft** down,
> averaging **24.7 mi** per hiking day. Campo → Rainy Pass, where the
> Ptarmigan fire closed the final ~61 miles. Full story in the
> [report](notebooks/report.py).

## Who / what

I'm a **data scientist turned thru-hiker** (ex-finance, pandas veteran)
learning the modern stack on my own hike data. Built with Python (uv),
**polars**, **marimo**, altair, plotly, garmin-fit-sdk, scipy — pair-built
with **Muse Spark** (via OpenCode) as coding partner and analyst. Start
with the report notebook; cleaning details live in
[`docs/methodology.md`](docs/methodology.md).

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
uv run python scripts/build_daily_report.py
uv run python scripts/find_oddities.py
uv run python scripts/build_segments.py
uv run python scripts/find_climbs.py
uv run python scripts/build_wellness.py      # needs the full export in rawdata/
uv run python scripts/build_breaks_hourly.py
uv run python scripts/make_graphics.py       # standalone PNGs → graphics/
uv run python scripts/make_graphics2.py
uv run python scripts/make_graphics3.py
uv run python scripts/make_graphics4.py
uv run marimo edit notebooks/report.py     # the story (overview → trail details)
uv run marimo edit notebooks/pct.py        # working analysis notebook
```

`rawdata/` (Garmin export, GPX), `data/` (generated Parquet),
`annotations/` (private timeline) and `graphics/` (generated PNGs) are
local-only and gitignored — scripts rebuild everything from raw.

## Research report

- **Interactive (recommended):** `uv run marimo export html notebooks/report.py -o report.html`
  and host it on the blog — all charts stay interactive.
- **GitHub viewing:** `uv run marimo export ipynb notebooks/report.py -o report.ipynb` —
  GitHub renders `.ipynb` with static plots (it does *not* render marimo `.py` files).
- **Print/archive:** `uv run marimo export pdf notebooks/report.py -o report.pdf`.

## Layout

```
scripts/          pipeline (sort → parse → flag → compare → cut → route → project → report tables → graphics)
notebooks/pct.py  marimo working notebook
notebooks/report.py  marimo report (overview → trail details)
docs/             methodology + decisions
reference/        public reference data (resupply towns + trail miles)
rawdata/          (untracked) Garmin export, PCTA GPX
data/             (untracked) generated Parquet tables
annotations/      (untracked, private) hiker timeline
graphics/         (untracked) generated PNGs for posting
```
