"""Breaks + per-hour mileage from the segment table.

- Break = run of consecutive in-file segments all slower than 0.7 mph
  lasting 10+ min (UltraTrac still records while standing around, so
  stillness is visible). data/breaks.parquet: date, start/end local,
  minutes, miles drifted while "stopped".
- data/hourly.parquet: per date+local-hour kept-segment miles (for
  fastest-hour analysis). H5-teleport segments excluded from miles but a
  break run ends at file/teleport boundaries.

Usage: uv run python scripts/build_breaks_hourly.py
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent.parent
PDT = dt.timedelta(hours=-7)


def main() -> None:
    s = pl.read_parquet(BASE / "data" / "segments.parquet").sort(
        ["date", "source_file", "seg_order"])
    ts = pl.read_parquet(BASE / "data" / "tracks.parquet").filter(
        pl.col("lat").is_not_null()).sort(
        ["date", "source_file", "point_index"])
    ts_list = np.array([x.timestamp() for x in ts["timestamp_utc"].to_list()])

    seg_ts = ts_list[s["seg_order"].to_numpy()]
    slow = (s["mph"].to_numpy() < 0.7) & (s["dt_s"].to_numpy() <= 300)
    files = s["source_file"].to_numpy()
    dates = s["date"].to_numpy()
    dists = s["dist_mi"].to_numpy()

    breaks, hourly = [], {}
    idx = np.where(slow)[0]
    if len(idx):
        same = (np.diff(idx) == 1) & (files[idx[1:]] == files[idx[:-1]])
        for run in np.split(idx, np.where(~same)[0] + 1):
            dur_min = (ts_list[run[-1] + 1] - ts_list[run[0]]) / 60
            if dur_min < 10:
                continue
            t0 = dt.datetime.fromtimestamp(ts_list[run[0]], tz=dt.timezone.utc) + PDT
            t1 = dt.datetime.fromtimestamp(ts_list[run[-1] + 1], tz=dt.timezone.utc) + PDT
            breaks.append({
                "date": str(dates[run[0]]),
                "start_local": t0.strftime("%H:%M"),
                "end_local": t1.strftime("%H:%M"),
                "minutes": round(dur_min, 1),
                "drift_mi": round(float(dists[run].sum()), 3),
            })
    brk = pl.DataFrame(breaks, schema={
        "date": pl.String, "start_local": pl.String, "end_local": pl.String,
        "minutes": pl.Float64, "drift_mi": pl.Float64})
    if len(brk):
        brk.write_parquet(BASE / "data" / "breaks.parquet")

    kept = s.filter("kept_h5")
    hh = kept.group_by(["date", "hour_local"]).agg(
        pl.col("dist_mi").sum().alias("mi")).sort(["date", "hour_local"])
    hh.write_parquet(BASE / "data" / "hourly.parquet")

    ndays = s["date"].n_unique()
    print(f"breaks: {len(brk)} on {brk['date'].n_unique() if len(brk) else 0} days "
          f"({ndays} days with fixes)")
    if len(brk):
        print(f"median break: {brk['minutes'].median():.0f} min, "
              f"longest: {brk['minutes'].max():.0f} min "
              f"({brk.sort('minutes', descending=True).row(0, named=True)['date']})")
    print("median hourly miles (full days) by hour:")
    full = pl.read_parquet(BASE / "data" / "daily_report.parquet").filter(
        "is_full")["date"].to_list()
    print(hh.filter(pl.col("date").is_in(full)).group_by("hour_local").agg(
        pl.col("mi").median().alias("med_mi")).sort("hour_local"))


if __name__ == "__main__":
    sys.exit(main())
