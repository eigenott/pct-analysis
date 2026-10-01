"""Rich per-day table for the report: joins route/corrected/summary data and
adds start/end times (PDT = UTC-7, whole trail), camp elevations, pace.

Output: data/daily_report.parquet (99 rows).

Usage: uv run python scripts/build_daily_report.py
"""
import datetime as dt
import sys
from pathlib import Path

import polars as pl

BASE = Path(__file__).resolve().parent.parent
PDT = dt.timedelta(hours=-7)


def main() -> None:
    route = pl.read_parquet(BASE / "data" / "daily_route.parquet")
    corr = pl.read_parquet(BASE / "data" / "daily_corrected.parquet").select(
        ["date", "raw_watch_mi", "corrected_mi", "cut_mi", "n_cut_seg"])
    summ = pl.read_parquet(BASE / "data" / "daily_summary.parquet").select(
        ["date", "ascent_m", "descent_m", "timer_s", "elapsed_s",
         "calories", "avg_hr", "max_hr"])

    t = pl.read_parquet(BASE / "data" / "tracks.parquet").sort(
        ["date", "timestamp_utc"])
    ends = (
        t.group_by("date")
        .agg([
            pl.col("timestamp_utc").min().alias("ts_min"),
            pl.col("timestamp_utc").max().alias("ts_max"),
            pl.col("altitude_m").filter(pl.col("altitude_m").is_not_null()).first().alias("ele_start_m"),
            pl.col("altitude_m").filter(pl.col("altitude_m").is_not_null()).last().alias("ele_end_m"),
            ((pl.col("timestamp_utc").max() - pl.col("timestamp_utc").min()).dt.total_hours()).round(1).alias("day_len_hrs"),
        ])
    )

    daily = (
        route.join(corr, on="date", how="left")
        .join(summ, on="date", how="left")
        .join(ends, on="date", how="left")
        .with_columns([
            (pl.col("ascent_m") * 3.28084).round(0).cast(pl.Int64).alias("ascent_ft"),
            (pl.col("descent_m") * 3.28084).round(0).cast(pl.Int64).alias("descent_ft"),
            (pl.col("timer_s") / 3600).round(1).alias("moving_hrs"),
            (pl.col("elapsed_s") / 3600).round(1).alias("elapsed_hrs"),
            (pl.col("ele_start_m") * 3.28084).round(0).cast(pl.Int64).alias("camp_start_ft"),
            (pl.col("ele_end_m") * 3.28084).round(0).cast(pl.Int64).alias("camp_end_ft"),
        ])
        .with_columns(
            (pl.col("net_mi") / pl.col("moving_hrs")).round(2).alias("pace_mph"))
        .sort("date")
    )
    # local start/end clock times (whole trail is Pacific → UTC-7 in Aug)
    times = pl.DataFrame([{
        "date": r["date"],
        "start_local": (r["ts_min"] + PDT).strftime("%H:%M"),
        "end_local": (r["ts_max"] + PDT).strftime("%H:%M"),
    } for r in ends.to_dicts()])
    daily = daily.drop(["ts_min", "ts_max"]).join(times, on="date", how="left")
    daily.write_parquet(BASE / "data" / "daily_report.parquet")
    print(f"days: {len(daily)}", flush=True)
    print(daily.select(["date", "net_mi", "pace_mph", "start_local", "end_local",
                        "camp_start_ft", "camp_end_ft"]).head(3))


if __name__ == "__main__":
    sys.exit(main())
