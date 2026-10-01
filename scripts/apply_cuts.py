"""Apply H5 teleport cuts and build corrected daily mileage.

H5 (teleport): cut a GPS segment only if implied speed > 26.8 mph AND
segment distance > 656 ft. Car rules deliberately excluded — the hiker
paused for rides, and run analysis found no vehicle-like segments.

Corrected day mileage = sum of kept GPS segment haversines for that date.
NaN-speed segments (dt=0) are kept. Small real-trail gaps at cut sites
(a few hundred feet each) are NOT interpolated — negligible next to
hundreds of miles of removed teleports.

Output: data/daily_corrected.parquet with, per date:
  raw_watch_mi (device session totals), gps_raw_mi (all GPS segments),
  cut_mi, corrected_mi, n_cut_seg, n_seg

Usage: uv run python scripts/apply_cuts.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flag_jumps import segments  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
M_TO_MI = 1 / 1609.344
MS_TO_MPH = 2.23694


def main() -> None:
    t = (
        pl.read_parquet(BASE / "data" / "tracks.parquet")
        .filter(pl.col("lat").is_not_null())
        .sort(["date", "source_file", "point_index"])
    )
    ok, dist, dts, spd, g, _pt_idx, _file_of_row = segments(t)
    valid = ~np.isnan(spd) & (dts > 0)
    cut = np.zeros(len(ok), bool)
    cut[valid] = ((spd[valid] * MS_TO_MPH > 26.8)
                  & (dist[valid] * M_TO_MI > 656 / 5280))

    dates = np.array(g["date"].to_list())
    sizes = np.array([len(s) for s in g["ts"].to_list()])
    seg_date = np.repeat(dates, sizes)[ok]
    day = pl.DataFrame({"date": seg_date, "mi": dist * M_TO_MI, "cut": cut})
    per_day = (
        day.group_by("date")
        .agg([
            pl.col("mi").sum().alias("gps_raw_mi"),
            pl.col("mi").filter(pl.col("cut")).sum().alias("cut_mi"),
            pl.col("cut").sum().alias("n_cut_seg"),
            pl.len().alias("n_seg"),
        ])
        .with_columns((pl.col("gps_raw_mi") - pl.col("cut_mi")).alias("corrected_mi"))
    )

    raw = pl.read_parquet(BASE / "data" / "daily_summary.parquet").select(
        pl.col("date"), (pl.col("distance_m") / 1609.344).alias("raw_watch_mi")
    )
    out = (
        raw.join(per_day, on="date", how="left")
        .with_columns(
            pl.col("gps_raw_mi").round(2),
            pl.col("cut_mi").fill_null(0).round(2),
            pl.col("corrected_mi").round(2),
            pl.col("raw_watch_mi").round(2),
            pl.col("n_cut_seg").fill_null(0),
            pl.col("n_seg").fill_null(0),
        )
        .sort("date")
    )
    out.write_parquet(BASE / "data" / "daily_corrected.parquet")

    tot = out.select(
        pl.col("raw_watch_mi").sum(), pl.col("gps_raw_mi").sum(),
        pl.col("cut_mi").sum(), pl.col("corrected_mi").sum(),
    ).row(0, named=True)
    print(f"days: {out.height}, cut segments: {int(cut.sum())}")
    print(f"raw watch total: {tot['raw_watch_mi']:.1f} mi | "
          f"GPS raw: {tot['gps_raw_mi']:.1f} mi | "
          f"cut: {tot['cut_mi']:.1f} mi | "
          f"corrected: {tot['corrected_mi']:.1f} mi")
    print("(hiker anchor: ~2450 recorded mi — residual gap = jitter, next step: smoothing)")
    print("top cut days:")
    print(out.sort("cut_mi", descending=True).head(8).select(["date", "corrected_mi", "cut_mi", "n_cut_seg"]))


if __name__ == "__main__":
    sys.exit(main())
