"""Project every GPS fix onto the route centerline + daily along-trail miles.

- Nearest centerline point per fix via cKDTree on 3D unit vectors
  (exact; centerline spacing is ~11 ft).
- data/track_route.parquet: (source_file, point_index) -> route_mi,
  off_route_ft.
- data/daily_route.parquet: per date net_mi (route max-min over kept
  fixes), gross_mi (sum of |route_mi| deltas over kept H5 segments),
  offroute_frac (fixes >500 ft off route), plus corrected_mi for reference.
  Kept segments exclude H5 teleports (same rule as apply_cuts.py).

Usage: uv run python scripts/project_tracks.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
M_TO_MI = 1 / 1609.344
MS_TO_MPH = 2.23694


def to_xyz(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def main() -> None:
    cl = pl.read_parquet(BASE / "data" / "route_centerline.parquet")
    tree = cKDTree(to_xyz(cl["lat"].to_numpy(), cl["lon"].to_numpy()))
    cum_mi = cl["route_mi"].to_numpy()
    print(f"centerline tree: {len(cl)} pts", flush=True)

    t = (
        pl.read_parquet(BASE / "data" / "tracks.parquet")
        .filter(pl.col("lat").is_not_null())
        .sort(["date", "source_file", "point_index"])
    )
    dist, idx = tree.query(to_xyz(t["lat"].to_numpy(), t["lon"].to_numpy()), k=1)
    proj = pl.DataFrame({
        "source_file": t["source_file"],
        "point_index": t["point_index"],
        "date": t["date"],
        "timestamp_utc": t["timestamp_utc"],
        "route_mi": cum_mi[idx],
        "off_route_ft": dist * 6371000.0 * 3.28084,
    })
    proj.write_parquet(BASE / "data" / "track_route.parquet")
    print(f"projected fixes: {len(proj)}, "
          f"median off-route: {proj['off_route_ft'].median():.0f} ft", flush=True)

    # kept-segment mask (H5) in track order
    g = t.group_by(["date", "source_file"], maintain_order=True).agg([
        pl.col("timestamp_utc").alias("ts"),
    ])
    sizes = np.array([len(s) for s in g["ts"].to_list()])
    ts_all = np.concatenate([np.array([x.timestamp() for x in s]) for s in g["ts"].to_list()])
    file_of_row = np.repeat(np.arange(len(sizes)), sizes)
    ok = np.arange(len(t) - 1)
    ok = ok[file_of_row[ok] == file_of_row[ok + 1]]
    la = np.radians(t["lat"].to_numpy())
    lo = np.radians(t["lon"].to_numpy())
    dl = la[ok + 1] - la[ok]
    dp = lo[ok + 1] - lo[ok]
    a = np.sin(dl / 2) ** 2 + np.cos(la[ok]) * np.cos(la[ok + 1]) * np.sin(dp / 2) ** 2
    seg_mi = 2 * 6371000.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1)) * M_TO_MI
    seg_dt = ts_all[ok + 1] - ts_all[ok]
    seg_spd = np.where(seg_dt > 0, seg_mi / (seg_dt / 3600), np.nan)  # mph
    h5 = (seg_spd > 26.8) & (seg_mi > 656 / 5280)

    rm = proj["route_mi"].to_numpy()
    off = proj["off_route_ft"].to_numpy()
    dates = proj["date"].to_numpy()

    seg = pl.DataFrame({
        "date": dates[ok],
        "delta": rm[ok + 1] - rm[ok],
        "kept": ~h5,
    })
    gross = (
        seg.filter("kept")
        .group_by("date")
        .agg(pl.col("delta").abs().sum().alias("gross_mi"))
    )
    per_fix = (
        proj.group_by("date")
        .agg([
            pl.col("route_mi").min().alias("route_min_mi"),
            pl.col("route_mi").max().alias("route_max_mi"),
            (pl.col("off_route_ft") > 500).mean().alias("offroute_frac"),
            pl.len().alias("n_fix"),
        ])
        .with_columns((pl.col("route_max_mi") - pl.col("route_min_mi")).alias("net_mi"))
    )
    daily = (
        per_fix.join(gross, on="date", how="left")
        .with_columns(
            pl.col("route_min_mi").round(2), pl.col("route_max_mi").round(2),
            pl.col("net_mi").round(2), pl.col("gross_mi").round(2),
            pl.col("offroute_frac").round(3),
        )
    )
    ref = pl.read_parquet(BASE / "data" / "daily_corrected.parquet").select(
        ["date", "corrected_mi", "cut_mi"])
    daily = daily.join(ref, on="date", how="left").sort("date")
    daily.write_parquet(BASE / "data" / "daily_route.parquet")

    tot = daily.select(
        pl.col("net_mi").sum(), pl.col("gross_mi").sum(),
        pl.col("corrected_mi").sum()).row(0, named=True)
    print(f"net: {tot['net_mi']:.1f} mi | gross kept-seg: {tot['gross_mi']:.1f} mi | "
          f"H5-corrected GPS sum: {tot['corrected_mi']:.1f} mi (anchor ~2450)")
    print("highest off-route-fraction days (alternate candidates):")
    print(daily.sort("offroute_frac", descending=True).head(10).select(
        ["date", "net_mi", "gross_mi", "offroute_frac"]))
    print("net-vs-gross gaps >2 mi (backtrack/jitter days):")
    g = daily.with_columns((pl.col("gross_mi") - pl.col("net_mi")).alias("gap"))
    print(g.filter(pl.col("gap") > 2).select(["date", "net_mi", "gross_mi", "gap"]))


if __name__ == "__main__":
    sys.exit(main())
