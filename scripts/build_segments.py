"""Per-segment table for analyses: distance, time, speed, elevation change,
grade, H5 teleport flag. One row per consecutive in-file GPS fix pair.

Output: data/segments.parquet (343,065 rows).

Usage: uv run python scripts/build_segments.py
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
    ok, dist, dts, spd, g, _pt, _fr = segments(t)
    alt = np.concatenate([
        np.array([(a if a is not None else np.nan) for a in s], dtype=float)
        for s in g["alt"].to_list()])
    dates = np.array(g["date"].to_list())
    sizes = np.array([len(s) for s in g["ts"].to_list()])
    files = np.array(g["source_file"].to_list())
    seg_date, seg_file = np.repeat(dates, sizes)[ok], np.repeat(files, sizes)[ok]
    ts0 = np.concatenate([np.array([x.timestamp() for x in s]) for s in g["ts"].to_list()])
    valid = ~np.isnan(spd) & (dts > 0)

    dalt_m = alt[ok + 1] - alt[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.where(dist > 0, dalt_m / dist, np.nan)  # rise/run
    h5 = np.zeros(len(ok), bool)
    h5[valid] = ((spd[valid] * MS_TO_MPH > 26.8) & (dist[valid] * M_TO_MI > 656 / 5280))

    seg = pl.DataFrame({
        "date": seg_date,
        "source_file": seg_file,
        "dist_mi": dist * M_TO_MI,
        "dt_s": dts,
        "mph": spd * MS_TO_MPH,
        "dalt_ft": dalt_m * 3.28084,
        "grade": grade,
        "kept_h5": ~h5,
        "hour_local": ((ts0[ok] - 7 * 3600) // 3600 % 24).astype(int),  # PDT
    })
    seg.write_parquet(BASE / "data" / "segments.parquet")
    print(f"segments: {len(seg)}, kept: {int(seg['kept_h5'].sum())}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
