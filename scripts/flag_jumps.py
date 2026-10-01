"""Flag impossible-motion GPS segments (UltraTrac jumps) and group into events.

Method: haversine distance / dt between consecutive GPS fixes *within the
same file*. Segments faster than SPEED_THRESHOLD_M_S are flagged, then
flagged segments within GAP_TOL segments of each other form one event.

Output: data/jump_candidates.parquet, one row per event, ranked by jump_km.
A verdict column is NOT included — verdicts live in data/jump_verdicts.csv,
written by the notebook review UI (scripts/ never overwrite human input).

Usage: uv run python scripts/flag_jumps.py [--threshold 12] [--gap 3]
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent.parent
R_EARTH = 6371000.0


def segments(t: pl.DataFrame):
    """Per-segment dist/dt/speed within each (date, source_file)."""
    g = t.group_by(["date", "source_file"], maintain_order=True).agg([
        pl.col("lat").alias("lat"),
        pl.col("lon").alias("lon"),
        pl.col("timestamp_utc").alias("ts"),
        pl.col("point_index").alias("idx"),
        pl.col("altitude_m").alias("alt"),
    ])
    lats = np.concatenate(g["lat"].to_list())
    lons = np.concatenate(g["lon"].to_list())
    pt_idx = np.concatenate(g["idx"].to_list())  # original point_index per GPS row
    ts = np.concatenate([
        np.array([x.timestamp() for x in s]) for s in g["ts"].to_list()
    ])
    sizes = np.array([len(s) for s in g["ts"].to_list()])
    file_of_row = np.repeat(np.arange(len(sizes)), sizes)

    ok = np.arange(len(lats) - 1)
    ok = ok[file_of_row[ok] == file_of_row[ok + 1]]  # no cross-file segments
    file_of_seg = file_of_row[ok]
    p1, p2 = np.radians(lats[ok]), np.radians(lats[ok + 1])
    dp = np.radians(lons[ok + 1] - lons[ok])
    dl = p2 - p1
    a = np.sin(dl / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dp / 2) ** 2
    dist = 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dts = ts[ok + 1] - ts[ok]
    spd = np.where(dts > 0, dist / dts, np.nan)
    return ok, dist, dts, spd, g, pt_idx, file_of_row


def group_events(flag_idx: np.ndarray, gap_tol: int):
    """Cluster flagged segment positions into events (gap <= gap_tol)."""
    if len(flag_idx) == 0:
        return []
    events, start, prev = [], flag_idx[0], flag_idx[0]
    for j in flag_idx[1:]:
        if j - prev > gap_tol + 1:
            events.append((start, prev))
            start = j
        prev = j
    events.append((start, prev))
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=12.0,
                    help="m/s above which a segment is impossible on foot")
    ap.add_argument("--gap", type=int, default=3)
    args = ap.parse_args()

    t = (
        pl.read_parquet(BASE / "data" / "tracks.parquet")
        .filter(pl.col("lat").is_not_null())
        .sort(["date", "source_file", "point_index"])
    )
    ok, dist, dts, spd, g, pt_idx, file_of_row = segments(t)
    flagged = ok[~np.isnan(spd) & (spd > args.threshold)]
    print(f"segments: {len(ok)}, flagged>{args.threshold} m/s: {len(flagged)}",
          flush=True)

    # map segment position -> (file, point rows) for event context
    files = list(zip(g["date"].to_list(), g["source_file"].to_list()))
    sizes = np.array([len(s) for s in g["ts"].to_list()])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    seg_file = np.repeat(np.arange(len(sizes)), sizes)[ok]

    events = group_events(np.sort(flagged), args.gap)
    rows = []
    for eid, (s0, s1) in enumerate(events):
        seg_mask = (ok >= s0) & (ok <= s1)
        fids = np.unique(seg_file[seg_mask])
        date, fname = files[fids[0]]
        # point rows spanned: segments s0..s1 touch points s0..s1+1 (same file here
        # unless event spans files — split those; with gap_tol small, assume one)
        # original point_index span (inclusive), clipped to the event's file
        span_rows = np.arange(s0, s1 + 2)
        span_rows = span_rows[file_of_row[span_rows] == file_of_row[s0]]
        rows.append({
            "jump_id": eid,
            "date": date,
            "source_file": fname,
            "seg_start": int(s0),
            "seg_end": int(s1),
            # original point_index span (inclusive) for slicing tracks.parquet
            "pt_start": int(pt_idx[span_rows].min()),
            "pt_end": int(pt_idx[span_rows].max()),
            "n_flagged_seg": int(seg_mask.sum()),
            "jump_km": round(float(dist[seg_mask].sum()) / 1000, 3),
            "max_speed_m_s": round(float(np.nanmax(spd[seg_mask])), 1),
            "worst_dt_s": round(float(dts[seg_mask][np.nanargmax(spd[seg_mask])]), 1),
            "multi_file": bool(len(fids) > 1),
        })
    cand = pl.DataFrame(rows).sort("jump_km", descending=True).with_row_index("rank")
    cand.write_parquet(BASE / "data" / "jump_candidates.parquet")
    print(f"events: {len(cand)}", flush=True)
    print("jump_km distribution (km):")
    print(cand["jump_km"].describe())
    print("top 10:")
    print(cand.head(10).select(
        ["rank", "jump_id", "date", "n_flagged_seg", "jump_km",
         "max_speed_m_s", "worst_dt_s"]))
    per_day = cand.group_by("date").agg(
        pl.len().alias("n_events"), pl.col("jump_km").sum().alias("jump_km"))
    print("worst 10 days by flagged km:")
    print(per_day.sort("jump_km", descending=True).head(10))


if __name__ == "__main__":
    sys.exit(main())
