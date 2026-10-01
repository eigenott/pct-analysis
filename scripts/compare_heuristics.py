"""Compare jump/car-cut heuristics on the same per-segment data (imperial).

Reuses segments() from flag_jumps.py (haversine dist / dt between
consecutive GPS fixes within a file). Speeds in mph, distances in miles.

Rules:
  H1 speed > 26.8 mph (12 m/s — impossible on foot, any duration)
  H2 mile-in-30s (>119.9 mph)
  H3 speed > 17.9 mph (8 m/s)
  H4 segment longer than 0.62 mi (1 km), any speed
  H5 H1 AND segment longer than 656 ft (200 m) — teleports only, not jitter
  H6 sustained >5 mph for over 60 s — hiker says they never did this;
      catches unpaused car rides (and long-gap jumps)
  H7 H5 OR H6 (proposed apply rule)

Usage: uv run python scripts/compare_heuristics.py
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

HEURISTICS = {
    "H1 speed>26.8mph": lambda dist, dt, spd: spd * MS_TO_MPH > 26.8,
    "H2 mile-in-30s": lambda dist, dt, spd: spd * MS_TO_MPH > 119.9,
    "H3 speed>17.9mph": lambda dist, dt, spd: spd * MS_TO_MPH > 17.9,
    "H4 seg>0.62mi": lambda dist, dt, spd: dist * M_TO_MI > 0.62,
    "H5 teleport (>26.8mph & >656ft)": lambda dist, dt, spd: (
        (spd * MS_TO_MPH > 26.8) & (dist * M_TO_MI > 656 / 5280)
    ),
    "H6 car (>5mph sustained 60s+)": lambda dist, dt, spd: (
        (spd * MS_TO_MPH > 5) & (dt > 60)
    ),
    "H7 H5+H6 combined": lambda dist, dt, spd: (
        ((spd * MS_TO_MPH > 26.8) & (dist * M_TO_MI > 656 / 5280))
        | ((spd * MS_TO_MPH > 5) & (dt > 60))
    ),
}

RUN_MPH, RUN_MIN_S = 5, 60  # H6': over 5 mph for over 60 s straight


def sustained_runs(ok, dist, ts, spd, file_of_seg, mph_thresh, min_secs):
    """Maximal within-file runs of consecutive segments all faster than
    mph_thresh; flag runs lasting longer than min_secs. Returns bool mask."""
    fast = (spd * MS_TO_MPH > mph_thresh) & ~np.isnan(spd)
    flag = np.zeros(len(ok), bool)
    if not fast.any():
        return flag
    idx = np.where(fast)[0]
    # split where not adjacent in segment order OR file changes
    breaks = np.where(
        (np.diff(idx) != 1) | (file_of_seg[idx[1:]] != file_of_seg[idx[:-1]])
    )[0]
    runs = np.split(idx, breaks + 1)
    for r in runs:
        dur = ts[r[-1] + 1] - ts[r[0]]  # end time of last seg minus start of first
        if dur > min_secs:
            flag[r] = True
    return flag


def main() -> None:
    t = (
        pl.read_parquet(BASE / "data" / "tracks.parquet")
        .filter(pl.col("lat").is_not_null())
        .sort(["date", "source_file", "point_index"])
    )
    ok, dist, dts, spd, g, _pt_idx, file_of_row = segments(t)
    valid = ~np.isnan(spd) & (dts > 0)
    raw_total_mi = float(np.nansum(dist[valid])) * M_TO_MI
    ts_all = np.concatenate([
        np.array([x.timestamp() for x in s]) for s in g["ts"].to_list()
    ])
    file_of_seg = file_of_row[ok]

    dates = np.array(g["date"].to_list())
    sizes = np.array([len(s) for s in g["ts"].to_list()])
    seg_date = np.repeat(dates, sizes)[ok]

    print(f"segments: {len(ok)}, raw GPS-distance total: {raw_total_mi:.1f} mi\n")
    rows, masks = [], {}
    for name, rule in HEURISTICS.items():
        m = np.zeros(len(ok), bool)
        m[valid] = rule(dist[valid], dts[valid], spd[valid])
        masks[name] = m
        cut_mi = float(dist[m].sum()) * M_TO_MI
        touched = np.unique(np.concatenate([np.where(m)[0], np.where(m)[0] + 1]))
        day_cut = {}
        for d in np.unique(seg_date[m]):
            day_cut[d] = float(dist[m & (seg_date == d)].sum()) * M_TO_MI
        worst = max(day_cut.items(), key=lambda kv: kv[1]) if day_cut else ("-", 0.0)
        rows.append({
            "heuristic": name,
            "seg_cut": int(m.sum()),
            "pts_touched": len(touched),
            "mi_cut": round(cut_mi, 1),
            "corrected_mi": round(raw_total_mi - cut_mi, 1),
            "days_hit": len(day_cut),
            "worst_day": f"{worst[0]} (-{worst[1]:.1f} mi)",
        })
    print(pl.DataFrame(rows))

    # H6': sustained runs — every segment in the run >5 mph, run lasts >60 s
    h6run = sustained_runs(ok, dist, ts_all, spd, file_of_seg, RUN_MPH, RUN_MIN_S)
    h5 = masks["H5 teleport (>26.8mph & >656ft)"]
    for name, m in [
        (f"H6' sustained >{RUN_MPH}mph {RUN_MIN_S}s+", h6run),
        ("H7' H5+H6' combined", h5 | h6run),
    ]:
        cut_mi = float(dist[m].sum()) * M_TO_MI
        touched = np.unique(np.concatenate([np.where(m)[0], np.where(m)[0] + 1]))
        day_cut = {}
        for d in np.unique(seg_date[m]):
            day_cut[d] = float(dist[m & (seg_date == d)].sum()) * M_TO_MI
        worst = max(day_cut.items(), key=lambda kv: kv[1]) if day_cut else ("-", 0.0)
        print(f"{name}: {m.sum()} seg, {len(touched)} pts, "
              f"{cut_mi:.1f} mi cut → {raw_total_mi - cut_mi:.1f} mi, "
              f"{len(day_cut)} days, worst {worst[0]} (-{worst[1]:.1f} mi)")

    # Diagnostic: fast-but-short (H1 minus H5) — isolated jitter or car runs?
    h1 = masks["H1 speed>26.8mph"]
    fss = h1 & ~h5
    idx = np.where(fss)[0]
    breaks = np.where(
        (np.diff(idx) != 1) | (file_of_seg[idx[1:]] != file_of_seg[idx[:-1]])
    )[0]
    runs = np.split(idx, breaks + 1) if len(idx) else []
    run_mi = sorted(
        ((float(dist[r].sum()) * M_TO_MI, len(r), str(seg_date[r[0]])) for r in runs),
        reverse=True,
    )
    print(f"\nH1-not-H5: {fss.sum()} seg in {len(runs)} runs, "
          f"{float(dist[fss].sum()) * M_TO_MI:.1f} mi total")
    print("longest runs (mi, n_seg, date) — long runs at speed smell like car:")
    for mi, n, d in run_mi[:10]:
        print(f"  {mi:6.1f} mi  {n:>4} seg  {d}")


if __name__ == "__main__":
    sys.exit(main())
