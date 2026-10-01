"""Find the longest sustained climbs on the PCTA centerline.

Method: rolling-median smooth elevation (41-pt window ≈ 450 ft — kills GPS
spikes, keeps switchbacks), then walk the profile: an ascent runs from a
start point until elevation drops more than TOLgap below its running max.
Ascents gaining ≥2000 ft are climbs. Switchback wiggles under the tolerance
don't break a climb; real descents do.

Output: data/climbs.parquet (rank, start/end mile, gain, length, ft/mi,
section, dates the hiker traversed it, mean pace those days).

Usage: uv run python scripts/find_climbs.py [--tol 250] [--min-gain 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent.parent
BOUNDS = [0, 702, 1092, 1694, 2146, 1e9]
SECTIONS = ["SoCal", "Sierra", "NorCal", "Oregon", "Washington"]


def section_of(mi: float) -> str:
    for b, s in zip(BOUNDS[1:], SECTIONS):
        if mi < b:
            return s
    return SECTIONS[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=600.0)
    ap.add_argument("--min-gain", type=float, default=2000.0)
    args = ap.parse_args()

    cl = pl.read_parquet(BASE / "data" / "route_centerline.parquet")
    mi = cl["route_mi"].to_numpy()
    ele = cl["ele_m"].to_numpy() * 3.28084
    ele = np.where(np.isnan(ele), np.nanmedian(ele), ele)
    k = 41
    sm = np.median(np.lib.stride_tricks.sliding_window_view(ele, k), axis=1)
    smi = np.convolve(mi, np.ones(k) / k, mode="valid")

    climbs, s, peak = [], 0, 0
    for i in range(1, len(sm)):
        if sm[i] > sm[peak]:
            peak = i
        if sm[peak] - sm[i] > args.tol:
            if sm[peak] - sm[s] >= args.min_gain:
                climbs.append((s, peak))
            s, peak = i, i
    if sm[peak] - sm[s] >= args.min_gain:
        climbs.append((s, peak))
    climbs.sort(key=lambda sp: sm[sp[1]] - sm[sp[0]], reverse=True)

    daily = pl.read_parquet(BASE / "data" / "daily_report.parquet")
    rows = []
    dsm = np.diff(sm)
    for rank, (a, b) in enumerate(climbs[:20]):
        gain = float(sm[b] - sm[a])
        length = float(smi[b] - smi[a])
        seg = dsm[a:b]
        gross_up = float(seg[seg > 0].sum())
        gross_dn = float(-seg[seg < 0].sum())
        hit = daily.filter(
            (pl.col("route_min_mi") <= smi[b]) & (pl.col("route_max_mi") >= smi[a]))
        rows.append({
            "rank": rank + 1,
            "start_mi": round(float(smi[a]), 1),
            "end_mi": round(float(smi[b]), 1),
            "gain_ft": int(round(gain)),
            "gross_up_ft": int(round(gross_up)),
            "gross_dn_ft": int(round(gross_dn)),
            "length_mi": round(length, 1),
            "ft_per_mi": int(round(gain / length)) if length > 0 else None,
            "section": section_of((smi[a] + smi[b]) / 2),
            "hike_dates": ", ".join(hit["date"].to_list()),
            "pace_mph": (round(float(hit["pace_mph"].mean()), 2)
                         if hit.height else None),
        })
    out = pl.DataFrame(rows)
    out.write_parquet(BASE / "data" / "climbs.parquet")
    print(out.select(["rank", "start_mi", "end_mi", "gain_ft", "gross_up_ft",
                      "length_mi", "ft_per_mi", "section", "pace_mph"]))


if __name__ == "__main__":
    sys.exit(main())
