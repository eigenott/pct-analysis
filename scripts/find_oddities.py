"""Find hike oddities: backtracks, side quests, southbound days.

- Backtracks: on-trail (≤500 ft off route) peak-to-trough route_mi drops
  >0.25 mi using fixes clear of H5-cut segments. Recovered = route later
  returns within 0.1 mi of the peak (classic dropped-item/water run).
- Side quests: contiguous off-route (>500 ft) stretches; short ones that
  return are water/view/dropped-item candidates, long ones are alternates.
- Southbound days: daily net below -0.5 mi.

Output: data/oddities.parquet.

Usage: uv run python scripts/find_oddities.py
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flag_jumps import segments  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
M_TO_MI = 1 / 1609.344
MS_TO_MPH = 2.23694
PDT = dt.timedelta(hours=-7)


def local(ts):
    return (ts + PDT).strftime("%H:%M")


def main() -> None:
    t = (
        pl.read_parquet(BASE / "data" / "tracks.parquet")
        .filter(pl.col("lat").is_not_null())
        .sort(["date", "source_file", "point_index"])
    )
    ok, dist, dts, spd, g, _pt, _fr = segments(t)
    valid = ~np.isnan(spd) & (dts > 0)
    h5 = np.zeros(len(ok), bool)
    h5[valid] = ((spd[valid] * MS_TO_MPH > 26.8)
                 & (dist[valid] * M_TO_MI > 656 / 5280))

    proj = pl.read_parquet(BASE / "data" / "track_route.parquet").sort(
        ["date", "source_file", "point_index"])
    rm = proj["route_mi"].to_numpy()
    off = proj["off_route_ft"].to_numpy()
    dates = proj["date"].to_numpy()
    ts = np.array([x.timestamp() for x in proj["timestamp_utc"].to_list()])

    # fixes adjacent to a cut segment are untrustworthy for shape analysis
    bad_rows = set(ok[h5].tolist()) | set((ok[h5] + 1).tolist())
    good = np.array([i not in bad_rows for i in range(len(proj))])
    on_trail = (off <= 500) & good

    rows = []
    for d in sorted(set(dates.tolist())):
        f = np.where(dates == d)[0]
        r = rm[f]
        o = off[f]
        tt = [proj["timestamp_utc"][int(i)] for i in f]
        order = np.argsort([x.timestamp() for x in tt])
        f, r, o = f[order], r[order], o[order]
        tt = [tt[i] for i in order]

        # --- backtracks on the on-trail portions
        m = on_trail[f]
        if m.sum() > 2:
            rr = r[m]
            ttm = [tt[i] for i in np.where(m)[0]]
            peak, trough, pi = rr[0], rr[0], 0
            for i, v in enumerate(rr):
                if v > peak:
                    peak, trough, pi = v, v, i
                trough = min(trough, v)
                if peak - trough > 0.25:
                    rec = bool((rr[i:] >= peak - 0.1).any())
                    dur_min = (ttm[i].timestamp() - ttm[pi].timestamp()) / 60
                    rows.append({
                        "type": "backtrack", "date": d,
                        "start_local": local(ttm[pi]),
                        "end_local": local(ttm[i]),
                        "mi": round(float(peak - trough), 2),
                        "extra_mi": round(float(2 * (peak - trough) if rec else (peak - trough)), 2),
                        "recovered": rec,
                        "at_route_mi": round(float(peak), 1),
                        "detail": (
                            f"{dur_min:.1f} min; "
                            + ("instant flip — likely hairpin projection artifact, not footsteps"
                               if dur_min < 2 else "sustained — likely real footsteps")
                        ),
                    })
                    break  # one (largest-first) event per day; rerun for more

        # --- off-route stretches
        om = (o > 500) & good[f]
        if om.any():
            idx = np.where(om)[0]
            for run in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
                dur_min = (tt[run[-1]].timestamp() - tt[run[0]].timestamp()) / 60
                if dur_min < 3:
                    continue
                returned = bool((o[run[-1] + 1:] <= 500).any()) if run[-1] + 1 < len(o) else False
                rows.append({
                    "type": "sidequest" if dur_min < 120 else "alternate",
                    "date": d,
                    "start_local": local(tt[run[0]]),
                    "end_local": local(tt[run[-1]]),
                    "mi": round(float(o[run].max()) / 5280, 2),
                    "extra_mi": None,
                    "recovered": returned,
                    "at_route_mi": round(float(r[run[0]]), 1),
                    "detail": f"{dur_min:.0f} min, max {o[run].max():,.0f} ft off route",
                })

    # --- southbound days
    daily = pl.read_parquet(BASE / "data" / "daily_route.parquet")
    for row in daily.filter(pl.col("net_mi") < -0.5).iter_rows(named=True):
        rows.append({
            "type": "southbound_day", "date": row["date"],
            "start_local": None, "end_local": None,
            "mi": round(abs(row["net_mi"]), 2), "extra_mi": None,
            "recovered": False, "at_route_mi": round(row["route_min_mi"], 1),
            "detail": f"net {row['net_mi']:.1f} mi southbound",
        })

    odd = pl.DataFrame(rows, schema={
        "type": pl.String, "date": pl.String, "start_local": pl.String,
        "end_local": pl.String, "mi": pl.Float64, "extra_mi": pl.Float64,
        "recovered": pl.Boolean, "at_route_mi": pl.Float64, "detail": pl.String,
    }).sort(["date", "type"])
    odd.write_parquet(BASE / "data" / "oddities.parquet")
    print(f"oddities: {len(odd)}", flush=True)
    print(odd.group_by("type").len().sort("type"))
    print(odd.filter(pl.col("type") == "backtrack").select(
        ["date", "start_local", "end_local", "mi", "recovered", "at_route_mi"]))


if __name__ == "__main__":
    sys.exit(main())
