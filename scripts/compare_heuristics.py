"""Compare 5 jump-cut heuristics on the same per-segment data.

Reuses segments() from flag_jumps.py (haversine dist / dt between
consecutive GPS fixes within a file). For each heuristic, reports:
segments cut, GPS points touched, km removed, corrected total,
days affected, and worst single-day cut.

Usage: uv run python scripts/compare_heuristics.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flag_jumps import segments  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
MILE_M = 1609.344

HEURISTICS = {
    "H1 speed>12 m/s": lambda dist, dt, spd: spd > 12,
    "H2 mile-in-30s (>53.6 m/s)": lambda dist, dt, spd: spd > MILE_M / 30,
    "H3 speed>8 m/s": lambda dist, dt, spd: spd > 8,
    "H4 dist>1km": lambda dist, dt, spd: dist > 1000,
    "H5 speed>12 & dist>200m": lambda dist, dt, spd: (spd > 12) & (dist > 200),
}


def main() -> None:
    t = (
        pl.read_parquet(BASE / "data" / "tracks.parquet")
        .filter(pl.col("lat").is_not_null())
        .sort(["date", "source_file", "point_index"])
    )
    ok, dist, dts, spd, g, _pt_idx, _file_of_row = segments(t)
    valid = ~np.isnan(spd) & (dts > 0)
    raw_total_km = float(np.nansum(dist[valid])) / 1000

    # date per segment (date of its start point's file)
    dates = np.array(g["date"].to_list())
    sizes = np.array([len(s) for s in g["ts"].to_list()])
    seg_date = np.repeat(dates, sizes)[ok]

    print(f"segments: {len(ok)}, raw GPS-distance total: {raw_total_km:.1f} km\n")
    rows = []
    masks = {}
    for name, rule in HEURISTICS.items():
        m = np.zeros(len(ok), bool)
        m[valid] = rule(dist[valid], dts[valid], spd[valid])
        masks[name] = m
        cut_km = float(dist[m].sum()) / 1000
        touched = np.unique(np.concatenate([np.where(m)[0], np.where(m)[0] + 1]))
        day_cut = {}
        for d in np.unique(seg_date[m]):
            day_cut[d] = float(dist[m & (seg_date == d)].sum()) / 1000
        worst = max(day_cut.items(), key=lambda kv: kv[1]) if day_cut else ("-", 0.0)
        rows.append({
            "heuristic": name,
            "seg_cut": int(m.sum()),
            "pts_touched": len(touched),
            "km_cut": round(cut_km, 1),
            "corrected_km": round(raw_total_km - cut_km, 1),
            "days_hit": len(day_cut),
            "worst_day": f"{worst[0]} (-{worst[1]:.1f} km)",
        })
    print(pl.DataFrame(rows))
    print("\nsubset check (is row-heuristic ⊆ col-heuristic?):")
    names = list(masks)
    head = " " * 24 + "".join(f"{n.split()[0]:>8}" for n in names)
    print(head)
    for a in names:
        line = f"{a[:22]:<24}"
        for b in names:
            line += f"{'yes' if (masks[a] & ~masks[b]).sum() == 0 else 'no':>8}"
        print(line)


if __name__ == "__main__":
    sys.exit(main())
