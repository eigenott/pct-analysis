"""Build route reference tables from PCTA GPX files.

- data/route_centerline.parquet: ordered centerline points with cumulative
  miles (haversine), one row per track point. Track order in the file is
  verified continuous (each starts where the last ended).
- data/route_markers.parquet: half-mile markers sorted by trail mile.

Usage: uv run python scripts/build_route.py
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent.parent
R_MI = 6371000.0 / 1609.344


def parse_tracks(fp: Path):
    """Yield (section_name, lats, lons, eles) per track, streaming."""
    name, la, lo, el = None, [], [], []
    for ev, elem in ET.iterparse(fp, events=("start", "end")):
        tag = elem.tag.split("}")[-1]
        if ev == "start" and tag == "trk":
            name, la, lo, el = None, [], [], []
        elif ev == "end" and tag == "name" and name is None:
            # first <name> inside a trk is the track name (parent check via stack depth
            # is overkill; track names come before any trkpt names in these files)
            if elem.text and not la:
                name = elem.text
        elif ev == "end" and tag == "trkpt":
            la.append(float(elem.get("lat")))
            lo.append(float(elem.get("lon")))
            e = elem.find("{*}ele")
            el.append(float(e.text) if e is not None and e.text else None)
            elem.clear()
        elif ev == "end" and tag == "trk":
            yield name or "?", np.array(la), np.array(lo), np.array(el, dtype=float)
            elem.clear()


def main() -> None:
    tracks = list(parse_tracks(BASE / "rawdata" / "pcta_route" / "pcta_centerline.gpx"))
    print(f"tracks: {len(tracks)}", flush=True)

    # continuity: end[i] should ≈ start[i+1]
    worst = 0.0
    for (n1, la1, lo1, _), (n2, la2, lo2, _) in zip(tracks, tracks[1:]):
        lat_a, lat_b = np.radians([la1[-1], la2[0]])
        lon_a, lon_b = np.radians([lo1[-1], lo2[0]])
        a = np.sin((lat_b - lat_a) / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_b - lon_a) / 2) ** 2
        gap_ft = 2 * 6371000.0 * np.arcsin(np.sqrt(a)) * 3.28084
        worst = max(worst, gap_ft)
        if gap_ft > 5280:
            print(f"  GAP {gap_ft:,.0f} ft between {n1} and {n2}")
    print(f"worst section joint: {worst:,.0f} ft", flush=True)

    lat = np.concatenate([t[1] for t in tracks])
    lon = np.concatenate([t[2] for t in tracks])
    ele = np.concatenate([t[3] for t in tracks])
    sec = np.concatenate([[t[0]] * len(t[1]) for t in tracks])
    p1, p2 = np.radians(lat[:-1]), np.radians(lat[1:])
    dp = np.radians(lon[1:] - lon[:-1])
    dl = p2 - p1
    a = np.sin(dl / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dp / 2) ** 2
    step = 2 * R_MI * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    cum = np.concatenate([[0.0], np.cumsum(step)])
    cl = pl.DataFrame({
        "order_idx": np.arange(len(lat), dtype=np.int64),
        "section": sec,
        "lat": lat,
        "lon": lon,
        "ele_m": ele,
        "route_mi": cum,
    })
    cl.write_parquet(BASE / "data" / "route_centerline.parquet")
    print(f"centerline: {len(cl)} pts, {cum[-1]:.2f} mi total", flush=True)

    mk = []
    for ev, elem in ET.iterparse(
        BASE / "rawdata" / "pcta_route" / "pcta_mile_markers.gpx", events=("end",)
    ):
        if elem.tag.split("}")[-1] == "wpt":
            nm = elem.find("{*}name")
            ds = elem.find("{*}desc")
            try:
                mk.append((float(nm.text), float(elem.get("lat")), float(elem.get("lon")),
                           ds.text if ds is not None else None))
            except (TypeError, ValueError):
                pass
            elem.clear()
    mk = pl.DataFrame(
        {"mile": [m[0] for m in mk], "lat": [m[1] for m in mk],
         "lon": [m[2] for m in mk], "section": [m[3] for m in mk]}
    ).sort("mile")
    mk.write_parquet(BASE / "data" / "route_markers.parquet")
    print(f"markers: {len(mk)} (miles {mk['mile'][0]}–{mk['mile'][-1]})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
