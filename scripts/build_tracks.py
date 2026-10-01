"""Build tracks.parquet + daily_summary.parquet from pct/ activity FIT files.

- Reads the 108 activity .fit files in rawdata/.../activities/pct/
- tracks.parquet: one row per record message (all rows kept; lat/lon null
  where UltraTrac had no GPS fix). Same-day files share a `date` value.
  `distance_m` is the watch's per-file cumulative distance (resets per file).
- daily_summary.parquet: one row per date, session totals summed across
  same-day files. All values RAW as recorded (UltraTrac jumps NOT cleaned).

Usage: uv run python scripts/build_tracks.py
"""
import datetime as dt
import sys
from pathlib import Path

import polars as pl
from garmin_fit_sdk import Decoder, Stream

BASE = Path(__file__).resolve().parent.parent
PCT = BASE / "rawdata" / "20261001_garmin_export" / "activities" / "pct"
OUT = BASE / "data"

FIT_EPOCH = dt.datetime(1989, 12, 31, tzinfo=dt.timezone.utc)
SEMICIRCLE_TO_DEG = 180.0 / 2**31


def fit_local_date(msgs) -> str | None:
    act = (msgs.get("activity_mesgs") or [{}])[0]
    if act.get("local_timestamp") is not None:
        try:
            return (FIT_EPOCH + dt.timedelta(seconds=int(act["local_timestamp"]))).date().isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
    for m in (act, (msgs.get("file_id_mesgs") or [{}])[0]):
        ts = m.get("timestamp") or m.get("time_created")
        if isinstance(ts, dt.datetime):
            return ts.date().isoformat()
    return None


def parse_file(fp: Path):
    """Return (track_rows, session_rows, local_date)."""
    msgs, _errs = Decoder(Stream.from_file(str(fp))).read()
    date = fit_local_date(msgs)
    if date is None:
        return [], [], None

    tracks, sessions = [], []
    for i, r in enumerate(msgs.get("record_mesgs", [])):
        lat = r.get("position_lat")
        lon = r.get("position_long")
        tracks.append({
            "date": date,
            "source_file": fp.name,
            "point_index": i,
            "timestamp_utc": r.get("timestamp"),
            "lat": lat * SEMICIRCLE_TO_DEG if isinstance(lat, (int, float)) else None,
            "lon": lon * SEMICIRCLE_TO_DEG if isinstance(lon, (int, float)) else None,
            "altitude_m": r.get("enhanced_altitude", r.get("altitude")),
            "heart_rate": r.get("heart_rate"),
            "distance_m": r.get("distance"),  # per-file cumulative, resets per file
            "speed_m_s": r.get("enhanced_speed", r.get("speed")),
            "cadence": r.get("cadence"),
        })
    for s in msgs.get("session_mesgs", []):
        sessions.append({
            "date": date,
            "source_file": fp.name,
            "distance_m": s.get("total_distance"),
            "ascent_m": s.get("total_ascent"),
            "descent_m": s.get("total_descent"),
            "timer_s": s.get("total_timer_time"),
            "elapsed_s": s.get("total_elapsed_time"),
            "calories": s.get("total_calories"),
            "avg_hr": s.get("avg_heart_rate"),
            "max_hr": s.get("max_heart_rate"),
        })
    return tracks, sessions, date


def main() -> None:
    files = sorted(PCT.glob("*.fit"))
    print(f"Parsing {len(files)} files...", flush=True)
    all_tracks, all_sessions = [], []
    skipped = []
    for i, fp in enumerate(files, 1):
        try:
            msgs, _ = Decoder(Stream.from_file(str(fp))).read()
            if (msgs.get("file_id_mesgs") or [{}])[0].get("type") != "activity":
                continue  # only activity files; pct/ should be all activity
            t, s, d = parse_file(fp)
            if d is None:
                skipped.append(fp.name)
                continue
            all_tracks.extend(t)
            all_sessions.extend(s)
        except Exception as e:  # noqa: BLE001 - record and keep going
            skipped.append(f"{fp.name} ({type(e).__name__})")
        if i % 25 == 0:
            print(f"  ...{i}/{len(files)}", flush=True)

    OUT.mkdir(exist_ok=True)
    tracks = pl.DataFrame(all_tracks)
    tracks.write_parquet(OUT / "tracks.parquet")

    sess = pl.DataFrame(all_sessions)
    daily = (
        sess.group_by("date")
        .agg([
            pl.col("source_file").n_unique().alias("n_files"),
            pl.col("distance_m").sum().alias("distance_m"),
            pl.col("ascent_m").sum().alias("ascent_m"),
            pl.col("descent_m").sum().alias("descent_m"),
            pl.col("timer_s").sum().alias("timer_s"),
            pl.col("elapsed_s").sum().alias("elapsed_s"),
            pl.col("calories").sum().alias("calories"),
            # HR averaged across same-day files, weighted by recording time
            (pl.col("avg_hr") * pl.col("timer_s")).sum() / pl.col("timer_s").sum().alias("avg_hr"),
            pl.col("max_hr").max().alias("max_hr"),
        ])
        .sort("date")
    )
    n_points = tracks.group_by("date").len().rename({"len": "n_points"})
    n_gps = tracks.filter(pl.col("lat").is_not_null()).group_by("date").len().rename({"len": "n_gps_points"})
    daily = daily.join(n_points, on="date", how="left").join(n_gps, on="date", how="left").sort("date")
    daily.write_parquet(OUT / "daily_summary.parquet")

    print(f"tracks.parquet: {tracks.height} rows, {tracks['date'].n_unique()} days", flush=True)
    print(f"daily_summary.parquet: {daily.height} days", flush=True)
    print(f"skipped: {len(skipped)} {skipped[:5]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
