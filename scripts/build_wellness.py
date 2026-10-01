"""Sleep + morning HRV tables from the Garmin wellness export.

- data/sleep.parquet: one row per calendarDate (longest window kept):
  sleep start/end (GMT), hours by stage, overall/quality/recovery scores,
  sleep stress, restlessness. calendarDate ≈ wake-up date → join to the
  SAME date in daily tables (that night's sleep fuels that day's hike).
- data/hrv.parquet: morning HRV + baseline band + status, resting HR.

Usage: uv run python scripts/build_wellness.py
"""
import glob
import json
import sys
from pathlib import Path

import polars as pl

BASE = Path(__file__).resolve().parent.parent
FULL = next((BASE / "rawdata" / "20261001_garmin_export" / "full_data").glob("*"))
WELL = FULL / "DI_CONNECT" / "DI-Connect-Wellness"


def load(pattern: str):
    rows = []
    for fp in sorted(glob.glob(str(WELL / pattern))):
        with open(fp) as f:
            data = json.load(f)
        rows.extend(data if isinstance(data, list) else [data])
    return rows


def main() -> None:
    sleeps = load("*_sleepData.json")
    by_date = {}
    for s in sleeps:
        d = s.get("calendarDate")
        if not d:
            continue
        deep = s.get("deepSleepSeconds", 0) or 0
        light = s.get("lightSleepSeconds", 0) or 0
        rem = s.get("remSleepSeconds", 0) or 0
        awake = s.get("awakeSleepSeconds", 0) or 0
        unm = s.get("unmeasurableSeconds", 0) or 0
        tot = deep + light + rem + awake
        sc = s.get("sleepScores") or {}
        row = {
            "date": d,
            "sleep_start_gmt": s.get("sleepStartTimestampGMT"),
            "sleep_end_gmt": s.get("sleepEndTimestampGMT"),
            "sleep_hrs": round(tot / 3600, 2),
            "deep_hrs": round(deep / 3600, 2),
            "rem_hrs": round(rem / 3600, 2),
            "awake_hrs": round(awake / 3600, 2),
            "unmeasurable_hrs": round(unm / 3600, 2),
            "score_overall": sc.get("overallScore"),
            "score_quality": sc.get("qualityScore"),
            "score_recovery": sc.get("recoveryScore"),
            "sleep_stress": s.get("avgSleepStress"),
            "restless": s.get("restlessMomentCount"),
            "awakenings": s.get("awakeCount"),
        }
        if d not in by_date or row["sleep_hrs"] > by_date[d]["sleep_hrs"]:
            by_date[d] = row  # longest window wins (skips naps/retro dups)
    sleep = pl.DataFrame(list(by_date.values()), schema={
        "date": pl.String, "sleep_start_gmt": pl.String, "sleep_end_gmt": pl.String,
        "sleep_hrs": pl.Float64, "deep_hrs": pl.Float64, "rem_hrs": pl.Float64,
        "awake_hrs": pl.Float64, "unmeasurable_hrs": pl.Float64,
        "score_overall": pl.Int64, "score_quality": pl.Int64,
        "score_recovery": pl.Int64, "sleep_stress": pl.Float64,
        "restless": pl.Int64, "awakenings": pl.Int64,
    }).sort("date")
    sleep.write_parquet(BASE / "data" / "sleep.parquet")

    hrows = []
    for h in load("*_healthStatusData.json"):
        d = h.get("calendarDate")
        if not d:
            continue
        m = {x.get("type"): x for x in (h.get("metrics") or [])}
        hrv, hr = m.get("HRV", {}), m.get("HR", {})
        hrows.append({
            "date": d,
            "hrv": hrv.get("value"),
            "hrv_lo": hrv.get("baselineLowerLimit"),
            "hrv_hi": hrv.get("baselineUpperLimit"),
            "hrv_status": hrv.get("status"),
            "rest_hr": hr.get("value"),
        })
    hrvt = pl.DataFrame(hrows).unique("date", keep="last").sort("date")
    hrvt.write_parquet(BASE / "data" / "hrv.parquet")

    hike = [f"2026-{m:02d}-{d:02d}" for m in (4, 5, 6, 7, 8)
            for d in range(1, 32)]
    hike = [x for x in hike if "2026-04-25" <= x <= "2026-08-11"]
    print(f"sleep nights in hike window: "
          f"{sleep.filter(pl.col('date').is_in(hike)).height} / {len(hike)}")
    print(f"hrv mornings in hike window: "
          f"{hrvt.filter(pl.col('date').is_in(hike)).height} / {len(hike)}")
    s = sleep.filter(pl.col("date").is_in(hike))
    print(f"avg sleep: {s['sleep_hrs'].mean():.2f} h, "
          f"avg score: {s['score_overall'].mean():.0f}")


if __name__ == "__main__":
    sys.exit(main())
