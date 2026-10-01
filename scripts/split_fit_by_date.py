"""Split Garmin UploadedFiles .fit files into pre / pct / post buckets by hike date.

Bucketing uses the watch-local activity date:
  1. activity_mesgs[0].local_timestamp (FIT epoch seconds -> naive local date)
  2. fallback: activity timestamp (UTC) -> date
  3. fallback: file_id time_created (UTC) -> date
  4. unparseable files -> unknown/

Writes activities/manifest.csv with one row per file (filename, dates,
device serial/product, file type, bucket, date source, error).
"""
import csv
import datetime as dt
import shutil
import sys
from pathlib import Path

from garmin_fit_sdk import Decoder, Stream

BASE = Path(__file__).resolve().parent.parent
ACT = BASE / "rawdata" / "20261001_garmin_export" / "activities"

PCT_START = dt.date(2026, 4, 25)
PCT_END = dt.date(2026, 8, 11)  # inclusive

FIT_EPOCH = dt.datetime(1989, 12, 31, tzinfo=dt.timezone.utc)


def local_date_from_fit_ts(value) -> dt.date | None:
    try:
        return (FIT_EPOCH + dt.timedelta(seconds=int(value))).date()
    except (TypeError, ValueError, OverflowError):
        return None


def extract(row: dict, fname: str) -> dict:
    """Return manifest fields for one decoded file."""
    out = {
        "filename": fname,
        "file_type": None,
        "serial_number": None,
        "garmin_product": None,
        "time_created_utc": None,
        "activity_ts_utc": None,
        "local_date": None,
        "date_source": None,
        "error": None,
    }
    try:
        msgs, errs = Decoder(Stream.from_file(str(row))).read()
        if errs:
            out["error"] = f"decoder_errors={errs[:1]}"
    except Exception as e:  # corrupt / not a FIT file
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    fid = (msgs.get("file_id_mesgs") or [{}])[0]
    out["file_type"] = fid.get("type")
    out["serial_number"] = fid.get("serial_number")
    out["garmin_product"] = fid.get("garmin_product") or fid.get("product")
    tc = fid.get("time_created")
    if isinstance(tc, dt.datetime):
        out["time_created_utc"] = tc.isoformat()

    act = (msgs.get("activity_mesgs") or [{}])[0]
    ts = act.get("timestamp")
    if isinstance(ts, dt.datetime):
        out["activity_ts_utc"] = ts.isoformat()

    # 1) watch-local date (preferred: correct across time zones)
    if act.get("local_timestamp") is not None:
        d = local_date_from_fit_ts(act["local_timestamp"])
        if d:
            out["local_date"], out["date_source"] = d.isoformat(), "activity.local_timestamp"
    # 2) UTC activity timestamp
    if out["local_date"] is None and isinstance(ts, dt.datetime):
        out["local_date"], out["date_source"] = ts.date().isoformat(), "activity.timestamp_utc"
    # 3) file creation time
    if out["local_date"] is None and isinstance(tc, dt.datetime):
        out["local_date"], out["date_source"] = tc.date().isoformat(), "file_id.time_created"
    return out


def bucket(local_date: str | None) -> str:
    if not local_date:
        return "unknown"
    d = dt.date.fromisoformat(local_date)
    if d < PCT_START:
        return "pre"
    if d <= PCT_END:
        return "pct"
    return "post"


def main() -> None:
    files = sorted(ACT.glob("*.fit"))
    print(f"Found {len(files)} .fit files", flush=True)
    for b in ("pre", "pct", "post", "unknown"):
        (ACT / b).mkdir(exist_ok=True)

    manifest_path = ACT / "manifest.csv"
    counts = {"pre": 0, "pct": 0, "post": 0, "unknown": 0}
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "bucket", "local_date", "date_source",
                "file_type", "serial_number", "garmin_product",
                "time_created_utc", "activity_ts_utc", "error",
            ],
        )
        w.writeheader()
        for i, fp in enumerate(files, 1):
            info = extract(fp, fp.name)
            b = bucket(info["local_date"])
            counts[b] += 1
            try:
                shutil.move(str(fp), str(ACT / b / fp.name))
            except Exception as e:
                b = "unknown"
                counts[b] += 1
                info["error"] = f"move_failed: {e}"
            w.writerow({"bucket": b, **info})
            if i % 1000 == 0:
                print(f"  ...{i}/{len(files)} pre={counts['pre']} pct={counts['pct']} "
                      f"post={counts['post']} unknown={counts['unknown']}", flush=True)

    print("DONE:", counts, flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
