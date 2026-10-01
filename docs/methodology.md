# Methodology — what we did and why

Hike: PCT 2026-04-25 → 2026-08-11 (109 calendar days, 99 with recordings —
the other 10 are town/zero days). Watch: Garmin Forerunner 255
(serial 3429421501, registered 2026-04-07, one week before the trail) in
UltraTrac low-power mode: sparse GPS fixes (~7 s cadence, ~65% of records
with a position), 1-second-grade horrors when the fix went bad.

## 1. Source data

Garmin Takeout export (`rawdata/20261001_garmin_export/`). The 10,515 `.fit`
files live zipped in `DI_CONNECT/DI-Connect-Uploaded-Files/UploadedFiles_0-_Part{1..4}.zip`
— there are no loose FITs. Filenames (`..._<number>.fit`) carry **internal
file IDs, not dates or activity IDs** (verified: 0 of 108 hike activity IDs
match a filename), so all date/device filtering reads the FIT contents.

Ground truth for the hike window comes from
`DI-Connect-Fitness/*_summarizedActivities.json`: 108 `trail_running`
activities, 2026-04-25 → 2026-08-11, all on device 3429421501.

InReach tracks (`INREACH/*.gpx|*.kmz`, 20–30 min fix interval) and FarOut
mile markers (only recorded from NorCal onward, not yet in repo) are **not
yet integrated**.

## 2. Sorting by date (`split_fit_by_date.py`)

Each FIT was decoded (`garmin-fit-sdk`) and dated by watch-local time:
`activity.local_timestamp` → else UTC `timestamp` → else file creation time.
292 files have no timestamp at all (tiny ~365-byte FR255 state files) and
sit in `unknown/`. Result: `pre/` 8,632 · `pct/` 1,057 · `post/` 534.
Every file in `pct/` is FR255. A `manifest.csv` records per-file date,
device, file type, and date source. Of the 1,057 pct files only 108 are
`activity` type — the rest are daily wellness/monitoring files.

## 3. Parsing (`build_tracks.py`)

The 108 activity FITs → `tracks.parquet` (524,963 rows; all record messages
kept, lat/lon null where UltraTrac had no fix — 34.6% of rows) and
`daily_summary.parquet` (99 days; session totals summed across same-day
files, HR time-weighted). Semicircles → degrees (×180/2³¹). Same-day
multi-recordings (8 days, e.g. 04-28) merged under one date. Everything raw:
no cleaning at this stage.

## 4. Teleport cuts — heuristics compared (`compare_heuristics.py`)

Per-segment speed = haversine distance ÷ dt between consecutive in-file GPS
fixes (343,065 segments; raw GPS-sum 2,927.0 mi vs watch total 2,714.6 mi —
both inflated). Five (+2) rules, same data:

| Rule | Segs cut | Mi cut | Corrected | Days hit |
|---|---|---|---|---|
| H1 speed > 26.8 mph | 5,628 | 400.7 | 2,526 | 99 |
| H2 mile-in-30s (>119.9 mph) | 776 | 266.2 | 2,661 | 94 |
| H3 speed > 17.9 mph | 9,141 | 450.3 | 2,477 | 99 |
| H4 segment > 0.62 mi | 42 | 232.3 | 2,695 | 23 |
| **H5 speed > 26.8 mph AND dist > 656 ft** | **203** | **250.1** | **2,677** | 58 |
| H6 speed > 5 mph for 60 s+ (single-seg) | 17 | 12.8 | 2,914 | 12 |
| H6′ sustained > 5 mph runs | 4,211 | 88.1 | 2,839→2,615 w/ H5 | 99 |

**Why H5.** The trail is ~2,653 mi and the hiker recorded ~2,450, so any
corrected total far below ~2,600 is overcutting: H1/H3 are out (fast
descents + hitch-era motion). H2/H4/H5 all land plausible; H5 removes
nearly as much garbage as H2 while touching 4× fewer points, because the
`>656 ft` clause spares sub-200 m GPS jitter (tree-cover bounce that reads
fast but can't move mileage — a smoothing problem, not a cutting problem).

**Car rides.** The hiker believes they always paused for cars. Verified:
a vehicle would appear as *long runs* of consecutive fast fixes — the
longest such run in the whole hike is 8 segments / 0.4 mi. Conclusion: no
car segments exist in the recordings (rides happened paused or on zero
days), so no car rule is applied. The giant events (e.g. 68 + 65 mi on
05-12 in 1-second gaps at 55,000 m/s) are GPS hardware glitches, not
vehicles.

**The 5-mph idea.** "Never exceeded 5 mph for >1 min" is true of the
hiker but unusable as a cut rule: H6′ lands at 2,615 mi, below trail
length — it eats real descents and, worse, sustained jitter clusters
(positions bouncing 100+ ft every few seconds reads as "6 mph"). Speed
alone cannot separate those; the route projection (§5) can.

Applied by `apply_cuts.py` → `daily_corrected.parquet`
(`raw_watch_mi`, `gps_raw_mi`, `cut_mi`, `corrected_mi` per date).
07-21 corrects 43→41.0 mi: the monster day survives, as designed.

## 5. Route projection (`build_route.py`, `project_tracks.py`)

PCTA centerline GPX: 29 ordered section tracks, joints ≤ 14 ft, 1,235,590
pts, 2,652.90 mi total (matches filename claim). The 499 MB `Full_PCT.gpx`
(94 unordered chunks, same line within ~50 ft) was discarded as redundant.
Half-mile markers (0.5–2,655.5) sorted by mile for reference.

Every fix → nearest centerline point (cKDTree, exact; median miss 21 ft)
giving `route_mi` + `off_route_ft` (`track_route.parquet`). Daily
along-trail miles = max − min `route_mi` over kept fixes (`daily_route.parquet`,
with `gross_mi` = |Δ| sum as a jitter diagnostic and `corrected_mi` joined).

**Result: net 2,442.8 mi vs the ~2,450 anchor (0.3%).** Cross-check:
2,652.9 − 61 (unfinished: Rainy Pass, Ptarmigan fire closure) − 8/67/98
(skipped 05-28/06-23/07-24: fire + tramily) ≈ 2,419 + backtrack
double-counts ≈ 2,443. ✓.

Days >50% off-route (alternate/town candidates): 06-13 (town day, 0.2
trail-mi recorded), 07-19, 08-02, 05-18, 05-05, 06-06, 05-08. Day-to-day
chaining also caught a 9-mi southbound day (07-16, plausible) and the
three skips above.

## 6. Not yet done

- Jitter smoothing (sub-656 ft bounce, ~227 mi residual in the GPS sum) —
  moot for mileage (net cancels it) but needed for clean track maps.
- Elevation/HR data untouched (as recorded).
- InReach + FarOut integration; alternate-day mileage method
  (route vs GPS-sum per flagged day).
- The `unknown/` bucket and pre/postpct files are unexamined beyond sorting.

## 7. Annotations & per-segment table

- `annotations/events.csv` (versioned): hiker-confirmed timeline — Whitney
  06-06, SHR split 06-14 / Truckee rejoin 06-22, 67-mi catch-up skip 06-23,
  Sisters fire skip 07-24, Rainy Pass terminus 08-11. Tentative rows carry
  `status=tentative`. Private context never enters this file or the report.
- `scripts/build_segments.py` → `data/segments.parquet`: one row per
  in-file fix pair (dist, dt, mph, dAlt, grade, H5 flag, local hour) —
  powers grade-vs-speed and time-of-day analyses in the report.
