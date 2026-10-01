"""15 standalone, post-ready graphics (Instagram / r/dataisbeautiful).

Each tells one story from the hike. Brand lavender #BBBBF2 throughout.
Output: graphics/*.png (gitignored, local only).

Usage: uv run python scripts/make_graphics.py
"""
import datetime as dt
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "graphics"
OUT.mkdir(exist_ok=True)

BRAND = "#BBBBF2"
DEEP = "#3D3D8F"
MID = "#7A7AD6"
RED = "#C0392B"
SECTIONS = ["SoCal", "Sierra", "NorCal", "Oregon", "Washington"]
SECC = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00"]
BOUNDS = [0, 702, 1092, 1694, 2146, 2660]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

daily = pl.read_parquet(BASE / "data" / "daily_report.parquet").sort("date")
d = daily.to_dicts()


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()
    print(" ", name, flush=True)


print("rendering 15 graphics...", flush=True)

# 1 — full-trail elevation profile -------------------------------------------
cl = pl.read_parquet(BASE / "data" / "route_centerline.parquet")
mi = cl["route_mi"].to_numpy()
ele = cl["ele_m"].to_numpy() * 3.28084
k = 400
filled = np.where(np.isnan(ele), np.nanmedian(ele), ele)
em = np.median(
    np.lib.stride_tricks.sliding_window_view(filled, k), axis=1)
mm = np.convolve(mi, np.ones(k) / k, mode="valid")
plt.figure(figsize=(12, 4.5))
for i in range(5):
    m = (mm >= BOUNDS[i]) & (mm <= BOUNDS[i + 1])
    plt.fill_between(mm[m], em[m], alpha=0.55, color=SECC[i], label=SECTIONS[i])
plt.plot(mm, em, color=DEEP, lw=1)
plt.xlabel("trail miles (PCTA centerline)")
plt.ylabel("elevation (ft)")
plt.title("2,653 miles of up and down — PCT elevation profile", fontsize=14, weight="bold")
plt.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
save("01_elevation_profile.png")

# 2 — calendar heatmap ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 3.2))
start = dt.date(2026, 4, 25)
grid = np.full((7, 16), np.nan)
for r in d:
    dd = dt.date.fromisoformat(r["date"])
    wk = (dd - start).days // 7
    grid[dd.weekday() % 7, wk] = r["net_mi"] if r["net_mi"] else 0
im = ax.imshow(grid, cmap="Purples", vmin=0, vmax=40, aspect="auto")
ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=9)
months_pos, months_lab = [], []
for wk in range(16):
    dd = start + dt.timedelta(days=wk * 7)
    if dd.day <= 7:
        months_pos.append(wk)
        months_lab.append(dd.strftime("%b"))
ax.set_xticks(months_pos)
ax.set_xticklabels(months_lab)
ax.set_title("Daily trail miles — one square per day (darker = more miles)",
             fontsize=13, weight="bold")
fig.colorbar(im, ax=ax, label="trail miles", shrink=0.8)
save("02_calendar.png")

# 3 — miles by hour of day, plain bars -------------------------------------------------
hh = (pl.read_parquet(BASE / "data" / "hourly.parquet")
      .join(daily.select(["date", "is_full"]), on="date")
      .filter(pl.col("is_full")).group_by("hour_local")
      .agg(pl.col("mi").median().alias("m")).sort("hour_local"))
plt.figure(figsize=(10, 5))
plt.bar(hh["hour_local"], hh["m"], color=BRAND, edgecolor="white")
plt.xlabel("hour (PT)")
plt.ylabel("median miles in that hour (full days)")
plt.title("Median miles hiked per hour of day", fontsize=13, weight="bold")
save("03_hourly.png")

# 4 — cumulative staircase, colored by section --------------------------------------
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
cum = np.cumsum([r["net_mi"] for r in d])
ideal = np.linspace(0, cum[-1], len(d))
fig, ax = plt.subplots(figsize=(10, 5))
pts = np.column_stack([np.arange(len(d)), cum])
lc = LineCollection([[pts[i], pts[i + 1]] for i in range(len(d) - 1)],
                    colors=[SECC[SECTIONS.index(d[i]["section"])] for i in range(len(d) - 1)],
                    linewidths=2.5)
ax.add_collection(lc)
ax.autoscale_view()
ax.plot(ideal, color=RED, ls="--", lw=1.5, label="even pace")
ax.set_xticks(range(0, len(d), 10),
              [d[i]["date"][5:] for i in range(0, len(d), 10)], rotation=45)
ax.set_ylabel("cumulative trail miles")
ax.set_title("Cumulative trail miles by section", fontsize=13, weight="bold")
ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=s)
                   for s, c in zip(SECTIONS, SECC)] + [ax.get_lines()[0]],
          frameon=False, ncol=3)
save("04_cumulative.png")

# 5 — grade vs speed ------------------------------------------------------------------
seg = pl.read_parquet(BASE / "data" / "segments.parquet").filter(
    pl.col("kept_h5") & (pl.col("dist_mi") * 5280 > 100)
    & (pl.col("dt_s") >= 8) & (pl.col("dt_s") <= 120)
    & (pl.col("grade").abs() < 0.4))
gg = seg.with_columns(((pl.col("grade") * 50).round() * 2).alias("gp")).group_by("gp").agg(
    pl.col("mph").median().alias("m"), pl.len().alias("n")).filter(
    pl.col("n") > 50).sort("gp")
overall = seg["mph"].median()
plt.figure(figsize=(9, 5))
plt.axhline(overall, color="black", alpha=0.4, ls="--")
plt.text(gg["gp"].max(), overall + 0.05, f"overall median {overall:.2f} mph",
         ha="right", fontsize=10)
plt.plot(gg["gp"], gg["m"], color=DEEP, lw=3, marker="o", ms=5,
         markerfacecolor=BRAND, markeredgecolor=DEEP)
plt.axvline(0, color="black", alpha=0.3, ls="--")
plt.xlabel("grade (%)")
plt.ylabel("median mph")
plt.title("Median hiking speed by trail grade", fontsize=13, weight="bold")
save("05_grade_speed.png")

# 6 — sleep vs miles, binned ------------------------------------------------------------
slp = {r["date"]: r["sleep_hrs"] for r in
       pl.read_parquet(BASE / "data" / "sleep.parquet").to_dicts()}
xb = np.array([slp.get(r["date"], np.nan) for r in d])
yb = np.array([r["net_mi"] for r in d])
ok = ~np.isnan(xb)
bins = np.arange(2, 12)
ind = np.digitize(xb[ok], bins)
means = [yb[ok][ind == i].mean() for i in range(1, len(bins))]
ns = [int((ind == i).sum()) for i in range(1, len(bins))]
plt.figure(figsize=(9, 5))
plt.bar([f"{bins[i]}–{bins[i+1]}h" for i in range(len(bins) - 1)], means,
        color=BRAND, edgecolor="white")
for i, n in enumerate(ns):
    plt.text(i, means[i] + 0.4, f"n={n}", ha="center", fontsize=9)
plt.xlabel("sleep that morning (1-hour bins)")
plt.ylabel("mean trail miles that day")
plt.title("Sleep duration vs daily miles (binned means)", fontsize=13, weight="bold")
save("06_sleep_vs_miles.png")

# 7 — start hour vs miles, stacked full/nero --------------------------------------------------
st = np.array([(int(r["start_local"][:2]) * 60 + int(r["start_local"][3:])) // 60 for r in d])
ym = np.array([r["net_mi"] for r in d])
fl = np.array([r["is_full"] for r in d])
hs = sorted(set(st))
full_sum = np.array([ym[(st == h) & fl].sum() for h in hs])
nero_sum = np.array([ym[(st == h) & ~fl].sum() for h in hs])
ns = [int((st == h).sum()) for h in hs]
plt.figure(figsize=(11, 5.5))
plt.bar([f"{h}:00" for h in hs], full_sum, color="#BBBBF2", edgecolor="white", label="full days")
plt.bar([f"{h}:00" for h in hs], nero_sum, bottom=full_sum, color="#D55E00", edgecolor="white",
        label="nero days")
for x, t, n in zip([f"{h}:00" for h in hs], full_sum + nero_sum, ns):
    plt.text(x, t + 1.5, f"n={n}", ha="center", fontsize=9)
plt.xlabel("start hour (PT)")
plt.ylabel("total trail miles started in that hour")
plt.title("Start hour vs daily miles, full and nero days stacked", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("07_early_bird.png")

# 8 — daily miles histogram, stacked by section ----------------------------------------
mm = [r["net_mi"] for r in d]
bins = np.linspace(0, 45, 19)
plt.figure(figsize=(10, 5))
plt.hist([[r["net_mi"] for r in d if r["section"] == s] for s in SECTIONS],
         bins=bins, stacked=True, label=SECTIONS,
         color=SECC, edgecolor="white")
plt.axvline(20, color=RED, lw=2.5, ls="--", label="20 mi nero line")
plt.xlabel("trail miles in a day")
plt.ylabel("days")
plt.title("Daily trail miles stacked by section", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("08_histogram.png")

# 9 — pace by section, geographical order ------------------------------------------------
med = daily.group_by("section").agg(pl.col("pace_mph").median().alias("m")).to_dicts()
med = sorted(med, key=lambda r: SECTIONS.index(r["section"]))
plt.figure(figsize=(8, 4.5))
plt.barh([r["section"] for r in med][::-1], [r["m"] for r in med][::-1],
         color=[SECC[SECTIONS.index(r["section"])] for r in med][::-1], edgecolor="white")
plt.xlabel("median pace, full days (mph)")
plt.title("Median pace on full days by section", fontsize=13, weight="bold")
save("09_section_pace.png")

# 10 — camp skyline, colored by section ----------------------------------------------------
plt.figure(figsize=(11, 4))
xx = np.arange(len(d))
yy = np.array([r["camp_end_ft"] for r in d])
for s, col in zip(SECTIONS, SECC):
    m = np.array([r["section"] == s for r in d])
    yym = np.where(m, yy, np.nan)
    plt.fill_between(xx, yym, color=col, alpha=0.75)
plt.plot(yy, color=DEEP, lw=1)
plt.xticks(range(0, len(d), 10), [d[i]["date"][5:] for i in range(0, len(d), 10)], rotation=45)
plt.ylabel("camp elevation (ft)")
plt.title("Camp elevation each night", fontsize=13, weight="bold")
save("10_camps.png")

# 12 — heart rate by section, boxplots ------------------------------------------------------
hr = pl.read_parquet(BASE / "data" / "tracks.parquet").filter(
    pl.col("heart_rate").is_not_null()).select(["date", "heart_rate"])
sec_of = {r["date"]: r["section"] for r in d}
plt.figure(figsize=(10, 5))
data = [[h for dt_, h in zip(hr["date"].to_list(), hr["heart_rate"].to_list())
         if sec_of.get(dt_) == s] for s in SECTIONS]
bp = plt.boxplot(data, tick_labels=SECTIONS, patch_artist=True, showfliers=False)
for p, c in zip(bp["boxes"], SECC):
    p.set_facecolor(c)
    p.set_alpha(0.8)
plt.ylabel("heart rate (bpm, every recorded fix)")
plt.title("Heart rate by section", fontsize=13, weight="bold")
save("12_heartbeat.png")

# 15 — cumulative climbing over the hours of the 10 biggest climbing days ------------
trk = pl.read_parquet(BASE / "data" / "tracks.parquet").filter(
    pl.col("altitude_m").is_not_null()).sort(["date", "timestamp_utc"])
cands = sorted([r for r in d if r["is_full"]], key=lambda r: -r["ascent_ft"])[:15]
big10 = []
for r in cands:
    day = trk.filter(pl.col("date") == r["date"])
    if day.height < 10:
        continue
    alt = day["altitude_m"].to_numpy() * 3.28084
    ts = np.array([x.timestamp() for x in day["timestamp_utc"].to_list()])
    hrs = (ts - ts[0]) / 3600
    climb = np.concatenate([[0], np.cumsum(np.clip(np.diff(alt), 0, None))])
    best, cur0 = 0.0, 0  # longest stretch gaining <100 ft/h (dead watch, not hiking)
    for i in range(1, len(hrs)):
        if climb[i] - climb[cur0] > 100 * (hrs[i] - hrs[cur0]):
            cur0 = i
        best = max(best, hrs[i] - hrs[cur0])
    if best <= 4:
        big10.append((r, hrs, climb))
big10 = big10[:10]
plt.figure(figsize=(11, 5.5))
for r, hrs, climb in big10:
    plt.plot(hrs, climb, color=SECC[SECTIONS.index(r["section"])], alpha=0.75, lw=2.5)
plt.xlim(0, 15)
plt.xlabel("hours since first fix of the day")
plt.ylabel("cumulative feet climbed that day")
plt.title("How the 10 biggest climbing days accumulated", fontsize=13, weight="bold")
save("15_climbing_days.png")

print("done → graphics/", flush=True)
