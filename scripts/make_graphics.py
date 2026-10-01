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
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False,
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
ax.set_yticks(range(7), ["M", "T", "W", "T", "F", "S", "S"])
ax.set_title("Every day of the hike, one square (darker = more miles)", fontsize=13, weight="bold")
ax.set_xticks([])
fig.colorbar(im, ax=ax, label="trail miles", shrink=0.8)
save("02_calendar.png")

# 3 — 24-hour radial clock -----------------------------------------------------
hh = (pl.read_parquet(BASE / "data" / "hourly.parquet")
      .join(daily.select(["date", "is_full"]), on="date")
      .filter(pl.col("is_full")).group_by("hour_local")
      .agg(pl.col("mi").median().alias("m")).sort("hour_local"))
hrs, vals = hh["hour_local"].to_numpy(), hh["m"].to_numpy()
theta = (hrs / 24) * 2 * np.pi - np.pi / 2
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_zero_location("N")
bars = ax.bar(theta, vals, width=2 * np.pi / 24 * 0.9, color=BRAND, edgecolor=DEEP)
ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
ax.set_xticklabels(["12a", "3a", "6a", "9a", "12p", "3p", "6p", "9p"])
ax.set_title("The hiker's clock — median miles hiked in each hour", weight="bold", pad=20)
save("03_clock.png")

# 4 — cumulative staircase vs perfect pace --------------------------------------
cum = np.cumsum([r["net_mi"] for r in d])
ideal = np.linspace(0, cum[-1], len(d))
plt.figure(figsize=(10, 5))
plt.fill_between(range(len(d)), cum, alpha=0.35, color=BRAND)
plt.plot(cum, color=DEEP, lw=2, label="actual")
plt.plot(ideal, color=RED, ls="--", lw=1.5, label="perfectly even pace")
plt.xticks(range(0, len(d), 10), [d[i]["date"][5:] for i in range(0, len(d), 10)], rotation=45)
plt.ylabel("cumulative trail miles")
plt.title("Cumulative trail miles vs even pace", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("04_cumulative.png")

# 5 — grade vs speed, dark poster ----------------------------------------------
seg = pl.read_parquet(BASE / "data" / "segments.parquet").filter(
    pl.col("kept_h5") & (pl.col("dist_mi") * 5280 > 100) & (pl.col("dt_s") <= 120)
    & (pl.col("grade").abs() < 0.4))
gg = seg.with_columns(((pl.col("grade") * 50).round() * 2).alias("gp")).group_by("gp").agg(
    pl.col("mph").median().alias("m"), pl.len().alias("n")).filter(
    pl.col("n") > 50).sort("gp")
plt.figure(figsize=(9, 5), facecolor="black")
ax = plt.gca()
ax.set_facecolor("black")
ax.plot(gg["gp"], gg["m"], color=BRAND, lw=3, marker="o", ms=5)
ax.axvline(0, color="white", alpha=0.3, ls="--")
ax.set_xlabel("grade (%)", color="white")
ax.set_ylabel("median mph", color="white")
ax.set_title("Median hiking speed by trail grade", fontsize=15, weight="bold", color="white")
ax.tick_params(colors="white")
save("05_grade_speed.png")

# 6 — sleep vs miles ------------------------------------------------------------
slp = {r["date"]: r["sleep_hrs"] for r in
       pl.read_parquet(BASE / "data" / "sleep.parquet").to_dicts()}
x = [slp.get(r["date"], np.nan) for r in d]
y = [r["net_mi"] for r in d]
plt.figure(figsize=(8, 5))
plt.scatter(x, y, c=[SECC[SECTIONS.index(r["section"])] for r in d], s=60, alpha=0.8, edgecolors="white")
ok = ~np.isnan(x)
b, a = np.polyfit(np.array(x)[ok], np.array(y)[ok], 1)
xs = np.linspace(min(np.array(x)[ok]), max(np.array(x)[ok]), 50)
plt.plot(xs, a + b * xs, color=RED, lw=2)
plt.xlabel("sleep that morning (hours)")
plt.ylabel("trail miles that day")
plt.title("Sleep duration vs daily miles (r = 0.03)", fontsize=13, weight="bold")
save("06_sleep_vs_miles.png")

# 7 — start time vs miles -------------------------------------------------------
st = [(int(r["start_local"][:2]) * 60 + int(r["start_local"][3:])) for r in d]
plt.figure(figsize=(9, 5))
plt.scatter(st, [r["net_mi"] for r in d],
            c=[SECC[SECTIONS.index(r["section"])] for r in d], s=55, alpha=0.8, edgecolors="white")
plt.xlabel("start time (minutes after midnight PT)")
plt.ylabel("trail miles")
plt.title("Start time vs daily miles", fontsize=13, weight="bold")
save("07_early_bird.png")

# 8 — daily miles histogram ------------------------------------------------------
mm = [r["net_mi"] for r in d]
plt.figure(figsize=(9, 5))
plt.hist(mm, bins=20, color=BRAND, edgecolor="white")
plt.axvline(20, color=RED, lw=2.5, ls="--", label="20 mi nero line")
plt.axvline(np.mean(mm), color=DEEP, lw=2, label=f"mean {np.mean(mm):.1f}")
plt.xlabel("trail miles in a day")
plt.ylabel("days")
plt.title("Distribution of daily trail miles", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("08_histogram.png")

# 9 — pace by section --------------------------------------------------------------
med = daily.group_by("section").agg(pl.col("pace_mph").median().alias("m")).to_dicts()
med = sorted(med, key=lambda r: -r["m"])
plt.figure(figsize=(8, 4.5))
plt.barh([r["section"] for r in med], [r["m"] for r in med],
         color=[SECC[SECTIONS.index(r["section"])] for r in med], edgecolor="white")
plt.xlabel("median pace, full days (mph)")
plt.title("Median pace on full days by section", fontsize=13, weight="bold")
save("09_section_pace.png")

# 10 — camp skyline -----------------------------------------------------------------
plt.figure(figsize=(11, 4))
plt.fill_between(range(len(d)), [r["camp_end_ft"] for r in d], color=BRAND, alpha=0.8)
plt.plot([r["camp_end_ft"] for r in d], color=DEEP, lw=1)
plt.xticks(range(0, len(d), 10), [d[i]["date"][5:] for i in range(0, len(d), 10)], rotation=45)
plt.ylabel("camp elevation (ft)")
plt.title("Camp elevation each night", fontsize=13, weight="bold")
save("10_camps.png")

# 11 — breaks gantt ------------------------------------------------------------------
brk = pl.read_parquet(BASE / "data" / "breaks.parquet").sort("date").to_dicts()
days = sorted(set(b["date"] for b in brk))
plt.figure(figsize=(10, max(3, len(days) * 0.28)))
for i, dd in enumerate(days):
    for b in [x for x in brk if x["date"] == dd]:
        s = int(b["start_local"][:2]) * 60 + int(b["start_local"][3:])
        plt.barh(i, min(b["minutes"], 1440 - s), left=s, height=0.6, color=MID)
plt.yticks(range(len(days)), [x[5:] for x in days], fontsize=8)
plt.xlim(0, 1440)
plt.xlabel("time of day (PT — overnight breaks run off the right edge)")
plt.title("Detected breaks by day and time of day",
          fontsize=13, weight="bold")
save("11_breaks.png")

# 12 — heart rate by section ------------------------------------------------------------
hr = pl.read_parquet(BASE / "data" / "tracks.parquet").filter(
    pl.col("heart_rate").is_not_null()).select(["date", "heart_rate"])
sec_of = {r["date"]: r["section"] for r in d}
plt.figure(figsize=(9, 5))
data = [[h for dt_, h in zip(hr["date"].to_list(), hr["heart_rate"].to_list())
         if sec_of.get(dt_) == s] for s in SECTIONS]
parts = plt.violinplot(data, showmedians=True)
for p, c in zip(parts["bodies"], SECC):
    p.set_facecolor(c)
    p.set_alpha(0.8)
plt.xticks(range(1, 6), SECTIONS)
plt.ylabel("heart rate (bpm, every recorded fix)")
plt.title("Heart rate distribution by section", fontsize=13, weight="bold")
save("12_heartbeat.png")

# 13 — skips waterfall ------------------------------------------------------------------
rmax = [r["route_max_mi"] for r in d]
rmin = [r["route_min_mi"] for r in d]
plt.figure(figsize=(11, 4.5))
plt.plot(rmax, color=DEEP, lw=2)
for i in range(1, len(d)):
    if rmin[i] - rmax[i - 1] > 5:
        plt.annotate("", xy=(i, rmin[i]), xytext=(i - 1, rmax[i - 1]),
                     arrowprops={"arrowstyle": "<->", "color": RED, "lw": 2})
        plt.text(i, (rmin[i] + rmax[i - 1]) / 2, f" +{rmin[i]-rmax[i-1]:.0f}mi skip",
                 color=RED, fontsize=9, va="center")
plt.xlabel("hiking day #")
plt.ylabel("northernmost trail mile")
plt.title("Fire closures and catching up — the miles I didn't walk", fontsize=13, weight="bold")
save("13_skips.png")

# 14 — 109-day strip -----------------------------------------------------------------------
start = dt.date(2026, 4, 25)
have = {r["date"] for r in d}
cats, labels = [], []
for i in range(109):
    day = str(start + dt.timedelta(days=i))
    cats.append(0 if day not in have else 1)
fig, ax = plt.subplots(figsize=(12, 1.6))
ax.imshow(np.array(cats)[None, :], cmap="Pastel2", aspect="auto")
ax.set_yticks([])
ax.set_xticks([0, 30, 60, 108])
ax.set_xticklabels(["Apr 25", "May 25", "Jun 24", "Aug 11"])
ax.set_title("109 days: green = hiking, gray = town/zero", fontsize=13, weight="bold")
save("14_strip.png")

# 15 — hardest days top 10 -------------------------------------------------------------------
top = sorted(d, key=lambda r: -r["net_mi"])[:10]
plt.figure(figsize=(9, 5))
plt.barh([f"{r['date'][5:]}" for r in top][::-1], [r["net_mi"] for r in top][::-1],
         color=BRAND, edgecolor="white")
for i, r in enumerate(top[::-1]):
    plt.text(r["net_mi"] + 0.3, i, f"{r['ascent_ft']:,} ft up", va="center", fontsize=9)
plt.xlabel("trail miles")
plt.title("Longest 10 days with elevation climbed", fontsize=13, weight="bold")
save("15_hardest.png")

print("done → graphics/", flush=True)
