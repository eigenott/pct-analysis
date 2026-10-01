"""Standalone graphics part 2 (16-30): section-flavored, professional titles.

All landscape, aspect no more extreme than 16:9. Output: graphics/*.png.

Usage: uv run python scripts/make_graphics2.py
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

SECTIONS = ["SoCal", "Sierra", "NorCal", "Oregon", "Washington"]
SECC = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00"]
BOUNDS = [0, 702, 1092, 1694, 2146, 2660]
DEEP = "#333333"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "serif", "axes.spines.top": False,
    "axes.spines.right": False,
})

daily = pl.read_parquet(BASE / "data" / "daily_report.parquet").sort("date")
d = daily.to_dicts()
sec_of = {r["date"]: r["section"] for r in d}


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()
    print(" ", name, flush=True)


print("rendering graphics 16-30...", flush=True)

# 16 — day-type composition per section ----------------------------------------
cats = ["full (≥20 mi)", "nero (<20 mi)"]
vals = {c: [] for c in cats}
for s in SECTIONS:
    sd = [r for r in d if r["section"] == s]
    vals[cats[0]].append(sum(1 for r in sd if r["is_full"]))
    vals[cats[1]].append(sum(1 for r in sd if r["is_nero"]))
plt.figure(figsize=(10, 5))
b = np.zeros(5)
for c, col in zip(cats, ["#888888", "#D55E00"]):
    plt.bar(SECTIONS, vals[c], bottom=b, label=c, color=col)
    b += np.array(vals[c])
plt.ylabel("days")
plt.title("Day types by section", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("16_day_types.png")

# 17 — daily miles, small multiples by section ----------------------------------
fig, axes = plt.subplots(5, 1, figsize=(11, 7), sharex=False)
maxn = max(sum(1 for r in d if r["section"] == s) for s in SECTIONS)
for ax, s, col in zip(axes, SECTIONS, SECC):
    sd = [r for r in d if r["section"] == s]
    ax.bar(range(len(sd)), [r["net_mi"] for r in sd], color=col)
    ax.set_ylabel(s, rotation=0, ha="right", va="center", fontsize=10)
    ax.set_xticks([])
    ax.set_ylim(0, 45)
    ax.set_xlim(-0.6, maxn - 0.4)
fig.suptitle("Daily miles by section (equal-width panels)", fontsize=13, weight="bold")
save("17_panels_miles.png")

# 18 — pace range by section (median dot + IQR bar) ---------------------------------
plt.figure(figsize=(10, 5))
for i, (s, col) in enumerate(zip(SECTIONS, SECC)):
    v = np.array([r["pace_mph"] for r in d if r["section"] == s and r["is_full"]])
    q = np.percentile(v, [25, 50, 75])
    plt.errorbar(q[1], i, xerr=[[q[1] - q[0]], [q[2] - q[1]]],
                 fmt="o", color=col, ecolor=col, elinewidth=4, ms=10,
                 capsize=8, capthick=3)
plt.yticks(range(5), SECTIONS)
plt.xlabel("pace on full days (mph) — dot = median, bar = middle 50%")
plt.title("Pace range by section", fontsize=13, weight="bold")
save("18_pace_dist.png")

# 19 — start times by section, violins -------------------------------------------------
plt.figure(figsize=(10, 5.5))
data19 = []
for s in SECTIONS:
    mins = []
    for r in d:
        if r["section"] == s:
            mins.append(int(r["start_local"][:2]) * 60 + int(r["start_local"][3:]))
    data19.append(np.array(mins))
parts = plt.violinplot(data19, showmedians=True, widths=0.8)
for p, c in zip(parts["bodies"], SECC):
    p.set_facecolor(c)
    p.set_alpha(0.75)
parts["cmedians"].set_color("black")
parts["cmedians"].set_linewidth(2)
plt.xticks(range(1, 6), SECTIONS)
plt.ylabel("start time (PT)")
plt.gca().yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)//60:02d}:{int(v)%60:02d}"))
plt.title("Start-time distributions by section (black = median)", fontsize=13, weight="bold")
save("19_starts.png")

# 20 — camp elevations by section (strip + median) -----------------------------------
plt.figure(figsize=(10, 5))
for i, (s, col) in enumerate(zip(SECTIONS, SECC)):
    v = np.array([r["camp_end_ft"] for r in d if r["section"] == s])
    j = np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
    plt.scatter(v, np.full(len(v), i) + j, color=col, s=44, alpha=0.85,
                edgecolors="white", zorder=3)
    plt.plot([np.median(v)], [i], marker="|", ms=24, mew=3, color="black")
plt.yticks(range(5), SECTIONS)
plt.xlabel("camp elevation (ft; black tick = median)")
plt.title("Camp elevations by section", fontsize=13, weight="bold")
save("20_camps_range.png")

# 21 — sleep by section (mean ± SD) --------------------------------------------------------
slp = {r["date"]: r["sleep_hrs"] for r in
       pl.read_parquet(BASE / "data" / "sleep.parquet").to_dicts()}
plt.figure(figsize=(10, 5))
means, sds = [], []
for i, (s, col) in enumerate(zip(SECTIONS, SECC)):
    v = np.array([slp[r["date"]] for r in d
                  if r["section"] == s and r["date"] in slp and slp[r["date"]] is not None])
    means.append(v.mean())
    sds.append(v.std())
    plt.bar(i, v.mean(), yerr=v.std(), color=col, width=0.6,
            ecolor="black", capsize=8, error_kw={"lw": 2})
    plt.text(i, v.mean() + v.std() + 0.1, f"{v.mean():.2f} h", ha="center", fontsize=10)
plt.xticks(range(5), SECTIONS)
plt.ylabel("sleep (hours, mean ± SD)")
plt.title("Mean sleep by section", fontsize=13, weight="bold")
save("21_sleep.png")

# 22 — daily ascent, small multiples, shared y -----------------------------------------------
fig, axes = plt.subplots(5, 1, figsize=(11, 7), sharey=True)
for ax, s, col in zip(axes, SECTIONS, SECC):
    sd = [r for r in d if r["section"] == s]
    ax.fill_between(range(len(sd)), [r["ascent_ft"] for r in sd], color=col, alpha=0.7)
    ax.set_ylabel(s, rotation=0, ha="right", va="center", fontsize=10)
    ax.set_xticks([])
fig.suptitle("Daily elevation climbed, one panel per section (ft)", fontsize=13, weight="bold")
save("22_panels_ascent.png")

# 23 — HRV trend with section bands ------------------------------------------------------------
hv = pl.read_parquet(BASE / "data" / "hrv.parquet").filter(
    (pl.col("date") >= "2026-04-25") & (pl.col("date") <= "2026-08-11")
).sort("date").to_dicts()
dates = [r["date"] for r in hv]
x = np.arange(len(dates))
sec_by_date = sec_of
plt.figure(figsize=(11, 5))
UNK = "#CCCCCC"
cur, start = sec_by_date.get(dates[0]), 0
for i, dd in enumerate(dates + [None]):
    s = sec_by_date.get(dd) if dd else None
    if s != cur:
        col = SECC[SECTIONS.index(cur)] if cur in SECTIONS else UNK
        plt.axvspan(start - 0.5, i - 0.5, color=col, alpha=0.12)
        cur, start = s, i
plt.plot(x, [r["hrv"] for r in hv], color=DEEP, lw=1.5, marker="o", ms=3)
plt.xticks(range(0, len(dates), 15), [dates[i][5:] for i in range(0, len(dates), 15)], rotation=45)
plt.ylabel("morning HRV (ms)")
plt.title("Morning HRV across the hike (bands = sections)", fontsize=13, weight="bold")
save("23_hrv.png")

# 24 — break time by section -----------------------------------------------------------------------
brk = pl.read_parquet(BASE / "data" / "breaks.parquet").to_dicts()
for b in brk:
    b["section"] = sec_of.get(b["date"], "?")
plt.figure(figsize=(10, 5))
n = [sum(1 for b in brk if b["section"] == s) for s in SECTIONS]
mins = [sum(b["minutes"] for b in brk if b["section"] == s) for s in SECTIONS]
x = np.arange(5)
plt.bar(x - 0.2, n, width=0.4, label="break count", color=SECC)
plt.bar(x + 0.2, np.array(mins) / 60, width=0.4, label="total hours",
        color=SECC, alpha=0.45, edgecolor=DEEP)
plt.xticks(x, SECTIONS)
plt.ylabel("count / hours")
plt.title("Detected breaks by section", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("24_breaks.png")

# 25 — nero rate by section ----------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
totals = [sum(1 for r in d if r["section"] == s) for s in SECTIONS]
neros = [sum(1 for r in d if r["section"] == s and r["is_nero"]) for s in SECTIONS]
rate = [100 * n / t for n, t in zip(neros, totals)]
plt.bar(SECTIONS, rate, color=SECC, edgecolor="white")
plt.ylabel("% of hiking days under 20 mi")
plt.title("Nero rate by section (desert excluded by definition)", fontsize=13, weight="bold")
for s, v, n in zip(SECTIONS, rate, neros):
    plt.text(s, v + 0.4, f"{n}", ha="center", fontsize=10)
save("25_neros.png")

# 26 — morning vs afternoon slopegraph per section ---------------------------------------
hh = pl.read_parquet(BASE / "data" / "hourly.parquet")
plt.figure(figsize=(10, 5.5))
for s, col in zip(SECTIONS, SECC):
    dd = [r["date"] for r in d if r["section"] == s and r["is_full"]]
    sub = hh.filter(pl.col("date").is_in(dd))
    am = sub.filter(pl.col("hour_local") < 12).group_by("date").agg(
        pl.col("mi").sum().alias("m"))["m"].median()
    pm = sub.filter(pl.col("hour_local") >= 12).group_by("date").agg(
        pl.col("mi").sum().alias("m"))["m"].median()
    plt.plot([0, 1], [am, pm], label=f"{s} ({am:.1f}→{pm:.1f})", color=col,
             lw=3, marker="o", ms=8)
plt.xticks([0, 1], ["morning (before noon)", "afternoon"])
plt.ylabel("median miles in that half (full days)")
plt.title("Morning people? Afternoon people? By section", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("26_hourly_sections.png")

# 27 — grade × section speed heatmap -------------------------------------------------------
seg = pl.read_parquet(BASE / "data" / "segments.parquet").filter(
    pl.col("kept_h5") & (pl.col("dist_mi") * 5280 > 100)
    & (pl.col("dt_s") >= 8) & (pl.col("dt_s") <= 120)
    & (pl.col("grade").abs() < 0.16))
seg = seg.with_columns(((pl.col("grade") * 50).round() * 2).alias("gp"))
ssec = seg.join(pl.read_parquet(BASE / "data" / "daily_report.parquet").select(["date", "section"]),
                on="date", how="left")
piv = (ssec.group_by(["section", "gp"]).agg(pl.col("mph").median().alias("m"))
       .to_dicts())
grades = sorted(set(r["gp"] for r in piv))
mat = np.full((5, len(grades)), np.nan)
for r in piv:
    mat[SECTIONS.index(r["section"]), grades.index(r["gp"])] = r["m"]
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=2, vmax=4)
ax.set_yticks(range(5), SECTIONS)
ax.set_xticks(range(0, len(grades), 2), [f"{grades[i]:.0f}%" for i in range(0, len(grades), 2)])
ax.set_xlabel("grade (%)")
fig.colorbar(im, ax=ax, label="median mph", shrink=0.85)
ax.set_title("Speed by grade and section (darker = faster)", fontsize=13, weight="bold")
save("27_grade_sections.png")

# 28 — cumulative ascent staircase -------------------------------------------------------------------------------
plt.figure(figsize=(10, 5.5))
cum = np.cumsum([r["ascent_ft"] for r in d]) / 5280
plt.fill_between(range(len(d)), cum, alpha=0.3, color="#7A7AD6")
plt.plot(cum, color=DEEP, lw=2)
for i, (s, col) in enumerate(zip(SECTIONS, SECC)):
    idx = [j for j, r in enumerate(d) if r["section"] == s]
    if idx:
        plt.axvspan(idx[0], idx[-1], color=col, alpha=0.12)
plt.xticks(range(0, len(d), 10), [d[i]["date"][5:] for i in range(0, len(d), 10)], rotation=45)
plt.ylabel("cumulative miles climbed (mi)")
plt.title("Every foot of the ~81 miles climbed", fontsize=13, weight="bold")
save("28_staircase.png")

# 29 — highest ground per section: trail vs camp -------------------------------------------
cl = pl.read_parquet(BASE / "data" / "route_centerline.parquet")
bounds = [0, 702, 1092, 1694, 2146, 3000]
trail_max, camp_max = [], []
for i in range(5):
    e = cl.filter((pl.col("route_mi") >= bounds[i]) & (pl.col("route_mi") < bounds[i + 1]))["ele_m"] * 3.28084
    trail_max.append(float(e.max()))
    v = np.array([r["camp_end_ft"] for r in d if r["section"] == SECTIONS[i]])
    camp_max.append(float(v.max()))
x = np.arange(5)
plt.figure(figsize=(10, 5.5))
plt.bar(x, trail_max, width=0.6, color=SECC, edgecolor="white", label="highest trail point")
plt.scatter(x, camp_max, s=120, color="black", zorder=3, label="highest camp")
for i, (t, c) in enumerate(zip(trail_max, camp_max)):
    plt.text(i, t + 150, f"{t:,.0f} ft", ha="center", fontsize=9)
plt.xticks(x, SECTIONS)
plt.ylabel("elevation (ft)")
plt.title("Highest trail point vs highest camp, by section", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("29_ranges.png")

# 30 — top 12 climbs ------------------------------------------------------------------------------------------------------
cb = pl.read_parquet(BASE / "data" / "climbs.parquet").head(12).to_dicts()
towns = pl.read_csv(BASE / "reference" / "towns.csv").to_dicts()


def nearest_town(mid):
    return min(towns, key=lambda t: abs(t["mile"] - mid))["name"]


plt.figure(figsize=(10, 6))
labels = [f"mi {r['start_mi']:.0f}–{r['end_mi']:.0f} ({r['ft_per_mi']} ft/mi)" for r in cb]
plt.barh(labels[::-1], [r["gain_ft"] for r in cb][::-1],
         color=[SECC[SECTIONS.index(r["section"])] for r in cb][::-1], edgecolor="white")
for i, r in enumerate(cb[::-1]):
    plt.text(60, i, nearest_town((r["start_mi"] + r["end_mi"]) / 2),
             va="center", fontsize=9, color="white", weight="bold")
plt.xlabel("net elevation gain (ft)")
plt.title("The 12 biggest sustained climbs (2,000 ft+, 600 ft dip tolerance)",
          fontsize=13, weight="bold")
save("30_climbs.png")

print("done → graphics/", flush=True)
