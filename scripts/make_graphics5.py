"""Standalone graphics part 5 (52-55): multi-panel composites + town rhythm.

All landscape, aspect no more extreme than 16:9, professional titles.
Output: graphics/*.png (gitignored, local only).

Usage: uv run python scripts/make_graphics5.py
"""
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
DEEP = "#333333"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

daily = pl.read_parquet(BASE / "data" / "daily_report.parquet").sort("date")
d = daily.to_dicts()
dates = [r["date"] for r in d]
xt = (range(0, len(d), 10), [dates[i][5:] for i in range(0, len(d), 10)])


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()
    print(" ", name, flush=True)


print("rendering graphics 52-55...", flush=True)

# 52 — the season at a glance: miles + cumulative + camps ---------------------------------
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
a1.bar(range(len(d)), [r["net_mi"] for r in d],
       color=[SECC[SECTIONS.index(r["section"])] for r in d])
a1.set_ylabel("trail miles")
a1.set_title("The season at a glance: daily miles, cumulative total, camp elevation",
             fontsize=14, weight="bold")
a2.plot(np.cumsum([r["net_mi"] for r in d]), color=DEEP, lw=2)
a2.set_ylabel("cumulative mi")
a3.fill_between(range(len(d)), [r["camp_end_ft"] for r in d], color="#BBBBF2", alpha=0.9)
a3.plot([r["camp_end_ft"] for r in d], color=DEEP, lw=1)
a3.set_ylabel("camp (ft)")
a3.set_xticks(*xt)
plt.setp(a3.get_xticklabels(), rotation=45)
save("52_season.png")

# 53 — recovery dashboard: sleep + HRV + resting HR ------------------------------------------------
slp = pl.read_parquet(BASE / "data" / "sleep.parquet").filter(
    (pl.col("date") >= "2026-04-25") & (pl.col("date") <= "2026-08-11")).sort("date")
hv = pl.read_parquet(BASE / "data" / "hrv.parquet").filter(
    (pl.col("date") >= "2026-04-25") & (pl.col("date") <= "2026-08-11")).sort("date")
fig, (b1, b2, b3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
sx = range(len(slp))
b1.bar(sx, slp["sleep_hrs"], color="#7A7AD6", edgecolor="white")
b1.set_ylabel("sleep (h)")
b1.set_title("Recovery dashboard: sleep, morning HRV, resting heart rate",
             fontsize=14, weight="bold")
b2.plot(sx, hv["hrv"], color="#009E73", lw=2, marker="o", ms=3)
b2.set_ylabel("HRV (ms)")
b3.plot(sx, hv["rest_hr"], color="#C0392B", lw=2, marker="o", ms=3)
b3.set_ylabel("rest HR (bpm)")
b3.set_xticks(range(0, len(slp), 10), [slp["date"][i][5:] for i in range(0, len(slp), 10)],
              rotation=45)
save("53_recovery.png")

# 54 — movement dashboard: grade, hours, elevation -----------------------------------------------------------------
seg = pl.read_parquet(BASE / "data" / "segments.parquet").filter(
    pl.col("kept_h5") & (pl.col("dist_mi") * 5280 > 100)
    & (pl.col("dt_s") >= 8) & (pl.col("dt_s") <= 300)
    & (pl.col("grade").abs() < 0.3))
gg = (seg.with_columns(((pl.col("grade") * 50).round() * 2).alias("gp"))
      .group_by("gp").agg(pl.col("mph").median().alias("m"),
                           pl.len().alias("n"))
      .filter(pl.col("n") > 50).sort("gp"))
hh = (pl.read_parquet(BASE / "data" / "hourly.parquet")
      .join(daily.select(["date", "is_full"]), on="date")
      .filter(pl.col("is_full")).group_by("hour_local")
      .agg(pl.col("mi").median().alias("m")).sort("hour_local"))
fig, (c1, c2) = plt.subplots(1, 2, figsize=(12, 5.5))
c1.plot(gg["gp"], gg["m"], color=DEEP, lw=2.5, marker="o", ms=4,
        markerfacecolor="#BBBBF2", markeredgecolor=DEEP)
c1.axvline(0, color="black", alpha=0.3, ls="--")
c1.set_xlabel("grade (%)")
c1.set_ylabel("median mph")
c1.set_title("Speed by grade", fontsize=12, weight="bold")
c2.bar(hh["hour_local"], hh["m"], color="#BBBBF2", edgecolor="white")
c2.set_xlabel("hour (PT)")
c2.set_ylabel("median miles")
c2.set_title("Miles by hour of day", fontsize=12, weight="bold")
fig.suptitle("How the hiking moves", fontsize=14, weight="bold")
save("54_movement.png")

# 55 — resupply rhythm: longest town-to-town stretches --------------------------------------------------------------------
tw = pl.read_csv(BASE / "reference" / "towns.csv").sort("mile").to_dicts()
gaps = []
for a, b in zip(tw, tw[1:]):
    gaps.append((b["mile"] - a["mile"], f"{a['name']} → {b['name']}"))
gaps.sort(reverse=True)
top = gaps[:15]
plt.figure(figsize=(11, 6))
plt.barh([g[1] for g in top][::-1], [g[0] for g in top][::-1],
         color="#E69F00", edgecolor="white")
for i, g in enumerate(top[::-1]):
    plt.text(g[0] + 1, i, f"{g[0]:.1f} mi", va="center", fontsize=9)
plt.xlabel("trail miles between resupply towns")
plt.title("Longest food carries: town-to-town stretches", fontsize=13, weight="bold")
save("55_carries.png")

print("done → graphics/", flush=True)
