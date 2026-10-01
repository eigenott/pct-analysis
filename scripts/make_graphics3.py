"""Standalone graphics part 3 (31-41): ten viral candidates + per-mile speed map.

All landscape, aspect no more extreme than 16:9, professional titles.
Output: graphics/*.png (gitignored, local only).

Usage: uv run python scripts/make_graphics3.py
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
BOUNDS = [0, 702, 1092, 1694, 2146, 3000]
DEEP = "#333333"
RED = "#C0392B"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False,
})

daily = pl.read_parquet(BASE / "data" / "daily_report.parquet").sort("date")
d = daily.to_dicts()
sec_of = {r["date"]: r["section"] for r in d}
dates = [r["date"] for r in d]
mval = np.array([r["net_mi"] for r in d])


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()
    print(" ", name, flush=True)


print("rendering graphics 31-41...", flush=True)

# 31 — lollipop top 15 days -----------------------------------------------------------
top = sorted(d, key=lambda r: -r["net_mi"])[:15]
plt.figure(figsize=(10, 6))
y = np.arange(len(top))
plt.hlines(y, 0, [r["net_mi"] for r in top], color="#BBBBF2", lw=3)
plt.scatter([r["net_mi"] for r in top], y, s=81,
            c=[SECC[SECTIONS.index(r["section"])] for r in top],
            edgecolors="white", zorder=3)
plt.yticks(y, [f"{r['date'][5:]} ({r['section'][:2]})" for r in top], fontsize=9)
plt.xlabel("trail miles")
plt.title("The 15 biggest days", fontsize=13, weight="bold")
save("31_lollipops.png")

# 32 — diverging vs personal average -------------------------------------------------------
avg = mval.mean()
dev = mval - avg
cols = [DEEP if v >= 0 else RED for v in dev]
plt.figure(figsize=(11, 5))
plt.bar(range(len(d)), dev, color=cols, width=1.0)
plt.axhline(0, color="black", lw=1)
plt.xticks(range(0, len(d), 10), [dates[i][5:] for i in range(0, len(d), 10)], rotation=45)
plt.ylabel(f"miles above/below my {avg:.1f} mi average")
plt.title("Above or below average, every day", fontsize=13, weight="bold")
save("32_diverging.png")

# 33 — sleep debt staircase ---------------------------------------------------------------------
slp = {r["date"]: r["sleep_hrs"] for r in
       pl.read_parquet(BASE / "data" / "sleep.parquet").to_dicts()}
debt = np.cumsum([8 - slp.get(x, 8) for x in dates])
plt.figure(figsize=(11, 5))
plt.fill_between(range(len(d)), debt, alpha=0.3, color="#7A7AD6")
plt.plot(debt, color=DEEP, lw=2)
plt.xticks(range(0, len(d), 10), [dates[i][5:] for i in range(0, len(d), 10)], rotation=45)
plt.ylabel("cumulative hours under 8 h/night")
plt.title("Sleep debt: the bill for 99 nights", fontsize=13, weight="bold")
save("33_sleep_debt.png")

# 34 — HRV vs pace quadrants -------------------------------------------------------------------------
hv = {r["date"]: r["hrv"] for r in
      pl.read_parquet(BASE / "data" / "hrv.parquet").to_dicts()}
x = np.array([hv.get(r["date"], np.nan) for r in d])
y = np.array([r["pace_mph"] for r in d])
m = ~np.isnan(x)
plt.figure(figsize=(9, 6))
plt.scatter(x[m], y[m], c=[SECC[SECTIONS.index(r["section"])] for r, k in zip(d, m) if k],
            s=60, alpha=0.85, edgecolors="white")
plt.axvline(np.median(x[m]), color="black", alpha=0.3, ls="--")
plt.axhline(np.median(y[m]), color="black", alpha=0.3, ls="--")
plt.xlabel("morning HRV (ms)")
plt.ylabel("pace that day (mph)")
plt.title("Recovered and fast? HRV vs pace by section", fontsize=13, weight="bold")
save("34_hrv_pace.png")

# 35 — start time vs miles, section poster ----------------------------------------------------------------
st = np.array([(int(r["start_local"][:2]) * 60 + int(r["start_local"][3:])) for r in d])
plt.figure(figsize=(10, 6))
for s, col in zip(SECTIONS, SECC):
    xx = np.array([t for t, r in zip(st, d) if r["section"] == s])
    yy = np.array([r["net_mi"] for r in d if r["section"] == s])
    plt.scatter(xx, yy, color=col, s=70, alpha=0.85, edgecolors="white", label=s)
ok = mval > 0
b, a = np.polyfit(st[ok], mval[ok], 1)
xs = np.linspace(st.min(), st.max(), 50)
plt.plot(xs, a + b * xs, color="black", lw=2, ls="--")
plt.xlabel("start time (minutes after midnight PT)")
plt.ylabel("trail miles")
plt.title("Do early starts make big days?", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("35_early_bird_sections.png")

# 36 — elevation × speed density --------------------------------------------------------------------------------
seg = pl.read_parquet(BASE / "data" / "segments.parquet").filter(
    pl.col("kept_h5") & (pl.col("dt_s") >= 8) & (pl.col("dt_s") <= 300))
plt.figure(figsize=(10, 6))
plt.hist2d(seg["dalt_ft"] / (seg["dist_mi"] * 5280) * 100, seg["mph"],
           bins=[60, 60], range=[[-30, 30], [0, 8]], cmap="Purples")
plt.colorbar(label="segments")
plt.xlabel("grade (%)")
plt.ylabel("mph")
plt.title("Where the hiking actually happens: grade × speed density",
          fontsize=13, weight="bold")
save("36_density.png")

# 37 — weekly boxes -------------------------------------------------------------------------------
wk = [((dt.date.fromisoformat(x) - dt.date(2026, 4, 25)).days // 7) + 1 for x in dates]
weeks = sorted(set(wk))
plt.figure(figsize=(12, 5))
plt.boxplot([[r["net_mi"] for r in d
               if ((dt.date.fromisoformat(r["date"]) - dt.date(2026, 4, 25)).days // 7) + 1 == w]
              for w in weeks],
             tick_labels=[f"W{w}" for w in weeks], patch_artist=True,
             boxprops={"facecolor": "#BBBBF2"}, medianprops={"color": DEEP})
plt.ylabel("daily trail miles")
plt.title("Week-by-week distributions: fitness arriving around week 8?",
          fontsize=13, weight="bold")
save("37_weeks.png")

# 38 — ascent vs descent -----------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter([r["descent_ft"] for r in d], [r["ascent_ft"] for r in d],
            c=[SECC[SECTIONS.index(r["section"])] for r in d],
            s=65, alpha=0.85, edgecolors="white")
lim = [0, max(r["ascent_ft"] for r in d) * 1.05]
plt.plot(lim, lim, color="black", alpha=0.4, ls="--", label="what goes up...")
plt.xlabel("descent (ft)")
plt.ylabel("ascent (ft)")
plt.title("What goes up must come down (mostly)", fontsize=13, weight="bold")
save("38_up_down.png")

# 39 — headlamp miles by week ------------------------------------------------------------------------------
hh = pl.read_parquet(BASE / "data" / "hourly.parquet")
wkday = {x: ((dt.date.fromisoformat(x) - dt.date(2026, 4, 25)).days // 7) + 1 for x in dates}
early = hh.filter(pl.col("hour_local") < 6).group_by("date").agg(pl.col("mi").sum().alias("m"))
late = hh.filter(pl.col("hour_local") >= 20).group_by("date").agg(pl.col("mi").sum().alias("m"))
em = {r["date"]: r["m"] for r in early.to_dicts()}
lm = {r["date"]: r["m"] for r in late.to_dicts()}
W = sorted(set(wkday.values()))
e = [sum(em.get(x, 0) for x in dates if wkday[x] == w) for w in W]
l = [sum(lm.get(x, 0) for x in dates if wkday[x] == w) for w in W]
x = np.arange(len(W))
plt.figure(figsize=(11, 5))
plt.bar(x, e, label="before 6am", color="#3D3D8F")
plt.bar(x, l, bottom=e, label="after 8pm", color="#E69F00")
plt.xticks(x, [f"W{w}" for w in W])
plt.ylabel("miles in the dark")
plt.title("Headlamp miles by week", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("39_headlamp.png")

# 40 — do zeros work? day-after-zero vs normal ----------------------------------------------------------------------
have = set(dates)
d0, d1 = dt.date(2026, 4, 25), dt.date(2026, 8, 11)
zeros = [str(d0 + dt.timedelta(days=i)) for i in range((d1 - d0).days + 1)
         if str(d0 + dt.timedelta(days=i)) not in have]
after = set()
for z in zeros:
    nxt = str(dt.date.fromisoformat(z) + dt.timedelta(days=1))
    if nxt in have:
        after.add(nxt)
a = [r["net_mi"] for r in d if r["date"] in after]
b = [r["net_mi"] for r in d if r["date"] not in after]
plt.figure(figsize=(8, 5))
plt.bar(["day after zero\n(n=%d)" % len(a), "normal day\n(n=%d)" % len(b)],
        [np.mean(a), np.mean(b)], color=["#E69F00", "#7A7AD6"], width=0.55)
plt.ylabel("mean trail miles")
plt.title("Do rest days pay off the next morning?", fontsize=13, weight="bold")
for i, v in enumerate([np.mean(a), np.mean(b)]):
    plt.text(i, v + 0.4, f"{v:.1f} mi", ha="center", fontsize=11)
save("40_zeros_work.png")

# 41 — speed map: centerline colored by median pace per trail mile ---------------------------------------------------------
trk = (pl.read_parquet(BASE / "data" / "tracks.parquet")
       .filter(pl.col("lat").is_not_null())
       .sort(["date", "source_file", "point_index"]))
pr = (pl.read_parquet(BASE / "data" / "track_route.parquet")
      .sort(["date", "source_file", "point_index"]))
rm_all = pr["route_mi"].to_numpy()
la_all = trk["lat"].to_numpy()
lo_all = trk["lon"].to_numpy()
sg = (pl.read_parquet(BASE / "data" / "segments.parquet")
      .sort(["date", "source_file", "seg_order"]))
sizes = trk.group_by(["date", "source_file"], maintain_order=True).len()["len"].to_numpy()
row0 = np.concatenate([[0], np.cumsum(sizes)[:-1]])
ok = np.concatenate([np.arange(s, s + n - 1) for s, n in zip(row0, sizes)]).astype(np.int64)
sg = (pl.read_parquet(BASE / "data" / "segments.parquet")
      .sort(["date", "source_file", "seg_order"]))
fm = ((sg["dt_s"] >= 8) & (sg["dt_s"] <= 300) & sg["kept_h5"]).to_numpy()
okf = ok[fm]
spf = sg.filter(fm)["mph"].to_numpy()
smi = np.floor(rm_all[okf]).astype(int)
mibins = np.arange(0, 2660)
med = np.full(2660, np.nan)
for bb in range(2660):
    sel = smi == bb
    if sel.sum() > 3:
        med[bb] = np.median(spf[sel])
cl = pl.read_parquet(BASE / "data" / "route_centerline.parquet")
cla, clo, cmi = cl["lat"].to_numpy(), cl["lon"].to_numpy(), cl["route_mi"].to_numpy()
step = 8
fig, axes = plt.subplots(1, 5, figsize=(13, 6))
for ax, s, (lo_b, hi_b) in zip(axes, SECTIONS, zip(BOUNDS, BOUNDS[1:])):
    m = (cmi >= lo_b) & (cmi < hi_b)
    sc = ax.scatter(clo[m][::step], cla[m][::step], c=med[np.floor(cmi[m][::step]).astype(int)],
                    cmap="RdYlGn", vmin=1.5, vmax=4.5, s=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(s, fontsize=11, weight="bold")
fig.colorbar(sc, ax=list(axes), shrink=0.8, label="median pace (mph) per trail mile")
fig.suptitle("How fast is every mile of the PCT?", fontsize=14, weight="bold")
save("41_speed_map.png")

print("done → graphics/", flush=True)
