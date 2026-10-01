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


def _light(hexcol, amt=0.55):
    import matplotlib.colors as mcolors
    c = np.array(mcolors.to_rgb(hexcol))
    return tuple(float(v) for v in 1 - (1 - c) * (1 - amt))


SEC_LIGHT = [_light(c) for c in SECC]  # washed-out twin of each section hue
BOUNDS = [0, 702, 1092, 1694, 2146, 3000]
DEEP = "#333333"
RED = "#C0392B"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
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
plt.hlines(y, 0, [r["net_mi"] for r in top], color="#CCCCCC", lw=3)
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
cols = ["#666666" if v >= 0 else RED for v in dev]
plt.figure(figsize=(11, 5))
plt.bar(range(len(d)), dev, color=cols, width=1.0)
plt.axhline(0, color="black", lw=1)
plt.xticks(range(0, len(d), 10), [dates[i][5:] for i in range(0, len(d), 10)], rotation=45)
plt.ylabel(f"miles above/below my {avg:.1f} mi average")
plt.title("Above or below average, every day", fontsize=13, weight="bold")
save("32_diverging.png")

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
plt.xlabel("start time (PT)")
plt.xticks(range(240, 1141, 120), [f"{h}:00" for h in range(4, 20, 2)])
plt.ylabel("trail miles")
plt.title("Do early starts make big days?", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("35_early_bird_sections.png")

# 36 — typical speed by grade band ------------------------------------------------------------------
seg = pl.read_parquet(BASE / "data" / "segments.parquet").filter(
    pl.col("kept_h5") & (pl.col("dist_mi") * 5280 > 100)
    & (pl.col("dt_s") >= 8) & (pl.col("dt_s") <= 300)
    & (pl.col("grade").abs() < 0.3))
gb = (seg.with_columns(((pl.col("grade") * 100 // 4 * 4)).alias("band"))
      .group_by("band").agg(pl.col("mph").median().alias("m"),
                             pl.len().alias("n"))
      .filter(pl.col("n") > 500).sort("band"))
plt.figure(figsize=(10, 5.5))
plt.bar([f"{b:.0f}%" for b in gb["band"]], gb["m"], color="#7A7AD6", edgecolor="white")
plt.xlabel("grade (4% bands)")
plt.ylabel("median mph")
plt.title("Typical speed by steepness", fontsize=13, weight="bold")
save("36_density.png")

# 37 — weekly boxes -------------------------------------------------------------------------------
wk = [((dt.date.fromisoformat(x) - dt.date(2026, 4, 25)).days // 7) + 1 for x in dates]
weeks = sorted(set(wk))
maj = []
for w in weeks:
    secs = [r["section"] for r in d
            if ((dt.date.fromisoformat(r["date"]) - dt.date(2026, 4, 25)).days // 7) + 1 == w]
    maj.append(max(set(secs), key=secs.count))
plt.figure(figsize=(12, 5))
bp = plt.boxplot([[r["net_mi"] for r in d
                   if ((dt.date.fromisoformat(r["date"]) - dt.date(2026, 4, 25)).days // 7) + 1 == w]
                  for w in weeks],
                 tick_labels=[f"W{w}" for w in weeks], patch_artist=True,
                 boxprops={"facecolor": "white"}, medianprops={"color": "black"})
for p, s in zip(bp["boxes"], maj):
    p.set_facecolor(SECC[SECTIONS.index(s)])
    p.set_alpha(0.85)
plt.ylabel("daily trail miles")
plt.title("Mileage by week", fontsize=13, weight="bold")
save("37_weeks.png")

# 38 — total climb vs descent by section ---------------------------------------------------------------
plt.figure(figsize=(10, 5.5))
asc = [sum(r["ascent_ft"] for r in d if r["section"] == s) / 5280 for s in SECTIONS]
des = [sum(r["descent_ft"] for r in d if r["section"] == s) / 5280 for s in SECTIONS]
x = np.arange(5)
plt.bar(x - 0.2, asc, width=0.4, color=SECC, label="climbed")
plt.bar(x + 0.2, des, width=0.4, color="#CCCCCC", label="descended")
plt.xticks(x, SECTIONS)
plt.ylabel("miles of vertical (mi)")
plt.title("Climbed vs descended by section", fontsize=13, weight="bold")
plt.legend(frameon=False)
for i, (a, b) in enumerate(zip(asc, des)):
    plt.text(i - 0.2, a + 0.5, f"{a:.0f}", ha="center", fontsize=9)
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
plt.bar(x, e, label="before 6am", color="#444444")
plt.bar(x, l, bottom=e, label="after 8pm", color="#BBBBF2")
plt.xticks(x, [f"W{w}" for w in W])
plt.ylabel("miles in the dark")
plt.title("Headlamp miles by week", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("39_headlamp.png")

# 40 — do rest days pay off? overall + by section -----------------------------------------------------
have = set(dates)
d0, d1 = dt.date(2026, 4, 25), dt.date(2026, 8, 11)
zeros = [str(d0 + dt.timedelta(days=i)) for i in range((d1 - d0).days + 1)
         if str(d0 + dt.timedelta(days=i)) not in have]
after_zero, after_nero = set(), set()
for z in zeros:
    nxt = str(dt.date.fromisoformat(z) + dt.timedelta(days=1))
    if nxt in have:
        after_zero.add(nxt)
for r in [x for x in d if x["is_nero"]]:
    nxt = str(dt.date.fromisoformat(r["date"]) + dt.timedelta(days=1))
    if nxt in have:
        after_nero.add(nxt)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
groups = [("normal day", [r for r in d if r["date"] not in after_zero | after_nero]),
          ("after zero", [r for r in d if r["date"] in after_zero]),
          ("after nero", [r for r in d if r["date"] in after_nero])]
ax1.bar([g[0] + f"\n(n={len(g[1])})" for g in groups],
        [np.mean([r["net_mi"] for r in g[1]]) for g in groups],
        color=["#E69F00", "#999999", "#CCCCCC"], width=0.6)
ax1.set_ylabel("mean trail miles")
ax1.set_title("Daily mileage following a zero/nero", fontsize=12, weight="bold")
x = np.arange(5)
w = 0.35
norm = [np.mean([r["net_mi"] for r in d if r["section"] == s
                 and r["date"] not in after_zero | after_nero]) for s in SECTIONS]
rest = [np.mean([r["net_mi"] for r in d if r["section"] == s
                 and r["date"] in after_zero | after_nero]) for s in SECTIONS]
ax2.bar(x - w / 2, norm, width=w, label="normal", color=SECC)
ax2.bar(x + w / 2, [0 if np.isnan(v) else v for v in rest], width=w,
        label="after zero/nero", color="#999999")
ax2.set_xticks(x, SECTIONS)
ax2.set_title("Daily mileage following a zero/nero by section", fontsize=12, weight="bold")
ax2.legend(frameon=False)
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
    ax.scatter(clo[m][::step], cla[m][::step], c=med[np.floor(cmi[m][::step]).astype(int)],
               cmap="RdYlGn", vmin=1.5, vmax=4.5, s=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(s, fontsize=11, weight="bold")
fig.suptitle("How fast is every mile of the PCT? (green = fast, red = slow)",
               fontsize=14, weight="bold")
save("41_speed_map.png")

# 59 — camp map: one dot per night, sized by that day's miles ---------------------------------------------
mid_mi = ((daily.select(["date", "net_mi", "section", "is_full", "route_min_mi", "route_max_mi"])
           .with_columns(((pl.col("route_min_mi") + pl.col("route_max_mi")) / 2).alias("mid")))
          .to_dicts())
cl = pl.read_parquet(BASE / "data" / "route_centerline.parquet")
cmi = cl["route_mi"].to_numpy()
cla, clo = cl["lat"].to_numpy(), cl["lon"].to_numpy()
fig, axes = plt.subplots(1, 5, figsize=(13, 5.6),
                           gridspec_kw={"width_ratios": [
                               # one shared scale: cell aspect matches data aspect
                               # so aspect="equal" never letterboxes any panel
                               (np.ptp(clo[(cmi >= lo_b) & (cmi < hi_b)])
                                / np.ptp(cla[(cmi >= lo_b) & (cmi < hi_b)]))
                               for lo_b, hi_b in zip(BOUNDS, BOUNDS[1:])]})
for ax, s, col, (lo_b, hi_b) in zip(axes, SECTIONS, SECC, zip(BOUNDS, BOUNDS[1:])):
    m = (cmi >= lo_b) & (cmi < hi_b)
    ax.scatter(clo[m][::15], cla[m][::15], c="#DDDDDD", s=3, zorder=1)
    dd = [r for r in mid_mi if r["section"] == s]
    at = [np.searchsorted(cmi, min(max(r["mid"], lo_b), hi_b - 0.01)) for r in dd]
    cols = [(col if r["is_full"] else SEC_LIGHT[SECTIONS.index(s)]) for r in dd]
    ax.scatter(clo[at], cla[at],
               s=[r["net_mi"] * 5 for r in dd], color=cols, edgecolors="white",
               linewidths=0.8, alpha=0.9, zorder=3)
    pad = 0.06
    ax.set_xlim(clo[m].min() - pad, clo[m].max() + pad)
    ax.set_ylim(cla[m].min() - pad, cla[m].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(s, fontsize=11, weight="bold")
for ms in (10, 20, 30, 40):
    axes[-1].scatter([], [], s=ms * 5, color="gray", edgecolors="white",
                     label=f"{ms} mi")
axes[-1].legend(frameon=False, fontsize=9, loc="lower right", title="daily miles")
fig.suptitle("Daily mileage", fontsize=14, weight="bold")
save("59_camp_map.png")

# 60 — resupply towns on the trail -------------------------------------------------------------------
tw60 = pl.read_csv(BASE / "reference" / "towns.csv").sort("mile").to_dicts()
cl = pl.read_parquet(BASE / "data" / "route_centerline.parquet")
cmi = cl["route_mi"].to_numpy()
cla, clo = cl["lat"].to_numpy(), cl["lon"].to_numpy()
tw_at = []
for t in tw60:
    i = int(np.searchsorted(cmi, min(max(t["mile"], 0), cmi[-1])))
    tw_at.append((t["name"], clo[i], cla[i], t["mile"]))
fig, axes = plt.subplots(1, 5, figsize=(13, 5.6),
                         gridspec_kw={"width_ratios": [
                             (np.ptp(clo[(cmi >= lo_b) & (cmi < hi_b)])
                              / np.ptp(cla[(cmi >= lo_b) & (cmi < hi_b)]))
                             for lo_b, hi_b in zip(BOUNDS, BOUNDS[1:])]})
for ax, s, (lo_b, hi_b) in zip(axes, SECTIONS, zip(BOUNDS, BOUNDS[1:])):
    m = (cmi >= lo_b) & (cmi < hi_b)
    ax.scatter(clo[m][::15], cla[m][::15], c="#DDDDDD", s=3, zorder=1)
    # (dxpt, dypt, ha, va) overrides for crowded labels; default: right/center
    tweaks = {
        "Cajon Junction": (-4, 0, "right", "center"),
        "Wrightwood": (5, 7, "left", "bottom"),
        "Government Camp": (-4, -9, "right", "top"),
        "Stehekin": (-4, -9, "right", "top"),
        "Stevens Pass": (-4, -9, "right", "top"),
    }
    for name, lo, la, mile in [t for t in tw_at if lo_b <= t[3] < hi_b]:
        ax.scatter([lo], [la], c="black", s=28, zorder=3)
        dx, dy, ha, va = tweaks.get(name, (4, 0, "left", "center"))
        ax.annotate(name, (lo, la), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, va=va, ha=ha, zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.65, "pad": 0.5, "boxstyle": "round,pad=0.2"})
    ax.set_xlim(clo[m].min() - 0.06, clo[m].max() + 0.35)
    ax.set_ylim(cla[m].min() - 0.06, cla[m].max() + 0.06)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(s, fontsize=11, weight="bold")
fig.suptitle("Resupply towns along the trail", fontsize=14, weight="bold")
save("60_towns_map.png")

# 61 — longest food carry in each section ------------------------------------------------------------------
all_gaps = sorted(
    [(tw60[i + 1]["mile"] - tw60[i]["mile"], tw60[i]["name"], tw60[i + 1]["name"])
     for i in range(len(tw60) - 1)
     if tw60[i + 1]["mile"] > tw60[i]["mile"]], reverse=True)


def _sec(mi):
    for b, s in zip([702, 1092, 1694, 2146, 1e9], SECTIONS):
        if mi < b:
            return s
    return SECTIONS[-1]


best = {}
tw_mile = {t["name"]: t["mile"] for t in tw60}
for gap, a, b in all_gaps:
    s = _sec((tw_mile[a] + tw_mile[b]) / 2)
    if s not in best:
        best[s] = (gap, a, b)
fig, axes = plt.subplots(1, 5, figsize=(13, 5.6),
                         gridspec_kw={"width_ratios": [
                             (np.ptp(clo[(cmi >= lo_b) & (cmi < hi_b)])
                              / np.ptp(cla[(cmi >= lo_b) & (cmi < hi_b)]))
                             for lo_b, hi_b in zip(BOUNDS, BOUNDS[1:])]})
for ax, s, col, (lo_b, hi_b) in zip(axes, SECTIONS, SECC, zip(BOUNDS, BOUNDS[1:])):
    m = (cmi >= lo_b) & (cmi < hi_b)
    ax.scatter(clo[m][::15], cla[m][::15], c="#DDDDDD", s=3, zorder=1)
    gap, a, b = best[s]
    ma, mb = tw_mile[a], tw_mile[b]
    seg = (cmi >= ma) & (cmi <= mb)
    ax.scatter(clo[seg][::6], cla[seg][::6], c=col, s=8, zorder=3)
    j = int(np.searchsorted(cmi, (ma + mb) / 2))
    ax.text(clo[j], cla[j], f"  {gap:.0f} mi", fontsize=8, weight="bold",
            color=col, va="center", zorder=4)
    for mm, nm in ((ma, a), (mb, b)):
        k = int(np.searchsorted(cmi, mm))
        ax.scatter([clo[k]], [cla[k]], c="black", s=28, zorder=4)
        ax.annotate(nm, (clo[k], cla[k]), xytext=(5, 0), textcoords="offset points",
                    fontsize=6.5, va="center", zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.7, "pad": 0.5, "boxstyle": "round,pad=0.2"})
    ax.set_xlim(clo[m].min() - 0.06, clo[m].max() + 0.06)
    ax.set_ylim(cla[m].min() - 0.06, cla[m].max() + 0.06)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(s, fontsize=11, weight="bold")
fig.suptitle("Longest food carry in each section", fontsize=14, weight="bold")
save("61_carries_map.png")

# 63 — town stays from recording gaps ------------------------------------------------------
# A gap >6h between consecutive (H5-clean) fixes within 3 trail-miles of a town
# counts as time in that town. Backcountry camp gaps fall nowhere near towns.
import sys as _sys
_sys.path.insert(0, str(BASE / "scripts"))
from flag_jumps import segments as _segments

t63 = (pl.read_parquet(BASE / "data" / "tracks.parquet")
       .filter(pl.col("lat").is_not_null())
       .sort(["date", "source_file", "point_index"]))
ok63, dist63, dts63, spd63, g63, _p63, _f63 = _segments(t63)
v63 = ~np.isnan(spd63) & (dts63 > 0)
h563 = np.zeros(len(ok63), bool)
h563[v63] = ((spd63[v63] * 2.23694 > 26.8) & (dist63[v63] / 1609.344 > 656 / 5280))
bad63 = set(ok63[h563].tolist()) | set((ok63[h563] + 1).tolist())
pr63 = (pl.read_parquet(BASE / "data" / "track_route.parquet")
        .sort(["date", "source_file", "point_index"]))
rm63 = pr63["route_mi"].to_numpy()
ts63 = np.array([x.timestamp() for x in pr63["timestamp_utc"].to_list()])
good63 = np.array([i not in bad63 for i in range(len(pr63))])
ord63 = np.argsort(ts63)
ts63, rm63, good63 = ts63[ord63], rm63[ord63], good63[ord63]
gap_h = np.diff(ts63) / 3600
is_gap = (gap_h > 6) & good63[:-1] & good63[1:]
towns63 = pl.read_csv(BASE / "reference" / "towns.csv").sort("mile").to_dicts()
tm63 = np.array([t["mile"] for t in towns63])
stay = {t["name"]: 0.0 for t in towns63}
for i in np.where(is_gap)[0]:
    j = int(np.argmin(abs(tm63 - rm63[i])))
    if abs(tm63[j] - rm63[i]) <= 3:
        stay[towns63[j]["name"]] += float(gap_h[i])
print(f"   town-stay gaps: {is_gap.sum()}, attributed: "
      f"{sum(1 for v in stay.values() if v > 0)} towns", flush=True)


def _secc(mi):
    for b, s in zip([702, 1092, 1694, 2146, 1e9], SECTIONS):
        if mi < b:
            return SECC[SECTIONS.index(s)]
    return SECC[-1]


fig, axes = plt.subplots(1, 5, figsize=(13, 5.6),
                         gridspec_kw={"width_ratios": [
                             (np.ptp(clo[(cmi >= lo_b) & (cmi < hi_b)])
                              / np.ptp(cla[(cmi >= lo_b) & (cmi < hi_b)]))
                             for lo_b, hi_b in zip(BOUNDS, BOUNDS[1:])]})
TOWN_TWEAKS = {
    "Cajon Junction": (-4, 0, "right", "center"),
    "Wrightwood": (5, 7, "left", "bottom"),
    "Government Camp": (-4, -9, "right", "top"),
    "Stehekin": (-4, -9, "right", "top"),
    "Stevens Pass": (-4, -9, "right", "top"),
}
for ax, s, (lo_b, hi_b) in zip(axes, SECTIONS, zip(BOUNDS, BOUNDS[1:])):
    m = (cmi >= lo_b) & (cmi < hi_b)
    ax.scatter(clo[m][::15], cla[m][::15], c="#DDDDDD", s=3, zorder=1)
    for t in [x for x in towns63 if lo_b <= x["mile"] < hi_b]:
        i = int(np.searchsorted(cmi, t["mile"]))
        h = stay[t["name"]]
        col = _secc(t["mile"]) if h > 0.5 else "#CCCCCC"
        ax.scatter([clo[i]], [cla[i]], c=col, s=(22 + min(h, 80) * 3.2) if h > 0.5 else 16,
                   zorder=3, edgecolors="white", linewidths=0.8)
        if h > 0.5:
            lab = f"{t['name']} {h:.0f}h" if h >= 10 else t["name"]
            dx, dy, ha, va = TOWN_TWEAKS.get(t["name"], (4, 0, "left", "center"))
            ax.annotate(lab, (clo[i], cla[i]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6.5, va=va, ha=ha,
                        zorder=4,
                        bbox={"facecolor": "white", "edgecolor": "none",
                              "alpha": 0.7, "pad": 0.5, "boxstyle": "round,pad=0.2"})
    ax.set_xlim(clo[m].min() - 0.06, clo[m].max() + 0.06)
    ax.set_ylim(cla[m].min() - 0.06, cla[m].max() + 0.06)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(s, fontsize=11, weight="bold")
fig.suptitle("Time in town", fontsize=14, weight="bold")
save("63_town_stays.png")

print("done → graphics/", flush=True)
