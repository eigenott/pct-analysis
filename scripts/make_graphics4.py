"""Standalone graphics part 4 (42-51): ten more viral candidates.

All landscape, aspect no more extreme than 16:9, professional titles.
Output: graphics/*.png (gitignored, local only).

Usage: uv run python scripts/make_graphics4.py
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
DEEP = "#333333"
RED = "#C0392B"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

daily = pl.read_parquet(BASE / "data" / "daily_report.parquet").sort("date")
d = daily.to_dicts()
dates = [r["date"] for r in d]


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()
    print(" ", name, flush=True)


def mins(tstr):
    return int(tstr[:2]) * 60 + int(tstr[3:])


print("rendering graphics 42-51...", flush=True)

# 42 — a day in each section (median start → median end) ---------------------------------
plt.figure(figsize=(10, 5))
for i, (s, col) in enumerate(zip(SECTIONS, SECC)):
    sd = [r for r in d if r["section"] == s]
    s0 = np.median([mins(r["start_local"]) for r in sd])
    s1 = np.median([mins(r["end_local"]) for r in sd])
    plt.barh(i, s1 - s0, left=s0, height=0.55, color=col)
    plt.text(s1 + 8, i, f"{int(s0)//60:02d}:{int(s0)%60:02d}–{int(s1)//60:02d}:{int(s1)%60:02d}",
             va="center", fontsize=10)
plt.yticks(range(5), SECTIONS)
plt.xlabel("time of day (PT)")
plt.gca().xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)//60:02d}:{int(v)%60:02d}"))
plt.xlim(240, 1260)
plt.title("Median hiking day by section", fontsize=13, weight="bold")
save("42_day_anatomy.png")

# 43 — biggest week --------------------------------------------------------------------------------
roll = np.convolve([r["net_mi"] for r in d], np.ones(7), mode="valid")
i = int(np.argmax(roll))
wk_dates = f"{dates[i][5:]} → {dates[i+6][5:]}"
plt.figure(figsize=(11, 5))
plt.bar(range(len(roll)), roll, color="#BBBBF2", edgecolor="white")
plt.bar(i, roll[i], color="#D55E00", edgecolor="white")
plt.xticks(range(0, len(roll), 10), [dates[j][5:] for j in range(0, len(roll), 10)], rotation=45)
plt.xlabel("7-day window start")
plt.ylabel("miles in 7 days")
plt.title(f"Rolling 7-day totals — peak {roll[i]:.0f} mi ({wk_dates})",
          fontsize=13, weight="bold")
save("43_biggest_week.png")

# 44 — start-time histogram -------------------------------------------------------------------------------
st = np.array([mins(r["start_local"]) for r in d])
plt.figure(figsize=(10, 5))
plt.hist(st, bins=24, color="#7A7AD6", edgecolor="white")
plt.axvline(np.median(st), color="black", lw=2, ls="--",
            label=f"median {int(np.median(st))//60:02d}:{int(np.median(st))%60:02d}")
plt.xlabel("start time (PT)")
plt.gca().xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)//60:02d}:{int(v)%60:02d}"))
plt.ylabel("days")
plt.title("Morning start time", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("44_starts_hist.png")

# 45 — weekday medals ------------------------------------------------------------------------------------------
wd = [dt.date.fromisoformat(x).strftime("%a") for x in dates]
order_wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
med = [np.median([r["net_mi"] for r in d if dt.date.fromisoformat(r["date"]).strftime("%a") == w])
       for w in order_wd]
plt.figure(figsize=(10, 5))
plt.bar(order_wd, med, color="#009E73", edgecolor="white")
plt.ylabel("median trail miles")
plt.title(" trail miles by weekday — is Saturday for sending?", fontsize=13, weight="bold")
save("45_weekday.png")

# 46 — start-elevation bands vs that day's miles ----------------------------------------------------------------------
bands = [(0, 3000), (3000, 6000), (6000, 9000), (9000, 20000)]
labels, means, ns = [], [], []
for lo, hi in bands:
    v = [r["net_mi"] for r in d if lo <= r["camp_start_ft"] < hi]
    labels.append(f"{lo//1000}–{hi//1000}k ft" if hi < 20000 else "9k+ ft")
    means.append(np.mean(v))
    ns.append(len(v))
plt.figure(figsize=(10, 5))
plt.bar(labels, means, color="#56B4E9", edgecolor="white")
for x, m, n in zip(labels, means, ns):
    plt.text(x, m + 0.3, f"n={n}", ha="center", fontsize=10)
plt.ylabel("mean trail miles that day")
plt.xlabel("morning start elevation band")
plt.title("Does starting high change the day?", fontsize=13, weight="bold")
save("46_camp_bands.png")

# 47 — resting heart rate trend ------------------------------------------------------------------------------------------
hv = pl.read_parquet(BASE / "data" / "hrv.parquet").filter(
    (pl.col("date") >= "2026-04-25") & (pl.col("date") <= "2026-08-11")).sort("date")
plt.figure(figsize=(11, 5))
x = np.arange(len(hv))
plt.plot(x, hv["rest_hr"], color=RED, lw=2, marker="o", ms=3)
plt.xticks(range(0, len(hv), 10), [hv["date"][i][5:] for i in range(0, len(hv), 10)], rotation=45)
plt.ylabel("morning resting HR (bpm)")
plt.title("Resting heart rate across the season", fontsize=13, weight="bold")
save("47_rest_hr.png")

# 48 — bedtimes and wake times -----------------------------------------------------------------------------------------------
slp = pl.read_parquet(BASE / "data" / "sleep.parquet").filter(
    (pl.col("date") >= "2026-04-25") & (pl.col("date") <= "2026-08-11")).sort("date").to_dicts()
def hr(s):
    return int(s[11:13]) - 7  # GMT → PT
bed = [hr(r["sleep_start_gmt"]) % 24 for r in slp if r["sleep_start_gmt"]]
wak = [hr(r["sleep_end_gmt"]) % 24 for r in slp if r["sleep_end_gmt"]]
plt.figure(figsize=(11, 5))
plt.plot(bed, color="#3D3D8F", lw=2, label="bedtime")
plt.plot(wak, color="#E69F00", lw=2, label="wake")
plt.xticks(range(0, len(slp), 10), [slp[i]["date"][5:] for i in range(0, len(slp), 10)], rotation=45)
plt.ylabel("hour (PT)")
plt.title("Bedtimes and wake times across the season", fontsize=13, weight="bold")
plt.legend(frameon=False)
save("48_sleep_times.png")

# 49 — speed by elevation band -----------------------------------------------------------------------------------------------
t49 = pl.read_parquet(BASE / "data" / "tracks.parquet").filter(
    pl.col("lat").is_not_null()).sort(["date", "source_file", "point_index"])
alt49 = t49["altitude_m"].to_numpy() * 3.28084
sizes49 = t49.group_by(["date", "source_file"], maintain_order=True).len()["len"].to_numpy().astype(int)
row049 = np.concatenate([[0], np.cumsum(sizes49)[:-1]])
ok49 = np.concatenate([np.arange(s, s + n - 1) for s, n in zip(row049, sizes49)]).astype(np.int64)
sg = pl.read_parquet(BASE / "data" / "segments.parquet").sort(["date", "source_file", "seg_order"])
fm = ((sg["dt_s"] >= 8) & (sg["dt_s"] <= 300) & sg["kept_h5"]).to_numpy()
has_alt = ~np.isnan(alt49[ok49][fm])
eb = ((alt49[ok49][fm][has_alt] // 2000) * 2000).astype(int)
sp = sg.filter(fm)["mph"].to_numpy()[has_alt]
dmi = sg.filter(fm)["dist_mi"].to_numpy()[has_alt]
bands = sorted(set(eb.tolist()))
fig49, ax49 = plt.subplots(figsize=(10, 5.5))
ax49.bar([f"{b//1000}–{b//1000+2}k" for b in bands],
         [np.median(sp[eb == b]) for b in bands], color="#009E73", edgecolor="white")
ax49.set_xlabel("elevation band (ft)")
ax49.set_ylabel("median mph")
ax49.set_title("Speed by elevation band", fontsize=13, weight="bold")
ax2 = ax49.twinx()
share = np.array([dmi[eb == b].sum() for b in bands])
share = share / share.sum() * 100
ax2.plot([f"{b//1000}–{b//1000+2}k" for b in bands], share, color="#333333",
         marker="o", ms=7, lw=2)
ax2.set_ylabel("% of all trail miles in that band")
save("49_speed_elev.png")

# 50 — 109-day donut ----------------------------------------------------------------------------------------------------------------
n_full = sum(1 for r in d if r["is_full"])
n_nero = sum(1 for r in d if r["is_nero"])
n_zero = 109 - len(d)
plt.figure(figsize=(7, 7))
plt.pie([n_full, n_nero, n_zero], labels=[f"full days ({n_full})", f"neros ({n_nero})", f"zeros ({n_zero})"],
        colors=["#009E73", "#E69F00", "#CCCCCC"], wedgeprops={"width": 0.45},
        textprops={"fontsize": 12})
plt.title("109 days, sliced", fontsize=14, weight="bold")
save("50_donut.png")

# 51 — full-day streaks -----------------------------------------------------------------------------------------------------------------
streaks, cur = [], []
prev = None
for r in d:
    day = dt.date.fromisoformat(r["date"])
    if r["is_full"] and (prev is not None and (day - prev).days == 1 and cur):
        cur.append(r["date"])
    elif r["is_full"]:
        cur = [r["date"]]
    else:
        if len(cur) >= 3:
            streaks.append(cur)
        cur = []
    prev = day
if len(cur) >= 3:
    streaks.append(cur)
streaks.sort(key=len, reverse=True)
plt.figure(figsize=(11, 5))
plt.barh([f"{s[0][5:]} ({len(s)}d)" for s in streaks][::-1], [len(s) for s in streaks][::-1],
         color="#56B4E9", edgecolor="white")
plt.xlabel("consecutive full (≥20 mi) days")
plt.title("Longest full-day streaks", fontsize=13, weight="bold")
save("51_streaks.png")

# 62 — hiking window vs daylight: starts/ends against sunrise/sunset -----------------------------------
# Sun times via the NOAA approximation (±a few min) at each day's midpoint location.
cl62 = pl.read_parquet(BASE / "data" / "route_centerline.parquet")
cmi62 = cl62["route_mi"].to_numpy()


def sun_utc(day_ord, lat, lon, rise=True):
    lng_h = lon / 15
    t = day_ord + ((6 if rise else 18) - lng_h) / 24
    m = math.radians(0.9856 * t - 3.289)
    lam = (math.degrees(m) + 1.916 * math.sin(m) + 0.020 * math.sin(2 * m)
           + 282.634) % 360
    ra = math.degrees(math.atan(0.91764 * math.tan(math.radians(lam)))) % 360
    ra = (ra + (90 if 90 < lam < 270 else (-90 if lam < 90 or lam > 270 else 0))) % 360
    sin_dec = 0.39782 * math.sin(math.radians(lam))
    cos_dec = math.cos(math.asin(sin_dec))
    lat_r = math.radians(lat)
    cosh = (math.cos(math.radians(90.833)) - sin_dec * math.sin(lat_r)) / (cos_dec * math.cos(lat_r))
    cosh = min(max(cosh, -1), 1)
    h = (360 - math.degrees(math.acos(cosh)) if rise
         else math.degrees(math.acos(cosh))) / 15
    return (h + ra / 15 - 0.06571 * t - 6.622) % 24


import math  # noqa: E402  (kept local to this graphic's block)
sr, ss = [], []
for r in d:
    mid = (r["route_min_mi"] + r["route_max_mi"]) / 2
    j = int(np.searchsorted(cmi62, mid))
    lat, lon = float(cl62["lat"][j]), float(cl62["lon"][j])
    n = dt.date.fromisoformat(r["date"]).timetuple().tm_yday
    sr.append((sun_utc(n, lat, lon, True) - 7) % 24)
    ss.append((sun_utc(n, lat, lon, False) - 7) % 24)
st62 = np.array([(int(r["start_local"][:2]) + int(r["start_local"][3:]) / 60) for r in d])
en62 = np.array([(int(r["end_local"][:2]) + int(r["end_local"][3:]) / 60) for r in d])
x62 = np.arange(len(d))
plt.figure(figsize=(12, 5.5))
plt.fill_between(x62, sr, ss, color="#E69F00", alpha=0.18, label="daylight")
plt.plot(x62, sr, color="#E69F00", lw=1.5)
plt.plot(x62, ss, color="#E69F00", lw=1.5)
plt.scatter(x62, st62, color="#3D3D8F", s=36, zorder=3, label="started hiking")
plt.scatter(x62, en62, color="#C0392B", s=36, zorder=3, label="stopped hiking")
plt.xticks(range(0, len(d), 10), [dates[i][5:] for i in range(0, len(d), 10)], rotation=45)
plt.ylabel("hour (PT)")
plt.title("Hiking window vs daylight (sunrise/sunset at that day's location)",
          fontsize=13, weight="bold")
plt.legend(frameon=False)
save("62_daylight.png")

print("done → graphics/", flush=True)
