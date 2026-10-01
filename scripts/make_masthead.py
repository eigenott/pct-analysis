"""Project masthead: title stats + elevation profile + section cards.

Output: masthead.png in the repo root (tracked — shown in README).

Usage: uv run python scripts/make_masthead.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent.parent

SECTIONS = ["SoCal", "Sierra", "NorCal", "Oregon", "Washington"]
SECC = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00"]
BOUNDS = [0, 702, 1092, 1694, 2146, 2660]


def main() -> None:
    daily = pl.read_parquet(BASE / "data" / "daily_report.parquet").sort("date")
    cl = pl.read_parquet(BASE / "data" / "route_centerline.parquet")
    mi = cl["route_mi"].to_numpy()
    ele = np.where(np.isnan(cl["ele_m"].to_numpy()), np.nanmedian(cl["ele_m"].to_numpy()),
                   cl["ele_m"].to_numpy()) * 3.28084
    k = 400
    em = np.median(np.lib.stride_tricks.sliding_window_view(ele, k), axis=1)
    mm = np.convolve(mi, np.ones(k) / k, mode="valid")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.35, 1.6, 1.1], hspace=0.35, top=0.88,
                          bottom=0.08)
    fig.suptitle("MEXICO TO CANADA ON FOOT", fontsize=42, weight="bold", y=0.98)
    fig.text(0.5, 0.92, "Pacific Crest Trail 2026 · 2,443 trail miles · "
             "Campo → Rainy Pass · 99 hiking days · 10 zeros",
             ha="center", fontsize=16)

    ax = fig.add_subplot(gs[1])
    for i in range(5):
        m = (mm >= BOUNDS[i]) & (mm <= BOUNDS[i + 1])
        ax.fill_between(mm[m], em[m], alpha=0.6, color=SECC[i])
    ax.plot(mm, em, color="#333333", lw=1)
    ax.set_xlim(0, 2660)
    ax.set_xlabel("trail miles")
    ax.set_ylabel("elevation (ft)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ag = fig.add_subplot(gs[2])
    ag.axis("off")
    for i, s in enumerate(SECTIONS):
        sd = daily.filter(pl.col("section") == s)
        x0 = i / 5
        ag.text(x0 + 0.1, 0.85, s, fontsize=18, weight="bold", color=SECC[i],
                transform=ag.transAxes)
        ag.text(x0 + 0.1, 0.62,
                f"{sd['net_mi'].sum():.0f} mi · {sd.height} days",
                fontsize=14, transform=ag.transAxes)
        best = sd.sort("net_mi", descending=True).row(0, named=True)
        ag.text(x0 + 0.1, 0.38, f"best {best['net_mi']:.1f} mi ({best['date'][5:]})",
                fontsize=12, transform=ag.transAxes)
        ag.text(x0 + 0.1, 0.14,
                f"median pace {sd['pace_mph'].median():.2f} mph",
                fontsize=12, transform=ag.transAxes)
    fig.text(0.5, 0.02, "Forerunner 255 · UltraTrac · cleaned GPS · polars + marimo",
             ha="center", fontsize=11, style="italic")
    fig.savefig(BASE / "masthead.png", bbox_inches="tight")
    print("masthead.png written", flush=True)


if __name__ == "__main__":
    sys.exit(main())
