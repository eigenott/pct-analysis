import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np

    return alt, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Walking 2,443 miles: a data story from the PCT

    *A data scientist turned thru-hiker, learning polars + marimo on
    99 days of Garmin tracks (Apr 25 – Aug 11, 2026, Campo → Rainy Pass).
    Broad stuff first, weird stuff at the bottom — the weird stuff is the point.*
    """)
    return


@app.cell
def _(mo, pl):
    daily = pl.read_parquet("data/daily_report.parquet").sort("date")
    odd = pl.read_parquet("data/oddities.parquet").sort("date")
    markers = pl.read_parquet("data/route_markers.parquet").sort("mile")
    SEC_ORDER = ["SoCal", "Sierra", "NorCal", "Oregon", "Washington"]
    SEC_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00"]
    mo.md(
        f"**{daily['net_mi'].sum():.1f} trail miles** · "
        f"{daily.height} hiking days · "
        f"**{daily['net_mi'].max():.1f} mi** biggest day "
        f"({daily.sort('net_mi', descending=True).row(0, named=True)['date']}) · "
        f"**{int(daily['ascent_ft'].sum()):,} ft** climbed · "
        f"{daily['pace_mph'].mean():.1f} mph lifetime average"
    )
    return daily, markers, odd, SEC_ORDER, SEC_COLORS


@app.cell
def _(alt, daily, mo, pl):
    mo.md("## The hike in one chart — daily miles, corrected")
    bars = (
        alt.Chart(daily.with_columns(pl.col("date").str.to_datetime().alias("day")).to_pandas())
        .mark_bar(color="#BBBBF2")
        .encode(
            x=alt.X("day:T", title="date"),
            y=alt.Y("net_mi:Q", title="trail miles"),
            tooltip=["date", "net_mi", "ascent_ft", "pace_mph", "start_local", "end_local"],
        )
        .properties(width=800, height=280)
    )
    bars
    return


@app.cell
def _(alt, daily, mo, pl):
    mo.md("## Rhythm — when did the day start and end? (Pacific)")
    clock = daily.with_columns([
        pl.col("date").str.to_datetime().alias("day"),
        (pl.col("start_local").str.slice(0, 2).cast(pl.Int64) * 60
         + pl.col("start_local").str.slice(3, 2).cast(pl.Int64)).alias("start_min"),
        (pl.col("end_local").str.slice(0, 2).cast(pl.Int64) * 60
         + pl.col("end_local").str.slice(3, 2).cast(pl.Int64)).alias("end_min"),
    ])
    bookends = (
        alt.Chart(clock.to_pandas())
        .mark_circle(size=40, color="#7A7AD6")
        .encode(
            x=alt.X("day:T", title="date"),
            y=alt.Y("start_min:Q", title="time of day",
                    axis=alt.Axis(format="d"), scale=alt.Scale(domain=[240, 1320])),
            tooltip=["date", "start_local", "end_local", "net_mi"],
        )
        .properties(width=800, height=280)
    )
    bookends
    return (clock,)


@app.cell
def _(clock, mo):
    early = clock.sort("start_min").row(0, named=True)
    late = clock.sort("end_min", descending=True).row(0, named=True)
    mo.md(
        f"Earliest start: **{early['start_local']}** on {early['date']} "
        f"({early['net_mi']:.0f} mi that day). Latest finish: **{late['end_local']}** "
        f"on {late['date']}. Median start **{clock['start_local'].median()}**, "
        f"median finish **{clock['end_local'].median()}**."
    )
    return


@app.cell
def _(alt, daily, mo, pl):
    mo.md("## Trail legs — did I get faster? (pace + 7-day trend)")
    legs = daily.with_columns(
        pl.col("pace_mph").rolling_mean(7, center=True).alias("pace_7d"),
        pl.col("date").str.to_datetime().alias("day"),
    ).to_pandas()
    legs_base = alt.Chart(legs).encode(
        x=alt.X("day:T", title="date")
    )
    (legs_base.mark_circle(size=30, color="#BBBBF2").encode(
        y=alt.Y("pace_mph:Q", title="mph", scale=alt.Scale(zero=False)),
        tooltip=["date", "pace_mph", "net_mi", "ascent_ft"],
    ) + legs_base.mark_line(color="#7A7AD6", strokeWidth=3).encode(y="pace_7d:Q")).properties(width=800, height=280)
    return


@app.cell
def _(alt, daily, mo, pl):
    mo.md("## Engine room — average heart rate trend (fitness = same pace, lower pulse?)")
    hr = daily.with_columns(
        pl.col("avg_hr").rolling_mean(7, center=True).alias("hr_7d"),
        pl.col("date").str.to_datetime().alias("day"),
    ).to_pandas()
    hr_base = alt.Chart(hr).encode(
        x=alt.X("day:T", title="date")
    )
    (hr_base.mark_circle(size=30, color="#BBBBF2").encode(
        y=alt.Y("avg_hr:Q", title="avg bpm", scale=alt.Scale(zero=False)),
        tooltip=["date", "avg_hr", "pace_mph", "net_mi"],
    ) + hr_base.mark_line(color="#7A7AD6", strokeWidth=3).encode(y="hr_7d:Q")).properties(width=800, height=280)
    return


@app.cell
def _(alt, daily, mo, pl):
    mo.md("## Where I slept — camp elevation each night")
    camps = daily.select(["date", "camp_end_ft"]).with_columns(
        pl.col("date").str.to_datetime().alias("day")
    ).to_pandas()
    alt.Chart(camps).mark_area(color="#BBBBF2", line={"color": "#7A7AD6"}).encode(
        x=alt.X("day:T", title="date"),
        y=alt.Y("camp_end_ft:Q", title="camp elevation (ft)"),
        tooltip=["date", "camp_end_ft"],
    ).properties(width=800, height=240)
    return


@app.cell
def _(alt, mo, pl):
    mo.md("## Steepness vs speed — how much do climbs cost? (kept segments, >100 ft, fixes 8 s apart — shorter gaps are timestamp noise)")
    gs = (
        pl.scan_parquet("data/segments.parquet")
        .filter(pl.col("kept_h5") & (pl.col("dist_mi") * 5280 > 100)
                & (pl.col("dt_s") >= 8) & (pl.col("dt_s") <= 120))
        .filter(pl.col("grade").abs() < 0.4)
        .with_columns(((pl.col("grade") * 50).round() * 2).alias("grade_pct"))
        .group_by("grade_pct")
        .agg(pl.col("mph").median().alias("med_mph"), pl.len().alias("n"))
        .filter(pl.col("n") > 50)
        .collect().to_pandas()
    )
    alt.Chart(gs).mark_line(point=True, color="#7A7AD6").encode(
        x=alt.X("grade_pct:Q", title="grade (%)"),
        y=alt.Y("med_mph:Q", title="median mph", scale=alt.Scale(zero=False)),
        tooltip=["grade_pct", "med_mph", "n"],
    ).properties(width=800, height=260)
    return


@app.cell
def _(alt, mo, pl):
    mo.md("## Speed by time of day — siesta hours? (Pacific, kept segments, 4am–11pm)")
    hh = (
        pl.scan_parquet("data/segments.parquet")
        .filter(pl.col("kept_h5") & (pl.col("dt_s") <= 120)
                & pl.col("hour_local").is_between(4, 23))
        .group_by("hour_local")
        .agg(pl.col("mph").median().alias("med_mph"), pl.len().alias("n"))
        .collect().to_pandas()
    )
    alt.Chart(hh).mark_bar(color="#BBBBF2").encode(
        x=alt.X("hour_local:O", title="hour (PT)"),
        y=alt.Y("med_mph:Q", title="median mph", scale=alt.Scale(zero=False)),
        tooltip=["hour_local", "med_mph", "n"],
    ).properties(width=800, height=260)
    return


@app.cell
def _(alt, daily, mo, pl):
    mo.md("## Elevation envelope — highest/lowest fix each day + camp line")
    eb = (
        pl.scan_parquet("data/tracks.parquet")
        .filter(pl.col("altitude_m").is_not_null())
        .group_by("date")
        .agg((pl.col("altitude_m").min() * 3.28084).alias("lo_ft"),
             (pl.col("altitude_m").max() * 3.28084).alias("hi_ft"))
        .collect()
        .with_columns(pl.col("date").str.to_datetime().alias("day"))
        .to_pandas()
    )
    camp = daily.select(["date", "camp_end_ft"]).with_columns(
        pl.col("date").str.to_datetime().alias("day")).to_pandas()
    band_elev = alt.Chart(eb).mark_area(color="#BBBBF2", opacity=0.6).encode(
        x=alt.X("day:T", title="date"),
        y=alt.Y("lo_ft:Q", title="elevation (ft)"),
        y2="hi_ft:Q",
        tooltip=["date", "lo_ft", "hi_ft"],
    )
    (band_elev + alt.Chart(camp).mark_line(color="#7A7AD6").encode(x="day:T", y="camp_end_ft:Q")).properties(width=800, height=260)
    return


@app.cell
def _(mo, pl):
    mo.md("## Confirmed timeline — annotate the changepoints against this")
    from pathlib import Path
    if Path("annotations/events.csv").exists():  # private, local-only
        ann = pl.read_csv("annotations/events.csv")
        mo.ui.table(ann)
    else:
        mo.md("*Timeline annotations are private and not in git.*")
    return


@app.cell
def _(daily, mo):
    hi = daily.sort("camp_end_ft", descending=True).row(0, named=True)
    lo = daily.sort("camp_end_ft").row(0, named=True)
    big = daily.sort("ascent_ft", descending=True).row(0, named=True)
    mo.md(
        f"Highest sleep: **{hi['camp_end_ft']:,} ft** ({hi['date']}). "
        f"Lowest sleep: **{lo['camp_end_ft']:,} ft** ({lo['date']}). "
        f"Biggest climb: **{big['ascent_ft']:,} ft** on {big['date']}."
    )
    return


@app.cell
def _(daily, mo, pl):
    mo.md("""
    ## Sections — five hikes in one (SoCal <702, Sierra <1092, NorCal <1694, Oregon <2146, Washington to end)

    Days assigned by midpoint trail mile. Past the desert, <20 mi = nero.
    """)
    order = ["SoCal", "Sierra", "NorCal", "Oregon", "Washington"]
    sec = (
        daily.group_by("section")
        .agg(pl.len().alias("days"),
             pl.col("net_mi").sum().round(1).alias("miles"),
             pl.col("pace_mph").mean().round(2).alias("avg_pace"),
             pl.col("is_full").sum().alias("full_days"),
             pl.col("is_nero").sum().alias("neros"))
        .with_columns(pl.col("section").cast(pl.Enum(order)).alias("s"))
        .sort("s").drop("s")
    )
    mo.ui.table(sec)
    return


@app.cell
def _(daily, mo, pl):
    mo.md("## Best & worst day in each section (trail miles)")
    rows = []
    for s in ["SoCal", "Sierra", "NorCal", "Oregon", "Washington"]:
        sd = daily.filter(pl.col("section") == s)
        if sd.height == 0:
            continue
        b = sd.sort("net_mi", descending=True).row(0, named=True)
        w = sd.sort("net_mi").row(0, named=True)
        rows.append({"section": s,
                     "best": f"{b['date']} ({b['net_mi']:.1f} mi)",
                     "worst": f"{w['date']} ({w['net_mi']:.1f} mi)"})
    mo.ui.table(pl.DataFrame(rows))
    return


@app.cell
def _(SEC_COLORS, SEC_ORDER, alt, daily, mo, pl):
    mo.md("## Full days only (≥20 mi) — pace by section, zeros and neros excluded")
    fd = daily.filter(pl.col("is_full")).with_columns(
        pl.col("date").str.to_datetime().alias("day"))
    pace_sec = (
        fd.group_by("section")
        .agg(pl.col("pace_mph").median().round(2).alias("med_pace"),
             pl.len().alias("n"))
        .sort("med_pace", descending=True).to_pandas()
    )
    alt.Chart(pace_sec).mark_bar().encode(
        x=alt.X("section:N", title=None, sort=SEC_ORDER),
        y=alt.Y("med_pace:Q", title="median pace, full days (mph)",
                scale=alt.Scale(zero=False)),
        color=alt.Color("section:N", legend=None,
                        scale=alt.Scale(domain=SEC_ORDER, range=SEC_COLORS)),
        tooltip=["section", "med_pace", "n"],
    ).properties(width=800, height=240)
    return


@app.cell
def _(mo, pl):
    mo.md("""
    ## The big climbs — sustained 2,000 ft+ ascents (600 ft dip tolerance)

    Net gain is start-to-top; gross counts every roller inside. Your
    mile-350 monster: **5,190 ft net (12,608 gross)** over 24.6 mi,
    May 16–18, at 1.74 mph — the slowest pace of any top climb.
    """)
    cl = pl.read_parquet("data/climbs.parquet").select(
        ["rank", "start_mi", "end_mi", "gain_ft", "gross_up_ft",
         "length_mi", "ft_per_mi", "section", "hike_dates", "pace_mph"])
    mo.ui.table(cl)
    return


@app.cell
def _(alt, daily, mo, pl):
    mo.md("## Heat — median speed by hour and section (kept segments, 4am–11pm PT)")
    sec_speed = (
        pl.scan_parquet("data/segments.parquet")
        .filter(pl.col("kept_h5") & (pl.col("dt_s") <= 120)
                & pl.col("hour_local").is_between(4, 23))
        .collect()
        .join(daily.select(["date", "section"]), on="date", how="left")
        .group_by(["hour_local", "section"])
        .agg(pl.col("mph").median().alias("med_mph"))
        .to_pandas()
    )
    alt.Chart(sec_speed).mark_rect().encode(
        x=alt.X("hour_local:O", title="hour (PT)"),
        y=alt.Y("section:N", title=None,
                sort=["SoCal", "Sierra", "NorCal", "Oregon", "Washington"]),
        color=alt.Color("med_mph:Q", title="median mph",
                        scale=alt.Scale(scheme="purples")),
        tooltip=["hour_local", "section", "med_mph"],
    ).properties(width=800, height=200)
    return


@app.cell
def _(alt, mo, pl):
    mo.md("## When were the big hours? — miles per clock hour, all season")
    hm = (
        pl.read_parquet("data/hourly.parquet").with_columns(
            pl.col("date").str.to_datetime().alias("day"))
        .to_pandas()
    )
    alt.Chart(hm).mark_rect().encode(
        x=alt.X("hour_local:O", title="hour (PT)"),
        y=alt.Y("day:T", title="date"),
        color=alt.Color("mi:Q", title="miles",
                        scale=alt.Scale(scheme="purples")),
        tooltip=["date", "hour_local", "mi"],
    ).properties(width=800, height=500)
    return


@app.cell
def _(SEC_COLORS, SEC_ORDER, alt, daily, mo):
    mo.md("## 2D scatters — what actually drives a big day?")
    base = alt.Chart(daily.to_pandas()).mark_circle(size=60).encode(
        color=alt.Color("section:N", legend=None,
                        scale=alt.Scale(domain=SEC_ORDER,
                                        range=SEC_COLORS)),
        tooltip=["date", "section", "net_mi", "ascent_ft", "pace_mph",
                 "start_local", "avg_hr"],
    )
    one = base.encode(x="ascent_ft:Q", y="pace_mph:Q").properties(
        title="climb vs pace", height=240)
    two = base.encode(x="avg_hr:Q", y="pace_mph:Q").properties(
        title="effort vs pace", height=240)
    mo.hstack([one, two])
    return


@app.cell
def _(SEC_COLORS, daily, mo, pl):
    mo.md("## 3D — big-day anatomy: climb × sleep × miles (color = section)")
    import plotly.express as px
    slp3 = pl.read_parquet("data/sleep.parquet").select(["date", "sleep_hrs"])
    d3 = daily.join(slp3, on="date", how="left").to_pandas()
    px.scatter_3d(d3, x="ascent_ft", y="sleep_hrs", z="net_mi", color="section",
                  color_discrete_sequence=SEC_COLORS,
                  hover_name="date",
                  labels={"ascent_ft": "climb (ft)", "sleep_hrs": "sleep (h)",
                          "net_mi": "miles"},
                  height=520, width=800).update_traces(marker={"size": 5})
    return


@app.cell
def _(alt, mo, pl):
    mo.md("""
    ## Breaks — detected from stillness (watch left running, <0.7 mph for 10+ min)

    Pausing the watch hides breaks, so these are a lower bound — 74 days show
    no detected stop, mostly because the watch was paused, not because rest
    didn't happen.
    """)
    brk = pl.read_parquet("data/breaks.parquet")
    hist = (
        alt.Chart(brk.with_columns(
            pl.col("start_local").str.slice(0, 2).cast(pl.Int64).alias("hr"))
            .to_pandas())
        .mark_bar(color="#BBBBF2").encode(
            x=alt.X("hr:O", title="break start hour (PT)"),
            y=alt.Y("count():Q", title="breaks"),
            tooltip=["hr", "count()"],
        ).properties(width=800, height=220)
    )
    hist
    return (brk,)


@app.cell
def _(brk, mo):
    mo.md(f"""
    **{brk.height} breaks** on {brk['date'].n_unique()} days "
        f"(median **{brk['minutes'].median():.0f} min**). Longest: "
        f"**{brk['minutes'].max():.0f} min** on "
        f"{brk.sort('minutes', descending=True).row(0, named=True)['date']} "
        f"— watch probably ran overnight.
    """)
    return


@app.cell
def _(alt, mo, pl):
    mo.md("## Fastest hour — median miles per clock hour, full days only (4am–11pm)")
    hrly = (
        pl.read_parquet("data/hourly.parquet")
        .filter(pl.col("hour_local").is_between(4, 23))
        .join(pl.read_parquet("data/daily_report.parquet")
              .select(["date", "is_full"]), on="date", how="left")
        .filter(pl.col("is_full"))
        .group_by("hour_local")
        .agg(pl.col("mi").median().alias("med_mi"))
        .sort("hour_local").to_pandas()
    )
    alt.Chart(hrly).mark_bar(color="#7A7AD6").encode(
        x=alt.X("hour_local:O", title="hour (PT)"),
        y=alt.Y("med_mi:Q", title="median miles in that hour"),
        tooltip=["hour_local", "med_mi"],
    ).properties(width=800, height=240)
    return


@app.cell
def _(alt, mo, pl):
    mo.md("## Sleep — duration trend, then does it matter tomorrow?")
    slp = pl.read_parquet("data/sleep.parquet").with_columns(
        pl.col("date").str.to_datetime().alias("day")).to_pandas()
    trend = alt.Chart(slp).mark_line(point=True, color="#7A7AD6").encode(
        x=alt.X("day:T", title="date"),
        y=alt.Y("sleep_hrs:Q", title="sleep (h)", scale=alt.Scale(zero=False)),
        tooltip=["date", "sleep_hrs", "score_overall", "deep_hrs", "rem_hrs"],
    ).properties(width=800, height=240)
    trend
    return


@app.cell
def _(SEC_COLORS, SEC_ORDER, alt, daily, mo, pl):
    mo.md("## Sleep vs next-day miles — do big sleeps make big days?")
    js = daily.select(["date", "net_mi", "section"]).join(
        pl.read_parquet("data/sleep.parquet").select(["date", "sleep_hrs"]),
        on="date", how="left").to_pandas()
    (alt.Chart(js).mark_circle(size=60).encode(
        x=alt.X("sleep_hrs:Q", title="sleep that morning (h)"),
        y=alt.Y("net_mi:Q", title="trail miles that day"),
        color=alt.Color("section:N", legend=None,
                        scale=alt.Scale(domain=SEC_ORDER,
                                        range=SEC_COLORS)),
        tooltip=["date", "sleep_hrs", "net_mi"],
    ) + alt.Chart(js).transform_regression("sleep_hrs", "net_mi").mark_line(
        color="firebrick").encode(x="sleep_hrs:Q", y="net_mi:Q")
    ).properties(width=800, height=280)
    return


@app.cell
def _(alt, mo, pl):
    mo.md("## HRV mornings — recovery signal vs pace that day")
    hv = (
        pl.read_parquet("data/hrv.parquet").with_columns(
            pl.col("date").str.to_datetime().alias("day"))
        .join(pl.read_parquet("data/daily_report.parquet").select(["date", "pace_mph"]),
              on="date", how="left")
        .to_pandas()
    )
    band_hrv = alt.Chart(hv).mark_area(opacity=0.25, color="#BBBBF2").encode(
        x="day:T", y="hrv_lo:Q", y2="hrv_hi:Q")
    (band_hrv + alt.Chart(hv).mark_line(point=True, color="#7A7AD6").encode(
        x="day:T", y=alt.Y("hrv:Q", title="morning HRV (ms)", scale=alt.Scale(zero=False)),
        tooltip=["date", "hrv", "hrv_status", "pace_mph"],
    )).properties(width=800, height=260)
    return


@app.cell
def _(markers, mo, odd, pl):
    mo.md("""
    ## Weird I — backtracks (did I drop something?)

    Sustained southbound wiggles >0.25 mi with time to be real footsteps
    (instant flips are hairpin projection artifacts — labeled, kept for fun).
    *at_mi* joined to the nearest half-mile marker section for location.
    """)
    bt = (
        odd.filter(pl.col("type") == "backtrack")
        .sort("at_route_mi")
        .join_asof(markers.select(["mile", "section"]).rename({"mile": "at_route_mi"}),
                   left_on="at_route_mi", right_on="at_route_mi", strategy="nearest")
        .select(["date", "start_local", "end_local", "mi", "extra_mi",
                 "at_route_mi", "section", "detail"])
        .sort("mi", descending=True)
    )
    mo.ui.table(bt)
    return


@app.cell
def _(markers, mo, odd, pl):
    mo.md("""
    ## Weird II — side quests & alternates (water? views? wrong turns?)

    Stretches >500 ft off the canonical line. Short + returned = side quest.
    Long = alternate (confirm!). *max_ft* is how far off-trail it got.
    """)
    sq = (
        odd.filter(pl.col("type").is_in(["sidequest", "alternate"]))
        .sort("at_route_mi")
        .join_asof(markers.select(["mile", "section"]).rename({"mile": "at_route_mi"}),
                   left_on="at_route_mi", right_on="at_route_mi", strategy="nearest")
        .select(["type", "date", "start_local", "end_local", "mi",
                 "at_route_mi", "section", "detail"])
        .sort("date")
    )
    mo.ui.table(sq)
    return


@app.cell
def _(daily, mo, pl):
    mo.md("""
    ## Weird III — regime changes (caffeine? shoes? tramily? you tell me)

    Greedy binary segmentation on three daily series: it finds the split
    that most reduces variance, then splits again (×3 each). These are
    *candidate* change dates, not conclusions — annotate freely.
    """)
    def sse(y):
        return ((y - y.mean()) ** 2).sum()

    def top_splits(y, k=3, min_len=7):
        segs = [(0, len(y))]
        out = []
        for _ in range(k):
            best = None
            for a, b in segs:
                if b - a < 2 * min_len:
                    continue
                for c in range(a + min_len, b - min_len + 1):
                    gain = sse(y[a:b]) - sse(y[a:c]) - sse(y[c:b])
                    if best is None or gain > best[0]:
                        best = (gain, a, c, b)
            if best is None:
                break
            _, a, c, b = best
            out.append(c)
            segs.remove((a, b))
            segs += [(a, c), (c, b)]
        return sorted(out)

    dates = daily["date"].to_list()
    regs = []
    for col in ["pace_mph", "net_mi", "avg_hr"]:
        y = daily[col].to_numpy()
        for c in top_splits(y):
            regs.append({
                "date": dates[c],
                "series": col,
                "before": round(float(y[:c].mean()), 2),
                "after": round(float(y[c:].mean()), 2),
                "your_annotation": "???",
            })
    mo.ui.table(pl.DataFrame(regs).sort("date"))
    return


@app.cell
def _(daily, mo):
    mo.md("## Zero days — the 10 missing dates (town? rest? rain?)")
    import datetime as dt
    have = set(daily["date"].to_list())
    d0, d1 = dt.date(2026, 4, 25), dt.date(2026, 8, 11)
    missing = [str(d0 + dt.timedelta(days=i)) for i in range((d1 - d0).days + 1)
               if str(d0 + dt.timedelta(days=i)) not in have]
    mo.md("**" + "**, **".join(missing) + "** — annotate: town / rest / weather / chaos?")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Records & superlatives (workshop these)

    - Biggest day, biggest climb, highest/lowest camp: computed above.
    - TODO: fastest mile? (sparse UltraTrac — approximate only)
    - TODO: total footsteps (cadence → strides is in the data).
    - TODO: morning person arc — starts trending earlier/later by state?
    - TODO: rain/heat overlays (no temp sensor on watch — needs external weather).
    - TODO: FarOut mile markers + inReach 20-min fixes as second opinions.

    *Throwing things at the wall per instructions — tune away.*
    """)
    return


if __name__ == "__main__":
    app.run()
