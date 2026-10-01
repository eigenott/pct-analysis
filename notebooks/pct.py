import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt

    return alt, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # PCT 2026 — hike analysis

    Reads the parsed Parquet files in `data/` (built by
    `scripts/build_tracks.py` from the Forerunner 255 FIT files).
    Values are **raw as recorded** — UltraTrac GPS jumps not yet cleaned.
    """)
    return


@app.cell
def _(mo, pl):
    daily = pl.read_parquet("data/daily_summary.parquet").with_columns(
        (pl.col("distance_m") / 1609.344).round(1).alias("miles"),
        (pl.col("ascent_m") * 3.28084).round(0).cast(pl.Int64).alias("ascent_ft"),
        (pl.col("descent_m") * 3.28084).round(0).cast(pl.Int64).alias("descent_ft"),
        (pl.col("timer_s") / 3600).round(1).alias("moving_hrs"),
    )
    mo.ui.table(daily)
    return (daily,)


@app.cell
def _(alt, daily, mo, pl):
    mo.md("## Daily mileage (raw, UltraTrac jumps included)")
    mileage = (
        alt.Chart(daily.with_columns(pl.col("date").str.to_datetime().alias("day")).to_pandas())
        .mark_bar(color="#BBBBF2")
        .encode(
            x=alt.X("day:T", title="date"),
            y=alt.Y("miles:Q", title="miles"),
            tooltip=["date", "miles", "ascent_ft", "avg_hr"],
        )
        .properties(width=800, height=300)
    )
    mileage
    return


@app.cell
def _(daily, mo):
    mo.md("## Single-day elevation profile — pick a date")
    date_picker = mo.ui.dropdown(
        options=daily["date"].to_list(), value=daily["date"][0]
    )
    date_picker
    return (date_picker,)


@app.cell
def _(alt, date_picker, pl):
    day = (
        pl.scan_parquet("data/tracks.parquet")
        .filter(
            (pl.col("date") == date_picker.value)
            & pl.col("lat").is_not_null()
        )
        .select(["timestamp_utc", "altitude_m", "heart_rate"])
        .collect()
        .with_columns((pl.col("altitude_m") * 3.28084).alias("altitude_ft"))
        .to_pandas()
    )
    alt.Chart(day).mark_line(color="#7A7AD6").encode(
        x=alt.X("timestamp_utc:T", title=f"{date_picker.value} (UTC)"),
        y=alt.Y("altitude_ft:Q", title="altitude (ft)", scale=alt.Scale(zero=False)),
        tooltip=["timestamp_utc", "altitude_ft", "heart_rate"],
    ).properties(width=800, height=300)
    return


@app.cell
def _(alt, mo, pl):
    mo.md("## Corrected mileage — teleports cut (red), kept miles (blue)")
    corr = pl.read_parquet("data/daily_corrected.parquet").with_columns(
        pl.col("date").str.to_datetime().alias("day")
    )
    stacked = corr.select(["date", "day", "corrected_mi", "cut_mi"]).unpivot(
        index=["date", "day"], value_name="miles", variable_name="component"
    )
    alt.Chart(stacked.to_pandas()).mark_bar().encode(
        x=alt.X("day:T", title="date"),
        y=alt.Y("miles:Q", title="miles"),
        color=alt.Color(
            "component:N",
            scale=alt.Scale(
                domain=["corrected_mi", "cut_mi"], range=["#BBBBF2", "firebrick"]
            ),
            legend=None,
        ),
        tooltip=["date", "component", "miles"],
    ).properties(width=800, height=300)
    return


@app.cell
def _(alt, mo, pl):
    mo.md("## Along-trail progress — northernmost trail mile by date (flats = zeros, gaps = skips)")
    route = pl.read_parquet("data/daily_route.parquet").with_columns(
        pl.col("date").str.to_datetime().alias("day")
    )
    from pathlib import Path

    import pandas as pd
    if Path("annotations/events.csv").exists():  # private, local-only overlay
        ann = (
            pl.read_csv("annotations/events.csv")
            .with_columns([
                pl.col("start_date").str.to_datetime().alias("start"),
                pl.col("end_date").str.to_datetime().alias("end"),
            ])
            .with_columns(
                pl.when(pl.col("end") > pl.col("start"))
                .then(pl.col("end")).otherwise(pl.col("start") + pl.duration(hours=12))
                .alias("end")
            )
            .to_pandas()
        )
    else:
        ann = pd.DataFrame({"start": pd.to_datetime([]), "end": pd.to_datetime([]),
                            "label": [], "status": []})
    prog = (
        alt.Chart(route.to_pandas())
        .mark_line(point=True, color="#7A7AD6")
        .encode(
            x=alt.X("day:T", title="date"),
            y=alt.Y("route_max_mi:Q", title="trail mile"),
            tooltip=["date", "route_min_mi", "route_max_mi", "net_mi", "offroute_frac"],
        )
        .properties(width=800, height=300)
    )
    spans = (
        alt.Chart(ann)
        .mark_rect(opacity=0.18, color="#7A7AD6")
        .encode(x="start:T", x2="end:T", tooltip=["label", "status"])
    )
    rpd = route.to_pandas()
    x0, x1 = rpd["day"].min(), rpd["day"].max()
    bands = pl.DataFrame({
        "section": ["SoCal", "Sierra", "NorCal", "Oregon", "Washington"],
        "y0": [0, 702, 1092, 1694, 2146],
        "y1": [702, 1092, 1694, 2146, 2660],
    }).with_columns(
        ((pl.col("y0") + pl.col("y1")) / 2).alias("mid"),
        pl.lit(x0).alias("x0"), pl.lit(x1).alias("x1"),
    ).to_pandas()
    sec_bg = (
        alt.Chart(bands).mark_rect(opacity=0.06).encode(
            x="x0:T", x2="x1:T", y="y0:Q", y2="y1:Q")
    )
    sec_text = (
        alt.Chart(bands).mark_text(align="left", dx=6, opacity=0.55, fontSize=11).encode(
            x="x0:T", y="mid:Q", text="section:N")
    )
    (sec_bg + sec_text + prog + spans)
    return (route,)


@app.cell
def _(mo, pl, route):
    mo.md(f"""
    **Along-trail net total: {route['net_mi'].sum():.1f} mi** "
        f"(anchor ~2450). Days more than half off-route (alternate candidates): "
        f"{', '.join(route.filter(pl.col('offroute_frac') > 0.5)['date'].to_list())}.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Next steps / exercises

    - Add a heart-rate panel to the day view (hint: swap `altitude_ft` for
      `heart_rate`, or layer both with `alt.layer`).
    - Colour mileage bars by elevation gain to spot big climbing days.
    - 2026-07-21 corrected to ~41 mi — the monster day survives the cuts,
      exactly as designed.
    - Next: jitter smoothing. ~227 mi of tree-cover bounce remains in the
      corrected total; a light median/speed filter on sub-656 ft segments
      should close most of the gap to the ~2450 anchor.
    - The route projection already nets 2442.8 mi vs the anchor — use
      `daily_route.parquet` (`net_mi`, `offroute_frac`) to confirm the
      alternate-route days and the three skipped sections (05-28, 06-23,
      07-24) plus the sub-9 mi southbound day (07-16).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
