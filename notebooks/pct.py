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
def _(alt, daily, mo):
    mo.md("## Daily mileage (raw, UltraTrac jumps included)")
    mileage = (
        alt.Chart(daily.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("date:O", title="date", axis=alt.Axis(labelAngle=-60)),
            y=alt.Y("miles:Q", title="miles"),
            tooltip=["date", "miles", "ascent_ft", "avg_hr"],
        )
        .properties(height=300)
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
    alt.Chart(day).mark_line().encode(
        x=alt.X("timestamp_utc:T", title=f"{date_picker.value} (UTC)"),
        y=alt.Y("altitude_ft:Q", title="altitude (ft)", scale=alt.Scale(zero=False)),
        tooltip=["timestamp_utc", "altitude_ft", "heart_rate"],
    ).properties(height=300)
    return


@app.cell
def _(alt, mo, pl):
    mo.md("## Corrected mileage — teleports cut (red), kept miles (blue)")
    corr = pl.read_parquet("data/daily_corrected.parquet")
    stacked = corr.select(["date", "corrected_mi", "cut_mi"]).unpivot(
        index="date", value_name="miles", variable_name="component"
    )
    alt.Chart(stacked.to_pandas()).mark_bar().encode(
        x=alt.X("date:O", title="date", axis=alt.Axis(labelAngle=-60)),
        y=alt.Y("miles:Q", title="miles"),
        color=alt.Color(
            "component:N",
            scale=alt.Scale(
                domain=["corrected_mi", "cut_mi"], range=["steelblue", "firebrick"]
            ),
            legend=None,
        ),
        tooltip=["date", "component", "miles"],
    ).properties(height=300)
    return (corr,)


@app.cell
def _(corr, mo, pl):
    mo.md(
        f"**Totals:** {corr['corrected_mi'].sum():.0f} mi kept, "
        f"{corr['cut_mi'].sum():.0f} mi cut across "
        f"{corr.filter(pl.col('cut_mi') > 0).height} days "
        f"(anchor: ~2450 recorded mi — the rest is jitter, see below)."
    )
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
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
