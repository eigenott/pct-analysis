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
def _(mo):
    mo.md("""
    ## Next steps / exercises

    - Add a heart-rate panel to the day view (hint: swap `altitude_ft` for
      `heart_rate`, or layer both with `alt.layer`).
    - Colour mileage bars by elevation gain to spot big climbing days.
        - 2026-07-21 (~43 mi) is a confirmed legit monster day — use it as a
          reference when tuning the UltraTrac jump cleanup: real
          hiking can look extreme, so cuts must key on impossible
          point-to-point speed, not daily totals.
        - See `scripts/compare_heuristics.py` for 5 candidate cut rules
          and how much each removes; next step is picking one and
          producing corrected daily mileage alongside raw.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
