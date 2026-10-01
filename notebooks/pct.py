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
      reference when tuning the per-point UltraTrac jump detector: real
      hiking can look extreme, so the detector must key on impossible
      point-to-point speed, not daily totals.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Jump review — human in the loop

    Segments faster than 12 m/s (27 mph — impossible on foot) were
    grouped into events by `scripts/flag_jumps.py`. The top 90 events
    (≥0.5 mi… er, ≥0.5 km of bogus distance) are queued below — biggest
    first. For each: check the map, then verdict.
    Your calls are appended to `data/jump_verdicts.csv` (scripts never
    touch that file, so re-running the detector won't lose your work).
    """)
    return


@app.cell
def _(mo, pl):
    from pathlib import Path

    REVIEW_MIN_KM = 0.5
    cand = pl.read_parquet("data/jump_candidates.parquet").sort("rank")
    VERDICTS = Path("data/jump_verdicts.csv")
    decided = (
        pl.read_csv(VERDICTS)
        if VERDICTS.exists()
        else pl.DataFrame(schema={"jump_id": pl.Int64, "verdict": pl.String})
    )
    queue = cand.join(
        decided.select("jump_id"), on="jump_id", how="anti"
    ).filter(pl.col("jump_km") >= REVIEW_MIN_KM)
    mo.md(
        f"**Progress:** {decided.height} decided, "
        f"{queue.height} left in queue (≥{REVIEW_MIN_KM} km), "
        f"{cand.height - decided.height - queue.height} small fry deferred"
    )
    return VERDICTS, cand, queue


@app.cell
def _(mo, queue):
    options = {
        f"#{r['rank']} {r['date']} — {r['jump_km']:.2f} km, "
        f"max {r['max_speed_m_s']:.0f} m/s": r["jump_id"]
        for r in queue.iter_rows(named=True)
    }
    picker = mo.ui.dropdown(
        options=options or {"(queue empty — all reviewed!)": -1},
        value=next(iter(options)) if options else "(queue empty — all reviewed!)",
    )
    picker
    return (picker,)


@app.cell
def _(cand, mo, picker, pl):
    import plotly.graph_objects as go

    if picker.value is None or picker.value == -1:
        mo.stop(True, mo.md("Queue empty — all reviewed! 🎉"))
    sel = cand.filter(pl.col("jump_id") == picker.value).row(0, named=True)
    window = (
        pl.scan_parquet("data/tracks.parquet")
        .filter(
            (pl.col("source_file") == sel["source_file"])
            & (pl.col("point_index") >= sel["pt_start"] - 40)
            & (pl.col("point_index") <= sel["pt_end"] + 40)
            & pl.col("lat").is_not_null()
        )
        .select(["point_index", "lat", "lon"])
        .collect()
        .sort("point_index")
    )
    flagged = window.filter(
        (pl.col("point_index") >= sel["pt_start"])
        & (pl.col("point_index") <= sel["pt_end"])
    )
    zoom = 12 if sel["jump_km"] < 2 else (10 if sel["jump_km"] < 10 else (8 if sel["jump_km"] < 50 else 6))
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=window["lat"], lon=window["lon"],
        mode="lines+markers", marker={"size": 5, "color": "blue"},
        name="track", hoverinfo="skip",
    ))
    fig.add_trace(go.Scattermap(
        lat=flagged["lat"], lon=flagged["lon"],
        mode="lines+markers", marker={"size": 9, "color": "red"},
        name="flagged", hoverinfo="skip",
    ))
    fig.update_layout(
        map_style="carto-positron",
        map_center={"lat": float(flagged["lat"].mean()), "lon": float(flagged["lon"].mean())},
        map_zoom=zoom, height=450, margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
    )
    mo.md(
        f"**Jump #{sel['rank']}** — {sel['date']}, {sel['jump_km']:.2f} km over "
        f"{sel['n_flagged_seg']} segments, max {sel['max_speed_m_s']:.0f} m/s "
        f"(worst gap {sel['worst_dt_s']:.0f}s)"
    )
    return (fig,)


@app.cell
def _(VERDICTS, fig, mo, picker):
    import csv
    from datetime import datetime, timezone

    def _save(jump_id, verdict):
        new = not VERDICTS.exists()
        with open(VERDICTS, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["jump_id", "verdict", "decided_at"])
            w.writerow([jump_id, verdict, datetime.now(timezone.utc).isoformat()])

    jid = picker.value
    yes_btn = mo.ui.button(
        label="✅ Yes — GPS jump, cut it",
        kind="danger",
        on_click=lambda _: _save(jid, "jump"),
    )
    no_btn = mo.ui.button(
        label="❌ No — legit (hitch, etc.), keep it",
        kind="success",
        on_click=lambda _: _save(jid, "legit"),
    )
    status = (
        mo.md(f"Verdicts saved so far: **{sum(1 for _ in open(VERDICTS)) - 1}**")
        if VERDICTS.exists()
        else mo.md("No verdicts yet — your calls will appear here.")
    )
    mo.vstack([fig, mo.hstack([yes_btn, no_btn]), status])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
