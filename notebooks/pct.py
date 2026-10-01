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
    (≥0.5 km of bogus distance) are queued below — biggest first.
    For each: check the map, click ✅ or ❌, and you'll auto-advance
    to the next undecided candidate. ◀ Prev goes back to re-vote
    (re-voting overwrites your earlier call).
    Your calls live in `data/jump_verdicts.csv` (scripts never
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
    review_ids = (
        cand.filter(pl.col("jump_km") >= REVIEW_MIN_KM)
        .sort("rank")["jump_id"]
        .to_list()
    )  # fixed order — stable even as verdicts accumulate
    status = (
        f"**Progress:** {decided.height} decided, "
        f"{queue.height} left in queue (≥{REVIEW_MIN_KM} km)"
    )
    mo.md(status + (" — **all reviewed! 🎉**" if queue.height == 0 else ""))
    return VERDICTS, cand, decided, queue, review_ids


@app.cell
def _(decided, mo, review_ids):
    get_pos, set_pos = mo.state(0)
    decided_ids = set(decided["jump_id"].to_list()) if decided.height else set()
    n = len(review_ids)
    pos = min(max(get_pos(), 0), max(n - 1, 0)) if n else 0
    cur_id = review_ids[pos] if n else -1
    prev_btn = mo.ui.button(
        label="◀ Prev", on_click=lambda _: set_pos(max(0, get_pos() - 1))
    )
    next_btn = mo.ui.button(
        label="Next ▶",
        on_click=lambda _: set_pos(min(max(n - 1, 0), get_pos() + 1)),
    )
    mo.hstack(
        [prev_btn, mo.md(f"**Candidate {pos + 1} of {n}**"), next_btn],
        justify="space-between",
    )
    return cur_id, get_pos, pos, set_pos


@app.cell
def _(cand, cur_id, decided, mo, pl):
    import plotly.graph_objects as go

    if cur_id == -1:
        mo.stop(True, mo.md("Queue empty — all reviewed! 🎉"))
    sel = cand.filter(pl.col("jump_id") == cur_id).row(0, named=True)
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
def _(VERDICTS, cur_id, decided, fig, get_pos, mo, pl, pos, review_ids, set_pos):
    import csv
    from datetime import datetime, timezone

    def _save(jump_id, verdict):
        rows = []
        if VERDICTS.exists():
            with open(VERDICTS, newline="") as f:
                r = list(csv.reader(f))
            rows = [x for x in r[1:] if x and x[0] != str(jump_id)]
        with open(VERDICTS, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["jump_id", "verdict", "decided_at"])
            w.writerows(rows)
            w.writerow(
                [jump_id, verdict, datetime.now(timezone.utc).isoformat()]
            )

    voted_ids = set(decided["jump_id"].to_list()) if decided.height else set()
    nxt = pos + 1
    while nxt < len(review_ids) and review_ids[nxt] in voted_ids:
        nxt += 1  # land on the next undecided candidate
    nxt = min(nxt, len(review_ids) - 1)

    yes_btn = mo.ui.button(
        label="✅ Yes — GPS jump, cut it",
        kind="danger",
        on_click=lambda _: (_save(cur_id, "jump"), set_pos(nxt)),
    )
    no_btn = mo.ui.button(
        label="❌ No — legit (hitch, etc.), keep it",
        kind="success",
        on_click=lambda _: (_save(cur_id, "legit"), set_pos(nxt)),
    )
    prior = decided.filter(pl.col("jump_id") == cur_id)
    vote_status = (
        mo.md(f"Your call: **{prior.row(0, named=True)['verdict']}** (click again to change)")
        if prior.height
        else mo.md("Not yet reviewed — your call will appear here.")
    )
    mo.vstack(
        [fig, mo.hstack([yes_btn, no_btn], gap=1, justify="center"), vote_status],
        gap=1,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
