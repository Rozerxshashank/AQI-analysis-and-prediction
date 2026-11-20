"""2. Input & Predict — manual entry, recorded city/date, batch CSV (PRD: Input Screen)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import core, nav, ui
from dashboard.tiers import ALL_TIERS
from data import FEATURES, TARGET

PRESETS = {
    "Clean day": {"PM2.5": 18.0, "PM10": 40.0, "NO2": 12.0, "SO2": 6.0, "CO": 0.4, "O3": 28.0},
    "Typical city": {"PM2.5": 48.6, "PM10": 95.7, "NO2": 21.7, "SO2": 9.2, "CO": 0.9, "O3": 31.0},
    "Winter smog": {"PM2.5": 210.0, "PM10": 340.0, "NO2": 68.0, "SO2": 18.0, "CO": 2.6, "O3": 22.0},
    "Clear form": {p: None for p in core.ALL_POLLUTANTS},
}


def _run(values, city=None, date=None, actual=None):
    key = st.session_state.model_key
    with st.status("Running prediction pipeline…", expanded=True) as status:
        job = core.predict_reading(values, key, city=city, date=date, actual=actual,
                                   on_state=lambda s: st.write(f"`{s}`"))
        if job["status"] == "PROCESSING_FAILED":
            status.update(label="Invalid input — nothing was predicted", state="error")
        else:
            status.update(label=f"Done · {job['status']}", state="complete")
    if job["status"] == "PROCESSING_FAILED":
        st.error("**Please fix the following:**\n\n" + "\n".join(f"- {e}" for e in job["errors"]))
        return
    st.session_state.result = job
    st.switch_page(nav.PAGES["result"])


def _manual():
    st.markdown("<div class='aq-muted'>Enter 24-hour average concentrations. Leave a field empty if it wasn't measured — "
                "it will be filled with the training median and the prediction flagged for review.</div>", unsafe_allow_html=True)
    st.write("")
    preset = st.pills("Quick fill", list(PRESETS), selection_mode="single", key="preset_pick", label_visibility="collapsed")
    if preset and st.session_state.get("_last_preset") != preset:
        for p, v in PRESETS[preset].items():
            st.session_state[f"in_{p}"] = v
        st.session_state._last_preset = preset
    if not preset:
        st.session_state._last_preset = None

    cities = sorted(core.dataset()["City"].unique())
    with st.form("manual"):
        a, b = st.columns(2)
        city = a.selectbox("City (optional)", ["Not specified"] + cities)
        date = b.date_input("Date (optional)", value=None, format="DD/MM/YYYY")
        cols = st.columns(3)
        values = {}
        for i, p in enumerate(core.ALL_POLLUTANTS):
            with cols[i % 3]:
                values[p] = st.number_input(
                    f"{p} ({core.UNITS[p]})", min_value=0.0, max_value=float(core.PLAUSIBLE_MAX[p]), value=None,
                    step=0.1 if p != "CO" else 0.01, format="%.2f", key=f"in_{p}", placeholder="not measured",
                    help=f"{core.FULL_NAMES[p]}. CPCB 24-h safe limit: {core.SAFE_LIMITS[p]} {core.UNITS[p]}."
                         + (" A value of 0 is treated as not measured." if p == "CO" else ""))
        submitted = st.form_submit_button("Predict AQI", type="primary", icon=":material/play_arrow:")
    if submitted:
        _run(values, city=None if city == "Not specified" else city, date=date)


def _recorded():
    df = core.dataset()
    train, test = core.split_frames()
    test_idx = set(test.index)
    st.markdown("<div class='aq-muted'>Pick a real day from the dataset. The recorded AQI is shown next to the prediction "
                "so you can see how close the model gets.</div>", unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([1, 1])
    city = a.selectbox("City", sorted(df["City"].unique()), index=sorted(df["City"].unique()).index("Delhi"), key="rec_city")
    days = df[(df["City"] == city) & df[TARGET].notna()].set_index("Date")
    date = b.date_input("Date", value=days.index.max().date(), min_value=days.index.min().date(),
                        max_value=days.index.max().date(), format="DD/MM/YYYY", key="rec_date")
    ts = pd.Timestamp(date)
    if ts not in days.index:
        st.warning(f"No AQI was recorded in {city} on {date:%d %b %Y}. Pick another date "
                   f"({days.index.min():%d %b %Y} – {days.index.max():%d %b %Y}).")
        return
    row = days.loc[ts]
    held_out = df[(df["City"] == city) & (df["Date"] == ts)].index[0] in test_idx
    st.markdown(f"<span class='aq-muted'>{'Held-out day — the model never trained on it.' if held_out else 'This day was part of the training data.'}</span>",
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{p: row[p] for p in core.ALL_POLLUTANTS}]), hide_index=True, width="stretch",
                 column_config={p: st.column_config.NumberColumn(f"{p} ({core.UNITS[p]})", format="%.2f") for p in core.ALL_POLLUTANTS})
    if st.button("Predict this day", type="primary", icon=":material/play_arrow:"):
        values = {p: (None if pd.isna(row[p]) else float(row[p])) for p in core.ALL_POLLUTANTS}
        _run(values, city=city, date=date, actual=float(row[TARGET]))


def _batch():
    st.markdown(f"<div class='aq-muted'>CSV with one reading per row. Columns: <span class='aq-mono'>{', '.join(core.ALL_POLLUTANTS)}</span> "
                f"(at least one of the model inputs), optional <span class='aq-mono'>City, Date</span>. "
                f"Up to {core.MAX_UPLOAD_MB} MB / {core.MAX_ROWS:,} rows.</div>", unsafe_allow_html=True)
    st.download_button("Download CSV template", core.template_csv(), "aqi_template.csv", "text/csv", icon=":material/description:")
    up = st.file_uploader("Upload pollutant CSV", type=["csv"], label_visibility="collapsed")
    if up is None:
        return
    df, errors = core.read_csv_upload(up)
    if errors:
        st.error("**This file can't be used.**\n\n" + "\n".join(f"- {e}" for e in errors))
        return
    present = [f for f in FEATURES if f in df.columns]
    absent = [f for f in FEATURES if f not in df.columns]
    c = st.columns(3)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Model inputs found", f"{len(present)} of {len(FEATURES)}")
    c[2].metric("Missing cells", f"{int(df[present].isna().sum().sum()):,}")
    if absent:
        st.warning(f"Column{'s' if len(absent) > 1 else ''} {', '.join(absent)} not found — filled with the training median and rows flagged for review.")
    with st.expander("Preview first rows"):
        st.dataframe(df.head(20), hide_index=True, width="stretch")

    if st.button(f"Score {len(df):,} rows", type="primary", icon=":material/play_arrow:"):
        bar = st.progress(0.0, text="Processing…")
        rows, summary = core.predict_batch(df, st.session_state.model_key, progress=lambda p: bar.progress(p, text=f"Processing… {p:.0%}"))
        bar.progress(1.0, text="Done")
        st.session_state.batch = {"rows": rows, "summary": summary, "filename": up.name}

    batch = st.session_state.get("batch")
    if batch and batch["filename"] == up.name:
        s, rows = batch["summary"], batch["rows"]
        skipped = sum(s["skipped"].values())
        st.success(f"Scored {s['rows_scored']:,} of {s['rows_total']:,} rows with `{s['model_version']}`."
                   + (f" Skipped {skipped:,} unusable rows ({', '.join(f'{v} {k}' for k, v in s['skipped'].items() if v)})." if skipped else ""))
        c = st.columns([1, 2])
        c[0].metric("Flagged for review", f"{s['review_required']:,}", help="Borderline AQI or imputed inputs")
        fig = go.Figure(go.Bar(x=[t.label for t in ALL_TIERS], y=[s["tiers"].get(t.label, 0) for t in ALL_TIERS],
                               marker_color=[t.color for t in ALL_TIERS], text=[s["tiers"].get(t.label, 0) for t in ALL_TIERS],
                               textposition="outside"))
        c[1].plotly_chart(ui.style_fig(fig, 220).update_layout(title="Rows per health tier"), width="stretch", config=ui.CHART_CONFIG)
        show = [c for c in ["City", "Date"] if c in rows.columns] + FEATURES + ["predicted_aqi", "tier", "review_required", "imputed_fields"]
        st.dataframe(rows[show], hide_index=True, width="stretch", height=320,
                     column_config={"predicted_aqi": st.column_config.NumberColumn("Predicted AQI", format="%.0f"),
                                    "tier": "Health tier", "imputed_fields": "Median-filled",
                                    "review_required": st.column_config.CheckboxColumn("Review")})
        st.page_link(nav.PAGES["export"], label="Download results on the Export screen", icon=":material/download:")


def render():
    ui.header("Input & Predict", "Enter or upload pollutant readings",
              f"Predictions use <b>{core.MODEL_LABELS[st.session_state.model_key]}</b> — change it in the sidebar.")
    st.write("")
    t1, t2, t3 = st.tabs(["Manual entry", "Recorded city & date", "Batch CSV upload"])
    with t1:
        _manual()
    with t2:
        _recorded()
    with t3:
        _batch()
