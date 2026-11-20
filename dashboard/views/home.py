"""1. Home / Overview"""
import streamlit as st

from dashboard import core, nav, ui


def render():
    art = core.load_artifacts()
    df = core.dataset()
    key = st.session_state.model_key
    m = art["results"]["models"][key]

    ui.header("Environmental science · Smart cities", "Air Quality Index prediction &amp; analysis",
              "Turn raw pollutant readings into a predicted AQI, a rule-based CPCB health tier and a health advisory — "
              "with the evidence, uncertainty and error analysis needed to trust it.")

    st.write("")
    c = st.columns(4)
    c[0].metric("Observations", f"{len(df):,}", help="Daily city readings in city_day.csv")
    c[1].metric("Cities", df["City"].nunique())
    c[2].metric("Period", f"{df['Date'].min():%Y}–{df['Date'].max():%Y}")
    c[3].metric(f"Test R² · {core.MODEL_LABELS[key].split(' (')[0]}", f"{m['R2']:.3f}",
                help=f"Held-out MAE {m['MAE']:.1f}, RMSE {m['RMSE']:.1f}")

    st.write("")
    with st.container(border=True):
        st.markdown("#### How a prediction is made")
        st.markdown(
            "<div class='aq-flow'>"
            "<span class='n'>Pollutant readings<br><span class='aq-muted'>manual · city/date · CSV</span></span><span class='a'>→</span>"
            "<span class='n'>Validation</span><span class='a'>→</span>"
            "<span class='n'>Cleaning · outlier caps · median fill</span><span class='a'>→</span>"
            "<span class='n hl'>Regression model</span><span class='a'>→</span>"
            "<span class='n'>Predicted AQI ± error band</span><span class='a'>→</span>"
            "<span class='n hl'>Rule-based tier (CPCB)</span><span class='a'>→</span>"
            "<span class='n'>Advisory + SHAP drivers</span><span class='a'>→</span>"
            "<span class='n'>Review flag · PDF</span>"
            "</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown(ui.tier_scale(), unsafe_allow_html=True)
        st.caption("Health tiers follow the CPCB National AQI standard and are assigned with deterministic if/elif "
                   "thresholds on the predicted AQI. Every badge carries its text label, not just a colour.")

    st.write("")
    st.markdown("#### Where to go")
    links = [
        ("predict", "Enter readings by hand, load a recorded city/date, or upload a CSV for batch scoring."),
        ("result", "Predicted AQI, tier badge, error band, top contributing pollutants and the advisory."),
        ("map", "Folium heatmap of average AQI or any pollutant across 26 Indian cities."),
        ("evaluation", "MAE, RMSE, R², actual vs predicted, residuals and tier confusion on held-out cities."),
        ("errors", "Missed severe events, large residuals, boundary flips and missing-data errors."),
        ("export", "Download the PDF health advisory, batch predictions and the evaluation report."),
    ]
    cols = st.columns(3)
    for i, (k, desc) in enumerate(links):
        with cols[i % 3].container(border=True):
            st.page_link(nav.PAGES[k], label=f"**{nav.PAGES[k].title}**")
            st.markdown(f"<div class='aq-muted' style='min-height:44px'>{desc}</div>", unsafe_allow_html=True)

    st.write("")
    ui.callout(ui.SCOPE_TEXT, "scope")

    with st.expander("Model card"):
        st.markdown(f"""
**Model** — {core.MODEL_LABELS[key]}, `{art['versions'][key]}`. Other models can be selected in the sidebar
(Linear Regression, Decision Tree and Random Forest written from scratch with NumPy, plus sklearn and XGBoost baselines).

**Inputs** — PM2.5, PM10, NO2, SO2, CO, O3 (24-hour averages). PM10 is 37.7% missing, but CPCB AQI is the highest
pollutant sub-index and PM10 sets it on 65% of days that have both particulates, so it is kept and median-filled when absent.

**Cleaning** — rows without AQI or with AQI above 500 (off the CPCB scale) are dropped; CO = 0 is treated as missing;
each pollutant is capped at the 99.5th percentile of training data; gaps are filled with the training median.

**Intended use** — environmental analytics and smart-city dashboard prototyping; AQI estimation and
health-tier communication from pollutant readings.

**Out of scope** — certified regulatory monitoring, autonomous legal enforcement, safety-critical medical decisions.

**Data** — Air Quality Data in India (2015–2020), Kaggle (Rohan Rao), `city_day.csv`: {len(df):,} rows,
{df['City'].nunique()} cities, {df['Date'].min():%d %b %Y} – {df['Date'].max():%d %b %Y}; {df['AQI'].notna().sum():,} rows have AQI.

**Evaluation** — held-out test set of {len(core.split_frames()[1]):,} days from six whole cities the model never saw
({', '.join(sorted(core.split_frames()[1]['City'].unique()))}), so scores measure generalisation to new cities.

**Limitations** — missing sensor data, regional and seasonal shift, and reliance on historical patterns that may
not hold during unprecedented pollution events. Borderline predictions are flagged for manual review.
""")
