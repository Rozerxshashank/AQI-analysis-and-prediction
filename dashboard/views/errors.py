"""6. Error Analysis — PRD failure classes: missed severe events, large residuals,
boundary-tier misclassification, missing-data driven errors."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import core, ui
from dashboard.tiers import ALL_TIERS

SEVERITY = {t.label: t.severity for t in ALL_TIERS}


def classify(ev: pd.DataFrame, mae: float, rmse: float) -> dict[str, pd.Series]:
    actual_sev = ev["actual_tier"].map(SEVERITY)
    pred_sev = ev["predicted_tier"].map(SEVERITY)
    return {
        "Missed severe events": (actual_sev >= SEVERITY["Very Poor"]) & (pred_sev < actual_sev),
        "Large residuals": ev["abs_error"] > 2 * rmse,
        "Boundary-tier flips": (ev["actual_tier"] != ev["predicted_tier"]) & (ev["abs_error"] <= mae),
        "Missing-data errors": (ev["missing_inputs"] > 0) & (ev["abs_error"] > rmse),
    }


def render():
    art = core.load_artifacts()
    key = st.session_state.model_key
    m = art["results"]["models"][key]
    ev = core.evaluation_frame(key)
    cats = classify(ev, m["MAE"], m["RMSE"])

    ui.header("Error Analysis", "Where and why the model goes wrong",
              f"{core.MODEL_LABELS[key]} on {len(ev):,} held-out days. Failures are sorted into the four classes from the PRD.")
    st.write("")
    defs = {
        "Missed severe events": "Actual tier was Very Poor or Severe, but the model predicted a lower tier.",
        "Large residuals": f"Absolute error above 2 × RMSE (> {2 * m['RMSE']:.0f} AQI points).",
        "Boundary-tier flips": f"Wrong tier even though the error was small (≤ MAE, {m['MAE']:.0f}) — the value sat near a boundary.",
        "Missing-data errors": f"At least one model input was missing (median-filled) and the error exceeded RMSE ({m['RMSE']:.0f}).",
    }
    cols = st.columns(4)
    for col, (name, mask) in zip(cols, cats.items()):
        with col.container(border=True):
            st.markdown(f"<div class='aq-k'>{name}</div><div class='aq-big' style='font-size:38px;margin:6px 0 2px'>{int(mask.sum()):,}</div>"
                        f"<div class='aq-muted'>{mask.mean():.1%} of test days</div>"
                        f"<div style='font-size:13px;color:#4A4F5A;margin-top:8px;min-height:58px'>{defs[name]}</div>",
                        unsafe_allow_html=True)

    st.write("")
    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        by_city = ev.groupby("City").agg(mae=("abs_error", "mean"), bias=("residual", "mean"), n=("abs_error", "size")).sort_values("mae")
        fig = go.Figure(go.Bar(y=by_city.index, x=by_city["mae"], orientation="h", marker_color="#3B5BDB",
                               text=[f"{v:.0f}  (bias {b:+.0f})" for v, b in zip(by_city["mae"], by_city["bias"])], textposition="outside",
                               customdata=by_city["n"], hovertemplate="%{y}: MAE %{x:.1f} over %{customdata} days<extra></extra>"))
        fig.update_xaxes(title="Mean absolute error", range=[0, by_city["mae"].max() * 1.45])
        st.plotly_chart(ui.style_fig(fig, 330).update_layout(title="Error by city (regional shift)"), width="stretch", config=ui.CHART_CONFIG)
        st.caption("Bias > 0 means the model under-predicts that city on average.")
    with b.container(border=True):
        order = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
        by_season = ev.groupby("season").agg(mae=("abs_error", "mean"), bias=("residual", "mean")).reindex(order)
        fig = go.Figure()
        fig.add_bar(x=order, y=by_season["mae"], name="MAE", marker_color="#3B5BDB")
        fig.add_bar(x=order, y=by_season["bias"], name="Bias (actual − predicted)", marker_color="#E8812C")
        st.plotly_chart(ui.style_fig(fig, 330, legend=True).update_layout(title="Error by season", barmode="group"),
                        width="stretch", config=ui.CHART_CONFIG)
        st.caption("Seasons per IMD: Winter Dec–Feb, Pre-monsoon Mar–May, Monsoon Jun–Sep, Post-monsoon Oct–Nov.")

    st.write("")
    a, b = st.columns([1.3, 1], gap="medium")
    with a.container(border=True):
        fig = go.Figure(go.Scattergl(x=ev["AQI"], y=ev["residual"], mode="markers",
                                     marker=dict(size=4, color=[next(t.color for t in ALL_TIERS if t.label == x) for x in ev["actual_tier"]], opacity=0.5),
                                     customdata=ev["City"], hovertemplate="%{customdata}: actual %{x:.0f}, residual %{y:+.0f}<extra></extra>"))
        fig.add_hline(y=0, line_dash="dash", line_color="#16181D")
        fig.update_xaxes(title="Actual AQI").update_yaxes(title="Residual (actual − predicted)")
        st.plotly_chart(ui.style_fig(fig, 340).update_layout(title="Residual vs actual AQI"), width="stretch", config=ui.CHART_CONFIG)
        st.caption("The upward drift at high AQI shows the model under-predicts the most polluted days — the same pattern behind missed severe events.")
    with b.container(border=True):
        g = ev.assign(missing=ev["missing_inputs"].clip(upper=3).map(lambda n: "3+" if n >= 3 else str(n))).groupby("missing").agg(
            mae=("abs_error", "mean"), n=("abs_error", "size"))
        fig = go.Figure(go.Bar(x=[f"{i} missing" for i in g.index], y=g["mae"], marker_color=["#3B5BDB"] + ["#E8812C"] * (len(g) - 1),
                               text=[f"{v:.0f}<br><span style='font-size:10px'>{n:,} days</span>" for v, n in zip(g["mae"], g["n"])],
                               textposition="outside"))
        fig.update_yaxes(title="Mean absolute error", range=[0, g["mae"].max() * 1.35])
        st.plotly_chart(ui.style_fig(fig, 340).update_layout(title="Error vs number of missing inputs"), width="stretch", config=ui.CHART_CONFIG)

    st.write("")
    with st.container(border=True):
        st.markdown("#### Error gallery")
        pick = st.segmented_control("Failure class", list(cats), default=list(cats)[0], key="err_cat")
        rows = ev[cats[pick or list(cats)[0]]].sort_values("abs_error", ascending=False)
        st.markdown(f"<span class='aq-muted'>{len(rows):,} days, largest errors first.</span>", unsafe_allow_html=True)
        show = ["City", "Date"] + core.ALL_POLLUTANTS + ["AQI", "predicted", "residual", "actual_tier", "predicted_tier", "missing_inputs"]
        st.dataframe(rows[show], hide_index=True, width="stretch", height=380, column_config={
            "Date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "AQI": st.column_config.NumberColumn("Actual", format="%.0f"),
            "predicted": st.column_config.NumberColumn("Predicted", format="%.0f"),
            "residual": st.column_config.NumberColumn("Residual", format="%+.0f"),
            "actual_tier": "Actual tier", "predicted_tier": "Predicted tier", "missing_inputs": "Missing inputs",
            **{p: st.column_config.NumberColumn(format="%.1f") for p in core.ALL_POLLUTANTS}})
        st.download_button(f"Download {pick or ''} cases (CSV)", rows[show].to_csv(index=False).encode(),
                           f"error_gallery_{(pick or 'all').lower().replace(' ', '_')}.csv", "text/csv", icon=":material/download:")
