"""5. Evaluation — MAE, RMSE, R², actual vs predicted, residuals, error cases (PRD: Evaluation Screen)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import core, ui
from dashboard.tiers import ALL_TIERS


def render():
    art = core.load_artifacts()
    key = st.session_state.model_key
    m = art["results"]["models"][key]
    ev = core.evaluation_frame(key)
    cities = ev["City"].unique()

    ui.header("Evaluation", f"How well does {core.MODEL_LABELS[key].split(' (')[0].lower()} predict AQI?",
              f"Scored on <b>{len(ev):,} held-out days</b> from {len(cities)} cities the model never saw "
              f"({', '.join(cities)}). Splitting by city keeps adjacent days from the same station out of both sets.")
    st.write("")
    c = st.columns(4)
    c[0].metric("MAE", f"{m['MAE']:.1f}", help="Mean absolute error, AQI points")
    c[1].metric("RMSE", f"{m['RMSE']:.1f}", help="Root mean squared error, AQI points")
    c[2].metric("R²", f"{m['R2']:.3f}", help="Share of AQI variance explained")
    tier_acc = (ev["actual_tier"] == ev["predicted_tier"]).mean()
    c[3].metric("Correct health tier", f"{tier_acc:.0%}", help="Predicted tier equals the tier of the recorded AQI")

    st.write("")
    with st.container(border=True):
        st.markdown("#### All models on the same held-out set")
        rows = []
        for k, r in sorted(art["results"]["models"].items(), key=lambda kv: -kv[1]["R2"]):
            rows.append({"Model": core.MODEL_LABELS[k], "Version": art["versions"][k].split(" ·")[0],
                         "MAE": r["MAE"], "RMSE": r["RMSE"], "R²": r["R2"], "Train time (s)": r["train_seconds"],
                         "Selected": "●" if k == key else ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "MAE": st.column_config.NumberColumn(format="%.2f"), "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "R²": st.column_config.ProgressColumn(format="%.4f", min_value=0.8, max_value=1.0),
            "Train time (s)": st.column_config.NumberColumn(format="%.2f"), "Selected": st.column_config.TextColumn(width="small")})
        R = art["results"]["models"]
        gap = R["sy_rf"]["R2"] - R["sk_rf"]["R2"]
        rf_note = (f"the scratch Random Forest (10 depth-5 trees) beats sklearn's 100 unlimited-depth trees by {gap:.3f} R²"
                   if gap > 0 else
                   f"the scratch Random Forest (10 depth-5 trees) is within {abs(gap):.3f} R² of sklearn's 100 unlimited-depth trees")
        st.caption("Linear Regression and Decision Tree written from scratch reproduce sklearn's predictions exactly; " + rf_note + ".")

    st.write("")
    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        top = float(np.ceil(max(ev["AQI"].max(), ev["predicted"].max()) / 100) * 100)
        fig = go.Figure()
        lo = 0
        for t, hi in zip(ALL_TIERS, [50, 100, 200, 300, 400, top]):
            fig.add_shape(type="rect", x0=lo, x1=hi, y0=lo, y1=hi, fillcolor=t.color, opacity=0.12, line_width=0, layer="below")
            lo = hi
        for t in ALL_TIERS:
            d = ev[ev["actual_tier"] == t.label]
            if len(d):
                fig.add_trace(go.Scattergl(x=d["AQI"], y=d["predicted"], mode="markers", name=t.label,
                                           marker=dict(size=4.5, color=t.color, opacity=0.55),
                                           customdata=np.c_[d["City"], d["Date"].dt.strftime("%d %b %Y")],
                                           hovertemplate="%{customdata[0]} · %{customdata[1]}<br>actual %{x:.0f} · predicted %{y:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color="#16181D", dash="dash", width=1.2),
                                 name="perfect", hoverinfo="skip"))
        fig.update_xaxes(title="Actual AQI", range=[0, top]).update_yaxes(title="Predicted AQI", range=[0, top])
        st.plotly_chart(ui.style_fig(fig, 430, legend=True).update_layout(title="Actual vs predicted"), width="stretch", config=ui.CHART_CONFIG)
        st.caption("Shaded squares mark where the predicted tier matches the actual tier. Colour = actual tier.")
    with b.container(border=True):
        fig = go.Figure(go.Histogram(x=ev["residual"].clip(-200, 200), xbins=dict(start=-200, end=200, size=10),
                                     marker_color="#3B5BDB", opacity=0.85, hovertemplate="%{x}: %{y} days<extra></extra>"))
        fig.add_vline(x=0, line_color="#16181D", line_dash="dash")
        fig.add_vline(x=ev["residual"].mean(), line_color="#E8812C",
                      annotation_text=f"mean {ev['residual'].mean():+.1f}", annotation_position="top right")
        fig.update_xaxes(title="Residual = actual − predicted (clipped to ±200)").update_yaxes(title="Days")
        st.plotly_chart(ui.style_fig(fig, 430).update_layout(title="Residual distribution", bargap=0.05), width="stretch", config=ui.CHART_CONFIG)
        st.caption("Right of zero: model predicted too low. Left: too high.")

    st.write("")
    a, b = st.columns([1, 1.2], gap="medium")
    with a.container(border=True):
        labels = [t.label for t in ALL_TIERS]
        cm = pd.crosstab(pd.Categorical(ev["actual_tier"], labels), pd.Categorical(ev["predicted_tier"], labels), dropna=False)
        pct = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        fig = go.Figure(go.Heatmap(z=pct.values, x=labels, y=labels, colorscale=[[0, "#F7F6F2"], [1, "#3B5BDB"]], showscale=False,
                                   text=cm.values, texttemplate="%{text}", hovertemplate="actual %{y} → predicted %{x}: %{text} days<extra></extra>"))
        fig.update_xaxes(title="Predicted tier", side="bottom").update_yaxes(title="Actual tier", autorange="reversed")
        st.plotly_chart(ui.style_fig(fig, 400).update_layout(title="Health-tier confusion (days)"), width="stretch", config=ui.CHART_CONFIG)
    with b.container(border=True):
        st.markdown("#### Example error cases")
        worst = ev.nlargest(12, "abs_error")
        st.dataframe(worst[["City", "Date", "AQI", "predicted", "residual", "actual_tier", "predicted_tier", "missing_inputs"]],
                     hide_index=True, width="stretch", height=390, column_config={
                         "Date": st.column_config.DateColumn(format="DD MMM YYYY"),
                         "AQI": st.column_config.NumberColumn("Actual", format="%.0f"),
                         "predicted": st.column_config.NumberColumn("Predicted", format="%.0f"),
                         "residual": st.column_config.NumberColumn("Residual", format="%+.0f"),
                         "actual_tier": "Actual tier", "predicted_tier": "Predicted tier",
                         "missing_inputs": st.column_config.NumberColumn("Missing inputs")})
        st.caption("The 12 largest absolute errors. Full breakdown on the Error Analysis screen.")
