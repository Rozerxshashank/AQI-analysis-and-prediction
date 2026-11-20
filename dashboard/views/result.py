"""3. Prediction Result — AQI, tier badge, SHAP drivers, error margin, advisory (PRD: Result Screen)."""
import plotly.graph_objects as go
import streamlit as st

from dashboard import core, nav, report, ui
from dashboard.tiers import get_health_tier


def _shap_chart(job):
    items = sorted(job["shap"].items(), key=lambda kv: -abs(kv[1]))
    base, pred = job["shap_base"], job["predicted_aqi"]
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(items) + ["total"],
        y=["Average prediction"] + [k for k, _ in items] + ["This prediction"],
        x=[base] + [v for _, v in items] + [pred],
        text=[f"{base:.0f}"] + [f"{v:+.1f}" for _, v in items] + [f"{pred:.0f}"], textposition="outside",
        increasing=dict(marker=dict(color="#E8812C")), decreasing=dict(marker=dict(color="#3B5BDB")),
        totals=dict(marker=dict(color=get_health_tier(pred).color)),
        connector=dict(line=dict(color="#D6D2C8", width=1)),
    ))
    lo = min(0, base + min(0, sum(v for _, v in items)))
    fig.update_xaxes(title="AQI", range=[lo, max(base, pred, base + sum(max(0, v) for _, v in items)) * 1.18])
    fig.update_yaxes(autorange="reversed")
    return ui.style_fig(fig, 330).update_layout(title="Top contributing pollutants (SHAP)")


def render():
    job = st.session_state.get("result")
    ui.header("Prediction Result", "What the model predicts, and how sure it is")
    if not job:
        st.write("")
        with st.container(border=True):
            st.markdown("#### No prediction yet")
            st.markdown("<span class='aq-muted'>Enter or upload readings first — the result will appear here.</span>", unsafe_allow_html=True)
            st.page_link(nav.PAGES["predict"], label="Go to Input & Predict", icon=":material/arrow_forward:")
        return

    tier = job["tier"]
    st.markdown(ui.stepper(job["states"]), unsafe_allow_html=True)
    where = " · ".join(x for x in [job.get("city"), f"{job['date']:%d %b %Y}" if job.get("date") else None] if x)
    st.markdown(f"<div class='aq-muted aq-mono' style='font-size:12px;margin:6px 0 16px'>{job['job_id']} · {job['model_version']} · "
                f"{job['latency_ms']:.0f} ms{' · ' + where if where else ''}</div>", unsafe_allow_html=True)

    if job["model_key"] != st.session_state.model_key:
        st.info(f"This result was produced by **{core.MODEL_LABELS[job['model_key']]}**. Re-run it on the Input screen to use the model selected in the sidebar.")

    with st.container(border=True):
        g, info = st.columns([1, 1.15], gap="large")
        with g:
            st.markdown(ui.gauge(job["predicted_aqi"], job.get("actual"), job["interval"]), unsafe_allow_html=True)
        with info:
            st.markdown(f"<div class='aq-k'>Health tier · CPCB</div><div style='margin:8px 0 10px'>{ui.tier_badge(tier)}"
                        f"<span class='aq-muted aq-mono' style='margin-left:10px'>AQI {tier.range_text}</span></div>"
                        f"<div style='font-size:16px;font-weight:600;color:#16181D;line-height:1.45'>{tier.health_impact}</div>",
                        unsafe_allow_html=True)
            lo, hi = job["interval"]
            st.markdown(f"<div style='margin-top:14px' class='aq-k'>Error margin</div>"
                        f"<div style='font-size:14.5px;color:#4A4F5A'>On held-out cities, 80% of this model's predictions were within "
                        f"<b class='aq-mono'>{lo - job['predicted_aqi']:+.0f} / {hi - job['predicted_aqi']:+.0f}</b> of the truth — "
                        f"likely range <b class='aq-mono'>{lo:.0f}–{hi:.0f}</b> "
                        f"({get_health_tier(lo).label} to {get_health_tier(hi).label}).</div>", unsafe_allow_html=True)
            if job.get("tree_spread") is not None:
                st.markdown(f"<div class='aq-muted' style='margin-top:6px'>Trees in the forest disagree by ±{job['tree_spread']:.0f} AQI (std of votes).</div>",
                            unsafe_allow_html=True)
            if job.get("actual") is not None:
                err = job["predicted_aqi"] - job["actual"]
                at = get_health_tier(job["actual"])
                st.markdown(f"<div style='margin-top:14px' class='aq-k'>Recorded that day</div>"
                            f"<div style='font-size:15px'>AQI <b class='aq-mono'>{job['actual']:.0f}</b> {ui.tier_badge(at, small=True)} · "
                            f"model off by <b class='aq-mono'>{abs(err):.0f}</b> ({'over' if err > 0 else 'under'}-predicted)"
                            f"{'' if at.label == tier.label else ' · <b>tier differs</b>'}</div>", unsafe_allow_html=True)
            st.write("")
            if job["status"] == "REVIEW_REQUIRED":
                ui.callout("<b>Manual review required</b><br>" + "<br>".join(f"• {r}" for r in job["review_reasons"]), "review")
            else:
                ui.callout("<b>Completed</b> — not near a tier boundary, all inputs present and within the training range.", "ok")

    st.write("")
    left, right = st.columns([1.1, 1], gap="medium")
    with left.container(border=True):
        st.plotly_chart(_shap_chart(job), width="stretch", config=ui.CHART_CONFIG)
        top = sorted(job["shap"].items(), key=lambda kv: -abs(kv[1]))[:3]
        st.markdown("<div class='aq-muted'>Bars show how much each pollutant moved this prediction away from the model's "
                    f"average ({job['shap_base']:.0f}). Biggest drivers: "
                    + ", ".join(f"<b>{k}</b> ({v:+.1f})" for k, v in top) + ".</div>", unsafe_allow_html=True)
    with right.container(border=True):
        st.markdown("#### Health advisory")
        st.markdown(f"<div style='font-size:14.5px;line-height:1.6'>{job['advisory'].split(' Health impact')[0]}</div>", unsafe_allow_html=True)
        st.markdown(f"**General public** — {tier.general_public}")
        st.markdown(f"**Sensitive groups** — {tier.sensitive_groups}")
        st.caption("Advisory text follows the CPCB health statement for the predicted tier. Decision support only — "
                   "not a regulatory notice or medical advice.")
        st.download_button("Download PDF report", report.prediction_report(job), f"aqi_advisory_{job['job_id']}.pdf",
                           "application/pdf", type="primary", icon=":material/picture_as_pdf:")

    st.write("")
    with st.container(border=True):
        st.markdown("#### Readings")
        rows = []
        for p in core.ALL_POLLUTANTS:
            v = job["inputs"].get(p)
            used = job["model_input"].get(p)
            rows.append({"Pollutant": f"{p} — {core.FULL_NAMES[p]}", "Entered": v,
                         "Used by model": None if p not in job["model_input"] else used,
                         "Note": "not a model input" if p not in job["model_input"] else (
                             "filled with median" if p in job["imputed"] else "capped at training max" if v is not None and used < v - 1e-9 else ""),
                         "× safe limit": None if v is None else v / core.SAFE_LIMITS[p]})
        st.dataframe(rows, hide_index=True, width="stretch", column_config={
            "Entered": st.column_config.NumberColumn(format="%.2f"), "Used by model": st.column_config.NumberColumn(format="%.2f"),
            "× safe limit": st.column_config.ProgressColumn(format="%.1f×", min_value=0, max_value=4)})

    with st.expander("Compare with every model"):
        keys = list(job["all_models"])
        fig = go.Figure(go.Bar(y=[core.MODEL_LABELS[k] for k in keys], x=[job["all_models"][k] for k in keys], orientation="h",
                               marker_color=["#3B5BDB" if k.startswith("sy_") else ui.LIB for k in keys],
                               text=[f"{job['all_models'][k]:.0f} · {get_health_tier(job['all_models'][k]).label}" for k in keys],
                               textposition="outside"))
        st.plotly_chart(ui.style_fig(fig, 300).update_xaxes(range=[0, max(job["all_models"].values()) * 1.35]),
                        width="stretch", config=ui.CHART_CONFIG)
        st.caption("Blue = models written from scratch; grey = library models.")
