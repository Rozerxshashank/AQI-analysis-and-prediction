"""7. Export — PDF advisory, batch results, evaluation report (PRD BR-009, UC-7)."""
import json
from datetime import datetime

import streamlit as st

from dashboard import core, nav, report, ui
from dashboard.views.errors import classify


def render():
    ui.header("Export", "Take the results with you")
    st.write("")
    job = st.session_state.get("result")
    batch = st.session_state.get("batch")
    a, b, c = st.columns(3, gap="medium")

    with a.container(border=True):
        st.markdown("#### PDF health advisory")
        if job:
            st.markdown(f"Prediction <span class='aq-mono'>{job['job_id']}</span>: AQI <b class='aq-mono'>{job['predicted_aqi']:.0f}</b> "
                        f"{ui.tier_badge(job['tier'], small=True)}", unsafe_allow_html=True)
            st.markdown("<div class='aq-muted'>Includes readings, tier, error band, review flags, SHAP drivers, advisory and scope notice.</div>",
                        unsafe_allow_html=True)
            include = st.checkbox("Include batch CSV summary", value=bool(batch), disabled=not batch)
            pdf = report.prediction_report(job, batch["summary"] if (batch and include) else None)
            st.download_button("Download PDF", pdf, f"aqi_advisory_{job['job_id']}.pdf", "application/pdf",
                               type="primary", icon=":material/picture_as_pdf:", width="stretch")
        else:
            st.markdown("<div class='aq-muted'>No prediction yet.</div>", unsafe_allow_html=True)
            st.page_link(nav.PAGES["predict"], label="Make a prediction", icon=":material/arrow_forward:")

    with b.container(border=True):
        st.markdown("#### Batch predictions (CSV)")
        if batch:
            s = batch["summary"]
            st.markdown(f"<span class='aq-mono'>{batch['filename']}</span> · {s['rows_scored']:,} rows scored · "
                        f"{s['review_required']:,} flagged for review", unsafe_allow_html=True)
            st.markdown(f"<div class='aq-muted'>Model {s['model_version']} · {s['processed_at']:%d %b %Y %H:%M}</div>", unsafe_allow_html=True)
            st.download_button("Download CSV", batch["rows"].to_csv(index=False).encode(),
                               f"aqi_batch_predictions_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv",
                               type="primary", icon=":material/table_view:", width="stretch")
        else:
            st.markdown("<div class='aq-muted'>No batch scored yet.</div>", unsafe_allow_html=True)
            st.page_link(nav.PAGES["predict"], label="Upload a CSV", icon=":material/arrow_forward:")

    with c.container(border=True):
        st.markdown("#### Evaluation report (JSON)")
        art = core.load_artifacts()
        st.markdown("<div class='aq-muted'>MAE, RMSE, R² and failure-class counts for every model, with versions — "
                    "an immutable record of this evaluation run.</div>", unsafe_allow_html=True)
        runs = {}
        for k, mres in art["results"]["models"].items():
            ev = core.evaluation_frame(k)
            runs[k] = {"model": core.MODEL_LABELS[k], "model_version": art["versions"][k], "MAE": mres["MAE"],
                       "RMSE": mres["RMSE"], "R2": mres["R2"], "train_seconds": mres["train_seconds"],
                       "tier_accuracy": float((ev["actual_tier"] == ev["predicted_tier"]).mean()),
                       "failure_classes": {n: int(m.sum()) for n, m in classify(ev, mres["MAE"], mres["RMSE"]).items()}}
        payload = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                   "dataset": "Air Quality Data in India (2015–2020), city_day.csv",
                   "test_set": {"days": len(core.split_frames()[1]), "cities": sorted(core.split_frames()[1]["City"].unique())},
                   "tier_standard": "CPCB National AQI", "models": runs}
        st.download_button("Download JSON", json.dumps(payload, indent=2).encode(), "aqi_evaluation_report.json",
                           "application/json", icon=":material/data_object:", width="stretch")

    st.write("")
    ui.callout(ui.SCOPE_TEXT, "scope")
