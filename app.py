"""AQI Prediction & Analysis — Streamlit dashboard.

Run:  streamlit run app.py      (train once first: python train.py)
"""
import streamlit as st

st.set_page_config(page_title="AQI Prediction & Analysis", page_icon="🌫️", layout="wide",
                   initial_sidebar_state="auto")

from dashboard import core, nav, ui  # noqa: E402
from dashboard.views import (errors, evaluation, export, home, pollution_map, predict,  # noqa: E402
                             result)

ui.inject_css()

if core.load_artifacts() is None:
    st.error("No trained model artifacts found in `models/`. Run `python train.py` first, then reload this page.")
    st.stop()

# PRD → UX Requirements → Information Architecture (7 screens)
nav.PAGES.update({
    "home": st.Page(home.render, title="Home / Overview", icon=":material/home:", url_path="home", default=True),
    "predict": st.Page(predict.render, title="Input & Predict", icon=":material/edit_note:", url_path="predict"),
    "result": st.Page(result.render, title="Prediction Result", icon=":material/insights:", url_path="result"),
    "map": st.Page(pollution_map.render, title="Pollution Map", icon=":material/map:", url_path="map"),
    "evaluation": st.Page(evaluation.render, title="Evaluation", icon=":material/monitoring:", url_path="evaluation"),
    "errors": st.Page(errors.render, title="Error Analysis", icon=":material/troubleshoot:", url_path="errors"),
    "export": st.Page(export.render, title="Export", icon=":material/download:", url_path="export"),
})
P = nav.PAGES

page = st.navigation({
    "": [P["home"]],
    "Predict": [P["predict"], P["result"]],
    "Analyse": [P["map"], P["evaluation"], P["errors"]],
    "Share": [P["export"]],
})

# ---- sidebar: interchangeable regression model (PRD LLD: Strategy pattern)
art = core.load_artifacts()
# The chosen model lives in `model_key`, a plain session value. The dropdown uses its own key
# (`_model_pick`) and is re-seeded from `model_key` on every run: Streamlit can wipe a widget's
# own state when the app switches pages in code (st.switch_page), which used to reset the sidebar.
if "model_key" not in st.session_state:
    st.session_state.model_key = core.DEFAULT_MODEL
st.session_state._model_pick = st.session_state.model_key


def _on_model_change():
    st.session_state.model_key = st.session_state._model_pick


with st.sidebar:
    st.markdown("<div class='aq-eyebrow' style='margin-top:4px'>Regression model</div>", unsafe_allow_html=True)
    st.selectbox("Regression model", list(core.MODEL_LABELS), format_func=core.MODEL_LABELS.get,
                 key="_model_pick", on_change=_on_model_change, label_visibility="collapsed")
    m = art["results"]["models"][st.session_state.model_key]
    st.markdown(f"<div class='aq-muted aq-mono' style='font-size:12px;line-height:1.7'>{art['versions'][st.session_state.model_key]}<br>"
                f"test R² {m['R2']:.3f} · MAE {m['MAE']:.1f} · RMSE {m['RMSE']:.1f}</div>", unsafe_allow_html=True)
    st.caption("Decision support only — not a certified regulatory monitor.")

page.run()
