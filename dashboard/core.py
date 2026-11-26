"""Prediction pipeline used by every dashboard screen.

SUBMITTED → PROCESSING → PREDICTED → TIER_MAPPED → REVIEW_REQUIRED / COMPLETED
failure: PROCESSING_FAILED   (PRD: Product States)
"""
import json
import time
import uuid
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from dashboard.tiers import TIER_BOUNDARIES, get_health_tier, nearest_boundary, advisory_message
from data import FEATURES, MODELS_DIR, TARGET, load_dataset, train_test_frames

ALL_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]   # PRD input screen fields
UNITS = {"PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³", "SO2": "µg/m³", "CO": "mg/m³", "O3": "µg/m³"}
SAFE_LIMITS = {"PM2.5": 60, "PM10": 100, "NO2": 80, "SO2": 80, "CO": 2, "O3": 100}   # CPCB 24-h (O3: 8-h)
FULL_NAMES = {"PM2.5": "Fine particulate matter", "PM10": "Coarse particulate matter", "NO2": "Nitrogen dioxide",
              "SO2": "Sulphur dioxide", "CO": "Carbon monoxide", "O3": "Ozone"}

MODEL_LABELS = {
    "sy_rf": "Random Forest (from scratch)",
    "sy_dt": "Decision Tree (from scratch)",
    "sy_lr": "Linear Regression (from scratch)",
    "xgb": "XGBoost (library)",
    "sk_rf": "Random Forest (sklearn)",
    "sk_dt": "Decision Tree (sklearn)",
    "sk_lr": "Linear Regression (sklearn)",
}
DEFAULT_MODEL = "sy_rf"

MAX_UPLOAD_MB = 5
MAX_ROWS = 50_000
PLAUSIBLE_MAX = {"PM2.5": 1000, "PM10": 1500, "NO2": 500, "SO2": 500, "CO": 200, "O3": 500}

CITY_COORDS = {
    "Ahmedabad": (23.0225, 72.5714), "Aizawl": (23.7271, 92.7176), "Amaravati": (16.5417, 80.5158),
    "Amritsar": (31.6340, 74.8723), "Bengaluru": (12.9716, 77.5946), "Bhopal": (23.2599, 77.4126),
    "Brajrajnagar": (21.8242, 83.9216), "Chandigarh": (30.7333, 76.7794), "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558), "Delhi": (28.6139, 77.2090), "Ernakulam": (9.9816, 76.2999),
    "Gurugram": (28.4595, 77.0266), "Guwahati": (26.1445, 91.7362), "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873), "Jorapokhar": (23.7042, 86.4131), "Kochi": (9.9312, 76.2673),
    "Kolkata": (22.5726, 88.3639), "Lucknow": (26.8467, 80.9462), "Mumbai": (19.0760, 72.8777),
    "Patna": (25.5941, 85.1376), "Shillong": (25.5788, 91.8933), "Talcher": (20.9509, 85.2168),
    "Thiruvananthapuram": (8.5241, 76.9366), "Visakhapatnam": (17.6868, 83.2185),
}


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    results_path = MODELS_DIR / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        results = json.load(f)
    pre_path = MODELS_DIR / "preprocessor.joblib"
    if not pre_path.exists():
        return None
    trained = (datetime.fromisoformat(results["trained_at"]) if "trained_at" in results
               else datetime.fromtimestamp(results_path.stat().st_mtime))
    models = {k: joblib.load(MODELS_DIR / f"{k}.joblib") for k in results["models"]}
    pre = joblib.load(pre_path)
    versions = {k: f"{k}-v2 · trained {trained:%Y-%m-%d}" for k in results["models"]}
    return {"results": results, "models": models, "pre": pre, "versions": versions}


@st.cache_data(show_spinner="Loading dataset…")
def dataset():
    return load_dataset()


@st.cache_data(show_spinner=False)
def split_frames():
    """Same cleaning and city-based split as train.py."""
    return train_test_frames(dataset())


@st.cache_data(show_spinner=False)
def evaluation_frame(model_key: str) -> pd.DataFrame:
    """Test-set rows with predictions, residuals, tiers and missing-data flags."""
    art = load_artifacts()
    _, test = split_frames()
    out = test[["City", "Date"] + ALL_POLLUTANTS + [TARGET]].copy().reset_index(drop=True)
    out["predicted"] = art["results"]["models"][model_key]["pred"]
    out["residual"] = out[TARGET] - out["predicted"]
    out["abs_error"] = out["residual"].abs()
    out["actual_tier"] = out[TARGET].map(lambda v: get_health_tier(v).label)
    out["predicted_tier"] = out["predicted"].map(lambda v: get_health_tier(v).label)
    out["missing_inputs"] = test[FEATURES].isna().sum(axis=1).to_numpy()
    out["season"] = out["Date"].dt.month.map(season_of)
    return out


def season_of(month: int) -> str:
    # Indian meteorological seasons (IMD)
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Pre-monsoon"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"


@st.cache_resource(show_spinner=False)
def explainer(model_key: str):
    """SHAP explainer (exact, 5 features) with a fixed training-set background sample."""
    art = load_artifacts()
    train, _ = split_frames()
    rng = np.random.default_rng(42)
    bg = art["pre"].transform(train[FEATURES])[rng.choice(len(train), 100, replace=False)]
    model = art["models"][model_key]
    f = lambda X: np.asarray(model.predict(np.asarray(X)), dtype=float).ravel()
    return shap.Explainer(f, shap.maskers.Independent(bg, max_samples=100), algorithm="exact",
                          feature_names=FEATURES)


@st.cache_data(show_spinner=False)
def residual_band(model_key: str):
    """Empirical 80% error band from held-out residuals (10th–90th percentile)."""
    ev = evaluation_frame(model_key)
    return float(np.percentile(ev["residual"], 10)), float(np.percentile(ev["residual"], 90))


@st.cache_data(show_spinner=False)
def training_ranges():
    return dict(load_artifacts()["pre"].caps)


def review_margin(model_key: str) -> float:
    """Predictions within ±(half the test MAE) of a tier boundary are borderline."""
    return round(load_artifacts()["results"]["models"][model_key]["MAE"] / 2, 1)


# ------------------------------------------------------------------ validation (TR-001)
def validate_reading(values: dict) -> tuple[dict, list[str]]:
    """Return cleaned values (None = missing) and a list of human-readable errors."""
    clean, errors = {}, []
    for p in ALL_POLLUTANTS:
        v = values.get(p)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            clean[p] = None
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{p} must be a number.")
            continue
        if v < 0:
            errors.append(f"{p} cannot be negative (got {v:g}).")
        elif v > PLAUSIBLE_MAX[p]:
            errors.append(f"{p} = {v:g} {UNITS[p]} is outside the plausible range (0–{PLAUSIBLE_MAX[p]}).")
        clean[p] = v
    # CO = 0 counts as "not measured" (same rule as training), so it doesn't count as an entered value.
    if all(clean.get(f) is None or (f == "CO" and clean.get(f) == 0) for f in FEATURES):
        errors.append("Enter at least one of the model's pollutants: " + ", ".join(FEATURES)
                      + " (CO = 0 counts as not measured).")
    return clean, errors


# ------------------------------------------------------------------ single prediction
def predict_reading(values: dict, model_key: str, city=None, date=None, actual=None, on_state=None) -> dict:
    """Run the full pipeline for one reading. Never raises: failures return PROCESSING_FAILED."""
    job = {"job_id": "job_" + uuid.uuid4().hex[:8], "submitted_at": datetime.now(), "model_key": model_key,
           "city": city, "date": date, "actual": actual, "inputs": dict(values), "states": []}

    def state(name):
        job["status"] = name
        job["states"].append(name)
        if on_state:
            on_state(name)

    t0 = time.perf_counter()
    try:
        state("SUBMITTED")
        art = load_artifacts()
        clean, errors = validate_reading(values)
        if errors:
            job["errors"] = errors
            state("PROCESSING_FAILED")
            return job
        state("PROCESSING")
        pre = art["pre"]
        missing = [f for f in FEATURES if clean[f] is None or (f == "CO" and clean[f] == 0)]
        row = pd.DataFrame([{f: (np.nan if clean[f] is None else clean[f]) for f in FEATURES}])
        x = pre.transform(row)
        job["model_input"] = dict(zip(FEATURES, map(float, x[0])))
        job["imputed"] = missing

        model = art["models"][model_key]
        pred = float(np.asarray(model.predict(x)).ravel()[0])
        job["predicted_aqi"] = pred
        job["model_version"] = art["versions"][model_key]
        state("PREDICTED")

        tier = get_health_tier(pred)
        job["tier"] = tier
        job["advisory"] = advisory_message(pred, city)
        lo, hi = residual_band(model_key)
        job["interval"] = (max(0.0, pred + lo), pred + hi)
        state("TIER_MAPPED")

        sv = explainer(model_key)(x)
        job["shap"] = dict(zip(FEATURES, map(float, sv.values[0])))
        job["shap_base"] = float(np.ravel(sv.base_values)[0])

        if hasattr(model, "trees"):            # my scratch forest: spread of tree votes
            votes = [float(t.predict(x)[0]) for t in model.trees]
            job["tree_spread"] = float(np.std(votes))
        elif hasattr(model, "estimators_"):    # sklearn forest
            votes = [float(e.predict(x)[0]) for e in model.estimators_]
            job["tree_spread"] = float(np.std(votes))

        reasons = []
        boundary, dist = nearest_boundary(pred)
        margin = review_margin(model_key)
        if dist <= margin:
            reasons.append(f"Borderline: {pred:.0f} is within ±{margin:g} of the {boundary}/{boundary + 1} tier boundary "
                           f"(half this model's test MAE), so the tier could flip.")
        if missing:
            reasons.append(f"Missing input{'s' if len(missing) > 1 else ''} ({', '.join(missing)}) filled with the training median.")
        ranges = training_ranges()
        beyond = [f for f in FEATURES if clean[f] is not None and clean[f] > ranges[f]]
        if beyond:
            reasons.append(f"{', '.join(beyond)} above the 99.5th percentile of training data — capped at "
                           + ", ".join(f"{ranges[f]:.1f}" for f in beyond) + " before predicting.")
        job["review_reasons"] = reasons
        job["all_models"] = {k: float(np.asarray(m.predict(x)).ravel()[0]) for k, m in art["models"].items()}
        state("REVIEW_REQUIRED" if reasons else "COMPLETED")
    except Exception as e:  # pragma: no cover - surfaced in the UI, never a stack trace
        job["errors"] = [f"Prediction failed ({type(e).__name__}). Check the model artifacts and try again."]
        state("PROCESSING_FAILED")
    job["latency_ms"] = (time.perf_counter() - t0) * 1000
    return job


# ------------------------------------------------------------------ batch CSV (BR-001, US-1)
COLUMN_ALIASES = {"pm25": "PM2.5", "pm2.5": "PM2.5", "pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO2",
                  "so2": "SO2", "co": "CO", "o3": "O3", "city": "City", "date": "Date", "aqi": "AQI"}


def read_csv_upload(uploaded) -> tuple[pd.DataFrame | None, list[str]]:
    """Validate file type/size and schema. Returns (frame, errors)."""
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        return None, [f"File is {uploaded.size / 1e6:.1f} MB — the limit is {MAX_UPLOAD_MB} MB."]
    if not uploaded.name.lower().endswith(".csv"):
        return None, ["Only .csv files are accepted."]
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        return None, ["The file could not be parsed as CSV. Check that it is comma-separated text."]
    if df.empty:
        return None, ["The file has no data rows."]
    if len(df) > MAX_ROWS:
        return None, [f"{len(df):,} rows is over the {MAX_ROWS:,}-row limit. Split the file and upload in parts."]
    df = df.rename(columns={c: COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip()) for c in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]   # e.g. both "PM2.5" and "pm25": keep the first
    present = [f for f in FEATURES if f in df.columns]
    if not present:
        return None, [f"No pollutant columns found. Expected at least one of: {', '.join(FEATURES)} "
                      f"(found: {', '.join(map(str, df.columns[:10]))})."]
    return df, []


def predict_batch(df: pd.DataFrame, model_key: str, progress=None) -> tuple[pd.DataFrame, dict]:
    art = load_artifacts()
    model, pre = art["models"][model_key], art["pre"]
    work = df.copy()
    for f in FEATURES:
        work[f] = pd.to_numeric(work[f], errors="coerce") if f in work.columns else np.nan
    usable = work[FEATURES].notna().any(axis=1)
    negative = (work[FEATURES] < 0).any(axis=1)
    implausible = (work[FEATURES] > pd.Series({f: PLAUSIBLE_MAX[f] for f in FEATURES})).any(axis=1)  # also catches inf
    keep = usable & ~negative & ~implausible
    skipped = {"no pollutant values": int((~usable).sum()), "negative values": int((usable & negative).sum()),
               "implausible values": int((usable & ~negative & implausible).sum())}

    rows = work[keep].copy()
    if rows.empty:  # nothing usable: report it instead of crashing
        return rows, {"rows_total": len(df), "rows_scored": 0, "skipped": skipped, "review_required": 0,
                      "tiers": {}, "model_version": art["versions"][model_key], "processed_at": datetime.now()}
    X = pre.transform(rows[FEATURES])
    preds = np.empty(len(rows))
    chunk = 500
    for i in range(0, len(rows), chunk):
        preds[i:i + chunk] = np.asarray(model.predict(X[i:i + chunk])).ravel()
        if progress:
            progress(min(1.0, (i + chunk) / max(1, len(rows))))
    margin = review_margin(model_key)
    rows["predicted_aqi"] = preds.round(1)
    rows["tier"] = [get_health_tier(p).label for p in preds]
    rows["imputed_fields"] = [", ".join(f for f in FEATURES if pd.isna(r[f]) or (f == "CO" and r[f] == 0))
                              for _, r in rows[FEATURES].iterrows()]
    rows["review_required"] = [nearest_boundary(p)[1] <= margin or bool(m) for p, m in zip(preds, rows["imputed_fields"])]
    rows["model_version"] = art["versions"][model_key]
    summary = {"rows_total": len(df), "rows_scored": len(rows), "skipped": skipped,
               "review_required": int(rows["review_required"].sum()),
               "tiers": rows["tier"].value_counts().to_dict(), "model_version": art["versions"][model_key],
               "processed_at": datetime.now()}
    return rows, summary


def template_csv() -> bytes:
    return pd.DataFrame([
        {"City": "Delhi", "Date": "2020-01-15", "PM2.5": 180.5, "PM10": 290.0, "NO2": 55.2, "SO2": 15.1, "CO": 2.1, "O3": 30.4},
        {"City": "Bengaluru", "Date": "2020-01-15", "PM2.5": 42.0, "PM10": 88.0, "NO2": 28.7, "SO2": 6.3, "CO": 0.9, "O3": 35.2},
    ]).to_csv(index=False).encode()
