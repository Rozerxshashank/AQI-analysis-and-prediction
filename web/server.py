"""AQI Lab — an interactive web frontend (FastAPI + hand-built HTML/JS), alongside the Streamlit app.

Run from the project folder:  .venv/bin/python web/server.py   →  http://localhost:8000
(Train once first with `python train.py`.)
"""
import json
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import AQI_BANDS, FEATURES, MODELS_DIR, TARGET, TEST_CITIES, aqi_band, load_dataset, train_test_frames  # noqa: E402
from sylearn import syLinearRegression  # noqa: E402
from dashboard.tiers import get_health_tier, nearest_boundary  # noqa: E402  (same CPCB rules + advice as the Streamlit app)

STATIC = Path(__file__).parent / "static"
UNITS = {"PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³", "SO2": "µg/m³", "CO": "mg/m³", "O3": "µg/m³"}
SAFE_LIMITS = {"PM2.5": 60, "PM10": 100, "NO2": 80, "SO2": 80, "CO": 2, "O3": 100}   # CPCB 24-hour limits
FULL_NAMES = {"PM2.5": "Fine particulates", "PM10": "Coarse particulates", "NO2": "Nitrogen dioxide",
              "SO2": "Sulphur dioxide", "CO": "Carbon monoxide", "O3": "Ozone"}
HEALTH = {
    "Good": "Minimal impact.",
    "Satisfactory": "Minor breathing discomfort to sensitive people.",
    "Moderate": "Breathing discomfort to people with lung disease such as asthma, and heart disease.",
    "Poor": "Breathing discomfort to most people on prolonged exposure.",
    "Very Poor": "Respiratory illness on prolonged exposure.",
    "Severe": "Affects healthy people and seriously impacts those with existing diseases.",
}
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
LABELS = {"sy_lr": "Linear Regression", "sk_lr": "Linear Regression", "sy_dt": "Decision Tree", "sk_dt": "Decision Tree",
          "sy_rf": "Random Forest", "sk_rf": "Random Forest", "xgb": "XGBoost"}

# ------------------------------------------------------------------ load everything once
print("Loading models and data…", flush=True)
with open(MODELS_DIR / "results.json") as f:
    RESULTS = json.load(f)
MODELS = {k: joblib.load(MODELS_DIR / f"{k}.joblib") for k in RESULTS["models"]}
PRE = joblib.load(MODELS_DIR / "preprocessor.joblib")

DF = load_dataset()
TRAIN, TEST = train_test_frames(DF)
X_TRAIN, Y_TRAIN = PRE.transform(TRAIN[FEATURES]), TRAIN[TARGET].to_numpy()
X_TEST, Y_TEST = PRE.transform(TEST[FEATURES]), TEST[TARGET].to_numpy()
CLEAN = DF.copy()                                   # for city views: same cleaning rules as training
CLEAN.loc[CLEAN[TARGET] > 500, TARGET] = np.nan
CLEAN.loc[CLEAN["CO"] == 0, "CO"] = np.nan


def _r(v, n=2):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), n)


def _serialise_tree(tree, X, y):
    """Flatten my syDecisionTreeRegressor into a node list (with training counts) for drawing."""
    nodes = []

    def walk(node, Xn, yn, depth, parent):
        idx = len(nodes)
        entry = {"id": idx, "depth": depth, "parent": parent, "n": int(len(yn)),
                 "value": float(np.mean(yn)) if len(yn) else 0.0}
        nodes.append(entry)
        if node.prediction is not None:
            entry.update(leaf=True, value=float(node.prediction))
            return idx
        mask = Xn[:, node.feature] <= node.threshold
        entry.update(leaf=False, feature=FEATURES[node.feature], threshold=float(node.threshold))
        entry["left"] = walk(node.left, Xn[mask], yn[mask], depth + 1, idx)
        entry["right"] = walk(node.right, Xn[~mask], yn[~mask], depth + 1, idx)
        return idx

    walk(tree.root, X, y, 0, None)
    return nodes


TREE_NODES = _serialise_tree(MODELS["sy_dt"], X_TRAIN, Y_TRAIN)


def _tree_path(x):
    path, node = [], TREE_NODES[0]
    while True:
        path.append(node["id"])
        if node["leaf"]:
            return path
        node = TREE_NODES[node["left"] if x[FEATURES.index(node["feature"])] <= node["threshold"] else node["right"]]


# ------------------------------------------------------------------ sylearn demos (computed once at start-up)
def _build_stream():
    """Feed the training data day by day: sylearn partial_fit (exact) vs sklearn SGDRegressor (approximate)."""
    order = TRAIN.sort_values("Date")
    days = [(d, PRE.transform(g[FEATURES]), g[TARGET].to_numpy()) for d, g in order.groupby("Date")]
    every = max(1, len(days) // 120)
    sy, sgd, scaler = syLinearRegression(), SGDRegressor(random_state=0), StandardScaler()
    points, t_sy, rows = [], 0.0, 0
    for i, (d, Xd, yd) in enumerate(days):
        t = time.perf_counter(); sy.partial_fit(Xd, yd); t_sy += time.perf_counter() - t
        scaler.partial_fit(Xd); sgd.partial_fit(scaler.transform(Xd), yd)
        rows += len(yd)
        if i % every == 0 or i == len(days) - 1:
            points.append({"day": i + 1, "date": str(d.date()), "rows": rows,
                           "weights": dict(zip(FEATURES, map(float, sy.weights))), "intercept": float(sy.intercept_),
                           "r2_sylearn": float(r2_score(Y_TEST, sy.predict(X_TEST))),
                           "r2_sgd": float(r2_score(Y_TEST, sgd.predict(scaler.transform(X_TEST))))})
    full = syLinearRegression(); full.fit(X_TRAIN, Y_TRAIN)
    # sklearn LinearRegression has no partial_fit: time retraining on all history every day
    from sklearn.linear_model import LinearRegression
    t = time.perf_counter(); seen_X, seen_y = [], []
    for _, Xd, yd in days:
        seen_X.append(Xd); seen_y.append(yd)
        LinearRegression().fit(np.vstack(seen_X), np.concatenate(seen_y))
    t_sk = time.perf_counter() - t
    raw = SGDRegressor(random_state=0)
    for _, Xd, yd in days:
        raw.partial_fit(Xd, yd)
    return {
        "days": len(days), "rows": rows, "points": points,
        "full": {"weights": dict(zip(FEATURES, map(float, full.weights))), "intercept": float(full.intercept_),
                 "r2": float(r2_score(Y_TEST, full.predict(X_TEST)))},
        "max_diff": float(max(np.abs(sy.weights - full.weights).max(), abs(sy.intercept_ - full.intercept_))),
        "sylearn_ms_total": t_sy * 1000, "sylearn_us_per_day": t_sy / len(days) * 1e6,
        "sklearn_retrain_ms_total": t_sk * 1000, "stored_numbers": int(sy.xtx.size + sy.xty.size),
        "sgd_default_r2": float(r2_score(Y_TEST, raw.predict(X_TEST))),
    }


CITY_MODELS = {c: syLinearRegression().partial_fit(PRE.transform(g[FEATURES]), g[TARGET].to_numpy())
               for c, g in TRAIN.groupby("City")}
CITY_ROWS = TRAIN.groupby("City").size().to_dict()
STREAM = _build_stream()


def _benchmark():
    """sylearn forest vs sklearn forest: same settings and sklearn's defaults."""
    def one_row_ms(m, reps):
        row = X_TEST[:1]; m.predict(row)
        t = time.perf_counter()
        for _ in range(reps):
            m.predict(row)
        return (time.perf_counter() - t) / reps * 1000

    t = time.perf_counter()
    same = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42).fit(X_TRAIN, Y_TRAIN)
    same_train = time.perf_counter() - t
    out = []
    for name, m, train_s, reps in [("sylearn forest · 10 trees, depth 5", MODELS["sy_rf"], RESULTS["models"]["sy_rf"]["train_seconds"], 200),
                                   ("sklearn forest · same settings", same, same_train, 200),
                                   ("sklearn forest · default (100 deep trees)", MODELS["sk_rf"], RESULTS["models"]["sk_rf"]["train_seconds"], 20)]:
        p = m.predict(X_TEST)
        out.append({"name": name, "mine": name.startswith("sylearn"), "r2": float(r2_score(Y_TEST, p)),
                    "mae": float(mean_absolute_error(Y_TEST, p)), "size_kb": len(pickle.dumps(m)) / 1000,
                    "predict_ms": one_row_ms(m, reps), "train_s": float(train_s)})
    return out


BENCH = _benchmark()


# ------------------------------------------------------------------ static metadata (sent once)
def build_meta():
    feats = [{"key": f, "name": FULL_NAMES[f], "unit": UNITS[f], "safe": SAFE_LIMITS[f],
              "max": _r(np.nanpercentile(DF[f], 99.5), 1), "median": _r(PRE.statistics_[f]),
              "step": 0.01 if f == "CO" else 0.1} for f in FEATURES]
    city_medians = CLEAN.groupby("City")[FEATURES].median()
    city_means = CLEAN.groupby("City")[FEATURES + [TARGET]].mean()
    city_aqi = CLEAN.groupby("City")[TARGET].agg(["mean", "count"]).dropna()
    cities = [{"name": c, "aqi": _r(city_aqi.loc[c, "mean"], 1), "days": int(city_aqi.loc[c, "count"]),
               "test": c in TEST_CITIES, "medians": {f: _r(city_medians.loc[c, f]) for f in FEATURES},
               "coord": list(CITY_COORDS[c]), "means": {m: _r(city_means.loc[c, m]) for m in FEATURES + [TARGET]}}
              for c in city_aqi.sort_values("mean", ascending=False).index]
    models = {k: {kk: v for kk, v in m.items() if kk != "pred"} for k, m in RESULTS["models"].items()}
    for mine, lib in [("sy_lr", "sk_lr"), ("sy_dt", "sk_dt"), ("sy_rf", "sk_rf")]:
        diff = np.abs(np.array(RESULTS["models"][mine]["pred"]) - np.array(RESULTS["models"][lib]["pred"]))
        models[mine]["diff_vs_lib"] = float(diff.mean())
    lr, sk = MODELS["sy_lr"], MODELS["sk_lr"]
    missing = DF.drop(columns=["City", "Date", "AQI_Bucket"], errors="ignore").isna().mean() * 100
    return {
        "features": feats,
        "bands": [{"max": None if u == float("inf") else u, "name": n, "color": c, "health": HEALTH[n]} for u, n, c in AQI_BANDS],
        "cities": cities, "models": models,
        "linear": {"mine": {"intercept": float(lr.intercept_), "weights": dict(zip(FEATURES, map(float, lr.weights)))},
                   "sklearn": {"intercept": float(sk.intercept_), "weights": dict(zip(FEATURES, map(float, sk.coef_)))}},
        "tree": TREE_NODES,
        "stream": STREAM,
        "merge": {"cities": [{"name": c, "rows": int(CITY_ROWS[c])} for c in sorted(CITY_MODELS)]},
        "bench": BENCH,
        "dataset": {"rows": len(DF), "cities": int(DF["City"].nunique()), "start": str(DF["Date"].min().date()),
                    "end": str(DF["Date"].max().date()), "train_rows": len(TRAIN), "test_days": len(TEST),
                    "test_cities": TEST_CITIES, "missing": {k: _r(v, 1) for k, v in missing.sort_values(ascending=False).items()},
                    "corr": DF[FEATURES + [TARGET]].corr().round(3).values.tolist()},
    }


META = build_meta()
print("Ready.", flush=True)

# ------------------------------------------------------------------ API
app = FastAPI(title="AQI Lab")


class Inputs(BaseModel):
    values: dict[str, float | None]


class Cities(BaseModel):
    cities: list[str]


@app.get("/api/meta")
def meta():
    return META


@app.post("/api/predict")
def predict(inp: Inputs):
    vals = {f: inp.values.get(f) for f in FEATURES}
    if all(v is None or (f == "CO" and v == 0) for f, v in vals.items()):
        raise HTTPException(400, "Enter at least one pollutant.")
    if any(v is not None and (v < 0 or not np.isfinite(v)) for v in vals.values()):
        raise HTTPException(400, "Pollutant values must be non-negative numbers.")
    x = PRE.transform(pd.DataFrame([{f: (np.nan if v is None else v) for f, v in vals.items()}]))
    preds = {k: float(np.asarray(m.predict(x)).ravel()[0]) for k, m in MODELS.items()}
    lr, forest = MODELS["sy_lr"], MODELS["sy_rf"]
    return {
        "predictions": preds,
        "model_input": dict(zip(FEATURES, map(float, x[0]))),
        "explain": {
            "linear": {"intercept": float(lr.intercept_),
                       "contributions": {f: float(w * x[0, i]) for i, (f, w) in enumerate(zip(FEATURES, lr.weights))}},
            "tree_path": _tree_path(x[0]),
            "forest_votes": [float(t.predict(x)[0]) for t in forest.trees],
        },
    }


@app.get("/api/sample")
def sample():
    """A random real day from the held-out cities (never used in training)."""
    row = TEST.sample(1).iloc[0]
    return {"city": row["City"], "date": str(row["Date"].date()), "actual": float(row[TARGET]),
            "values": {f: (None if pd.isna(row[f]) else round(float(row[f]), 2)) for f in FEATURES},
            "imputed": [f for f in FEATURES if pd.isna(row[f])]}


@app.post("/api/merge")
def merge(req: Cities):
    chosen = [c for c in req.cities if c in CITY_MODELS]
    if not chosen:
        raise HTTPException(400, "Pick at least one city.")
    merged = None
    for c in chosen:
        merged = CITY_MODELS[c] if merged is None else merged.merge(CITY_MODELS[c])
    rows = TRAIN[TRAIN["City"].isin(chosen)]
    full = syLinearRegression(); full.fit(PRE.transform(rows[FEATURES]), rows[TARGET].to_numpy())
    diff = max(np.abs(merged.weights - full.weights).max(), abs(merged.intercept_ - full.intercept_))
    return {"cities": chosen, "rows": int(merged.n_seen),
            "merged": {"weights": dict(zip(FEATURES, map(float, merged.weights))), "intercept": float(merged.intercept_)},
            "full": {"weights": dict(zip(FEATURES, map(float, full.weights))), "intercept": float(full.intercept_)},
            "max_diff": float(diff), "r2_test": float(r2_score(Y_TEST, merged.predict(X_TEST))),
            "xtx": np.log10(np.abs(merged.xtx) + 1).round(2).tolist(), "shared_per_city": int(merged.xtx.size + merged.xty.size)}


@app.get("/api/eval/{key}")
def evaluate(key: str):
    if key not in RESULTS["models"]:
        raise HTTPException(404, "unknown model")
    return {"actual": RESULTS["y_test"], "pred": RESULTS["models"][key]["pred"], "cities": TEST["City"].tolist()}


@app.get("/api/city/{name}")
def city(name: str):
    c = CLEAN[CLEAN["City"] == name]
    if c.empty:
        raise HTTPException(404, "unknown city")
    s = c.set_index("Date")[TARGET]
    monthly = s.resample("MS").mean()
    grid = s.groupby([s.index.year, s.index.month]).mean()
    years = sorted({y for y, _ in grid.index})
    bands = c[TARGET].dropna().map(lambda v: aqi_band(v)[0]).value_counts()
    worst = c.loc[c[TARGET].idxmax()] if s.notna().any() else None
    return {"name": name,
            "monthly": {"dates": monthly.index.strftime("%Y-%m").tolist(), "aqi": [_r(v, 1) for v in monthly]},
            "heatmap": {"years": years, "values": [[_r(grid.get((y, m)), 0) for m in range(1, 13)] for y in years]},
            "band_days": {b["name"]: int(bands.get(b["name"], 0)) for b in META["bands"]},
            "worst": None if worst is None else {"date": str(worst["Date"].date()), "aqi": float(worst[TARGET])}}


# ------------------------------------------------------------------ AQI Predictor app (/predict)
MODEL_NAMES = {"sy_rf": "Random Forest · sylearn", "sy_dt": "Decision Tree · sylearn", "sy_lr": "Linear Regression · sylearn",
               "sk_rf": "Random Forest · scikit-learn", "sk_dt": "Decision Tree · scikit-learn", "sk_lr": "Linear Regression · scikit-learn",
               "xgb": "XGBoost"}
# 80% error band per model, from its residuals (actual − predicted) on the held-out cities
RESID_BAND = {k: tuple(np.percentile(np.array(RESULTS["y_test"]) - np.array(m["pred"]), [10, 90])) for k, m in RESULTS["models"].items()}
PLAUSIBLE_MAX = {"PM2.5": 1000, "PM10": 1500, "NO2": 500, "SO2": 500, "CO": 200, "O3": 500}


def _tier_dict(v):
    t = get_health_tier(v)
    return {"label": t.label, "range": t.range_text, "health": t.health_impact, "public": t.general_public, "sensitive": t.sensitive_groups}


def _check(vals):
    errors = []
    for f, v in vals.items():
        if v is None:
            continue
        if not np.isfinite(v) or v < 0:
            errors.append(f"{f} must be a number of 0 or more.")
        elif v > PLAUSIBLE_MAX[f]:
            errors.append(f"{f} = {v:g} is outside the plausible range (0–{PLAUSIBLE_MAX[f]}).")
    if all(v is None or (f == "CO" and v == 0) for f, v in vals.items()):
        errors.append("Enter at least one pollutant (CO = 0 counts as not measured).")
    return errors


class Assess(BaseModel):
    values: dict[str, float | None]
    model: str = "sy_rf"


@app.post("/api/assess")
def assess(req: Assess):
    """Full single-reading assessment: AQI, tier, likely range, review flags, advice, drivers, all models."""
    if req.model not in MODELS:
        raise HTTPException(400, "Unknown model.")
    vals = {f: req.values.get(f) for f in FEATURES}
    errors = _check(vals)
    if errors:
        raise HTTPException(400, " ".join(errors))
    x = PRE.transform(pd.DataFrame([{f: (np.nan if v is None else v) for f, v in vals.items()}]))
    preds = {k: float(np.asarray(m.predict(x)).ravel()[0]) for k, m in MODELS.items()}
    v = preds[req.model]
    lo, hi = RESID_BAND[req.model]
    mae = RESULTS["models"][req.model]["MAE"]
    reasons = []
    b, dist = nearest_boundary(v)
    if dist <= mae / 2:
        reasons.append(f"{v:.0f} is within ±{mae / 2:.0f} of the {b}/{b + 1} boundary between two health tiers, so the tier could flip.")
    missing = [f for f, val in vals.items() if val is None or (f == "CO" and val == 0)]
    if missing:
        reasons.append(f"{', '.join(missing)} not measured, so a typical value was used.")
    capped = [f for f, val in vals.items() if val is not None and val > PRE.caps[f]]
    if capped:
        reasons.append(f"{', '.join(capped)} is higher than almost anything in the training data; the model is extrapolating.")
    lr = MODELS["sy_lr"]
    return {
        "model": req.model, "model_name": MODEL_NAMES[req.model], "aqi": v, "tier": _tier_dict(v),
        "range": [max(0.0, v + lo), v + hi], "review": reasons, "mae": mae, "r2": RESULTS["models"][req.model]["R2"],
        "all": [{"key": k, "name": MODEL_NAMES[k], "aqi": p, "tier": get_health_tier(p).label} for k, p in preds.items()],
        "contributions": {f: float(w * x[0, i]) for i, (f, w) in enumerate(zip(FEATURES, lr.weights))}, "intercept": float(lr.intercept_),
        "used": dict(zip(FEATURES, map(float, x[0]))),
    }


class Batch(BaseModel):
    rows: list[dict[str, float | str | None]]
    model: str = "sy_rf"


@app.post("/api/batch")
def batch(req: Batch):
    """Score many readings at once. Rows with no usable pollutant, negatives or impossible values are skipped."""
    if req.model not in MODELS:
        raise HTTPException(400, "Unknown model.")
    if not req.rows:
        raise HTTPException(400, "The file has no rows.")
    if len(req.rows) > 20000:
        raise HTTPException(400, f"{len(req.rows):,} rows is over the 20,000-row limit. Split the file.")
    df = pd.DataFrame(req.rows)
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce") if f in df.columns else np.nan
    X = df[FEATURES]
    usable = X.notna().any(axis=1) & ~(X < 0).any(axis=1) & ~(X > pd.Series(PLAUSIBLE_MAX)).any(axis=1)
    out = [None] * len(df)
    idx = np.flatnonzero(usable.to_numpy())
    if len(idx):
        preds = np.asarray(MODELS[req.model].predict(PRE.transform(X.iloc[idx]))).ravel()
        mae = RESULTS["models"][req.model]["MAE"]
        for j, p in zip(idx, preds):
            miss = [f for f in FEATURES if pd.isna(X.iloc[j][f]) or (f == "CO" and X.iloc[j][f] == 0)]
            out[j] = {"aqi": round(float(p), 1), "tier": get_health_tier(p).label, "review": bool(miss or nearest_boundary(p)[1] <= mae / 2), "missing": miss}
    tiers = pd.Series([o["tier"] for o in out if o]).value_counts().to_dict()
    return {"results": out, "scored": int(len(idx)), "skipped": int(len(df) - len(idx)), "tiers": tiers,
            "review": int(sum(1 for o in out if o and o["review"])), "model_name": MODEL_NAMES[req.model]}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.api_route("/predict", methods=["GET", "HEAD"])
def predictor_page():
    return FileResponse(STATIC / "predict.html")


@app.api_route("/", methods=["GET", "HEAD"])
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
