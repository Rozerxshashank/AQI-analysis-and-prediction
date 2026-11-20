"""Train every model (mine + sklearn/xgboost) and save them to models/.

Run once:  python train.py
The app only loads what this script saves, so it starts instantly.
"""
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from datetime import datetime

from data import FEATURES, MODELS_DIR, TEST_CITIES, clean_dataset, load_dataset, train_test_data
from sylearn import syDecisionTreeRegressor, syLinearRegression, syRandomForestRegressor

# key -> (display name, family, built by, model)
MODELS = {
    "sy_lr": ("Linear Regression", "Linear Regression", "scratch", syLinearRegression()),
    "sk_lr": ("Linear Regression", "Linear Regression", "sklearn", LinearRegression()),
    "sy_dt": ("Decision Tree", "Decision Tree", "scratch", syDecisionTreeRegressor(max_depth=5)),
    "sk_dt": ("Decision Tree", "Decision Tree", "sklearn", DecisionTreeRegressor(max_depth=5)),
    "sy_rf": ("Random Forest", "Random Forest", "scratch", syRandomForestRegressor(n_trees=10, max_depth=5)),
    "sk_rf": ("Random Forest", "Random Forest", "sklearn", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    "xgb": ("XGBoost", "XGBoost", "xgboost", XGBRegressor(n_estimators=100, random_state=42)),
}


def main():
    np.random.seed(42)
    MODELS_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, pre = train_test_data()
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    (MODELS_DIR / "imputer.joblib").unlink(missing_ok=True)  # replaced by preprocessor.joblib

    raw = load_dataset()
    with_aqi = raw.dropna(subset=["AQI"])
    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "features": FEATURES,
        "test_cities": TEST_CITIES,
        "cleaning": {
            "raw_rows": len(raw),
            "rows_without_aqi": int(raw["AQI"].isna().sum()),
            "rows_aqi_above_500": int((with_aqi["AQI"] > 500).sum()),
            "co_zero_set_missing": int((with_aqi.loc[with_aqi["AQI"] <= 500, "CO"] == 0).sum()),
            "model_rows": len(clean_dataset(raw)),
            "outlier_caps": {k: round(float(v), 2) for k, v in pre.caps.items()},
        },
        "train_rows": len(y_train),
        "y_test": y_test.tolist(),
        "models": {},
    }
    for key, (name, family, built_by, model) in MODELS.items():
        start = time.time()
        model.fit(X_train, y_train)
        seconds = time.time() - start
        pred = model.predict(X_test)
        joblib.dump(model, MODELS_DIR / f"{key}.joblib")
        report["models"][key] = {
            "name": name,
            "family": family,
            "built_by": built_by,
            "train_seconds": round(seconds, 2),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": float(r2_score(y_test, pred)),
            "pred": np.asarray(pred, dtype=float).round(3).tolist(),
        }
        print(f"{key:6s} {built_by:8s} {name:18s} R2={report['models'][key]['R2']:.4f}  ({seconds:.1f}s)", flush=True)

    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(report, f)
    print("Saved models + results.json to", MODELS_DIR)


if __name__ == "__main__":
    main()
