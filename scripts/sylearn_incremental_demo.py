"""sylearn v0.2 demo: exact incremental learning and model merging on the AQI data.

Run from the project folder:  python scripts/sylearn_incremental_demo.py

Scenario: AQI readings arrive day by day from 20 cities.
  1. Daily updates  — sylearn learns each new day with partial_fit(); sklearn's LinearRegression
                      has no partial_fit, so it must retrain on all history every day.
  2. City merging   — each city trains its own model on its own data; merge() combines them.
                      Raw readings never leave the city.
  3. sklearn's incremental linear regressor, SGDRegressor, for comparison (approximate: gradient steps).
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import FEATURES, TARGET, Preprocessor, train_test_frames  # noqa: E402
from sylearn import syLinearRegression  # noqa: E402


def line(title):
    print(f"\n{title}\n" + "-" * len(title))


train, test = train_test_frames()
pre = Preprocessor().fit(train[FEATURES])
train = train.sort_values("Date")
X_test, y_test = pre.transform(test[FEATURES]), test[TARGET].to_numpy()
days = [(d, pre.transform(g[FEATURES]), g[TARGET].to_numpy()) for d, g in train.groupby("Date")]
print(f"Training data: {len(train):,} readings from {train['City'].nunique()} cities over {len(days):,} days.")
print(f"Test data: {len(test):,} readings from 6 cities never seen in training.")

# reference: ordinary one-shot training on everything
X_all, y_all = pre.transform(train[FEATURES]), train[TARGET].to_numpy()
full = syLinearRegression(); full.fit(X_all, y_all)

line("1. Daily updates")
inc = syLinearRegression()
t = time.perf_counter()
for _, X_day, y_day in days:
    inc.partial_fit(X_day, y_day)
sy_time = time.perf_counter() - t

t = time.perf_counter()
seen_X, seen_y = [], []
for _, X_day, y_day in days:                      # sklearn: append today's data, retrain on all history
    seen_X.append(X_day); seen_y.append(y_day)
    sk = LinearRegression().fit(np.vstack(seen_X), np.concatenate(seen_y))
sk_time = time.perf_counter() - t

diff = max(np.max(np.abs(inc.weights - full.weights)), abs(inc.intercept_ - full.intercept_))
print(f"sylearn partial_fit, one day at a time : {sy_time * 1000:7.0f} ms total, "
      f"{sy_time / len(days) * 1e6:5.0f} µs per day, keeps {inc.xtx_.size + inc.xty_.size} numbers")
print(f"sklearn LinearRegression, retrain daily: {sk_time * 1000:7.0f} ms total, "
      f"keeps all {len(y_all):,} rows x {len(FEATURES)} features")
print(f"Largest difference from one-shot training: {diff:.1e}  (exact, up to rounding)")
print(f"Test R²: sylearn incremental {r2_score(y_test, inc.predict(X_test)):.4f} | "
      f"one-shot {r2_score(y_test, full.predict(X_test)):.4f}")

line("2. City merging (raw data stays in each city)")
city_models = {c: syLinearRegression().partial_fit(pre.transform(g[FEATURES]), g[TARGET].to_numpy())
               for c, g in train.groupby("City")}
merged = None
for m in city_models.values():
    merged = m if merged is None else merged.merge(m)
diff = max(np.max(np.abs(merged.weights - full.weights)), abs(merged.intercept_ - full.intercept_))
print(f"{len(city_models)} city models merged; each city shared only {merged.xtx_.size + merged.xty_.size} numbers, "
      "no readings.")
print(f"Largest difference from training on all cities' data together: {diff:.1e}")
print(f"Test R²: merged {r2_score(y_test, merged.predict(X_test)):.4f} | "
      f"one-shot {r2_score(y_test, full.predict(X_test)):.4f}")

line("3. sklearn's incremental option: SGDRegressor.partial_fit (approximate)")
sgd_raw = SGDRegressor(random_state=0)
for _, X_day, y_day in days:
    sgd_raw.partial_fit(X_day, y_day)
scaler, sgd = StandardScaler(), SGDRegressor(random_state=0)
for _, X_day, y_day in days:                      # the fair version: scale features as they arrive
    scaler.partial_fit(X_day)
    sgd.partial_fit(scaler.transform(X_day), y_day)
r2_raw = r2_score(y_test, sgd_raw.predict(X_test))
print("SGDRegressor, default settings       : " + (f"test R² {r2_raw:.4f}" if r2_raw > -1 else
      f"diverged (test R² = {r2_raw:.1e}); it needs feature scaling and tuning first"))
print(f"SGDRegressor + incremental scaling   : test R² {r2_score(y_test, sgd.predict(scaler.transform(X_test))):.4f}")
print(f"sylearn partial_fit (no tuning)      : test R² {r2_score(y_test, inc.predict(X_test)):.4f}")
