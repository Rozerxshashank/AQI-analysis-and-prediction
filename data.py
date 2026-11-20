"""Data loading, cleaning and preprocessing.

Cleaning rules (chosen by a controlled experiment on held-out cities, see README):
  1. Drop rows without AQI (target missing).
  2. Drop rows with AQI > 500 — the CPCB scale tops out at 500, so these are invalid.
  3. CO == 0 is treated as missing (real air never has zero CO; 0 is a missing-value code here).
  4. Outliers are capped at the 99.5th percentile of the *training* data (not removed:
     removing high CO readings made every model worse).
  5. Remaining gaps are filled with the training median.

Split: by city. Six whole cities are held out, so no station has neighbouring days on
both sides of the split (PRD: Data Quality / Leakage).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "Data" / "raw" / "city_day.csv"
MODELS_DIR = ROOT / "models"

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET = "AQI"
MAX_VALID_AQI = 500
CAP_QUANTILE = 0.995
TEST_CITIES = ["Mumbai", "Patna", "Shillong", "Talcher", "Thiruvananthapuram", "Visakhapatnam"]

# CPCB (India) AQI categories: (upper bound, name, colour)
AQI_BANDS = [
    (50, "Good", "#2E9E5B"),
    (100, "Satisfactory", "#8DBF3F"),
    (200, "Moderate", "#E8B62C"),
    (300, "Poor", "#E8812C"),
    (400, "Very Poor", "#D9443A"),
    (float("inf"), "Severe", "#8E2C48"),
]


def aqi_band(aqi):
    for upper, name, color in AQI_BANDS:
        if aqi <= upper:
            return name, color


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Rules 1–3. Keeps all columns and the original index."""
    out = df.dropna(subset=[TARGET])
    out = out[out[TARGET] <= MAX_VALID_AQI].copy()
    out.loc[out["CO"] == 0, "CO"] = np.nan
    return out


class Preprocessor:
    """Rules 3–5, fitted on training rows only and reused unchanged at inference."""

    def __init__(self, features=FEATURES, cap_quantile=CAP_QUANTILE):
        self.features = list(features)
        self.cap_quantile = cap_quantile
        self.caps = None
        self.imputer = None

    def _frame(self, X):
        X = pd.DataFrame(X, columns=self.features) if not isinstance(X, pd.DataFrame) else X[self.features]
        X = X.astype(float).copy()
        if "CO" in X:
            X.loc[X["CO"] == 0, "CO"] = np.nan
        return X

    def fit(self, X):
        X = self._frame(X)
        self.caps = X.quantile(self.cap_quantile).to_dict()
        self.imputer = SimpleImputer(strategy="median").fit(X.clip(upper=pd.Series(self.caps), axis=1))
        return self

    def transform(self, X):
        X = self._frame(X).clip(upper=pd.Series(self.caps), axis=1)
        return self.imputer.transform(X)

    @property
    def statistics_(self):
        return dict(zip(self.features, self.imputer.statistics_))


def train_test_frames(df=None):
    """Cleaned train/test frames (all columns, original index), split by city."""
    df = clean_dataset(load_dataset() if df is None else df)
    test = df["City"].isin(TEST_CITIES)
    return df[~test], df[test]


def train_test_data(df=None):
    train, test = train_test_frames(df)
    pre = Preprocessor().fit(train[FEATURES])
    return (pre.transform(train[FEATURES]), pre.transform(test[FEATURES]),
            train[TARGET].to_numpy(), test[TARGET].to_numpy(), pre)
