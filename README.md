# Air Quality Index (AQI) Prediction and Analysis

A Data Science project that analyzes historical air-quality data and predicts the **Air Quality Index (AQI)** using pollutant measurements.

The project focuses on understanding pollution patterns, preprocessing environmental data, building a regression model, and presenting AQI predictions through an interactive dashboard.

## Project Overview

Air pollution varies across cities and over time. This project uses historical pollutant measurements to build a machine learning system that predicts AQI.

The project covers:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- AQI regression
- AQI health-tier assignment
- Model evaluation
- Data visualization
- Interactive Streamlit dashboard
- Pollution mapping
- Health advisories

## Dataset

The project uses the **Air Quality Data in India (2015–2020)** dataset from Kaggle.

Dataset source:

https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

The current implementation uses:

`city_day.csv`

The dataset contains air-quality observations from multiple Indian cities between **2015 and 2020**.

### Dataset Statistics

- Observations: **29,531**
- Cities: **26**
- Time period: **2015-01-01 to 2020-07-01**
- AQI observations available: **24,850**

## Features

The regression model uses the following pollutant measurements:

| Feature | Description |
|---|---|
| PM2.5 | Fine particulate matter |
| NO2 | Nitrogen dioxide |
| SO2 | Sulfur dioxide |
| CO | Carbon monoxide |
| O3 | Ozone |

### Target Variable

- **AQI** — Air Quality Index

Features with substantial missing data such as PM10, Xylene, Toluene, Benzene, NO, NOx, and NH3 are not used as model inputs.

## Project Pipeline

```text
Raw Air Quality Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Regression Model
        ↓
Predicted AQI
        ↓
Rule-Based AQI Health Tier
        ↓
Health Advisory
        ↓
Streamlit Dashboard

## Running it

```bash
python train.py          # trains the scratch models + sklearn/XGBoost baselines, saves to models/
streamlit run app.py     # opens the dashboard
```

## Dashboard (PRD information architecture)

1. **Home / Overview** — project summary, pipeline, CPCB tier scale, model card, scope boundary
2. **Input & Predict** — manual entry (PM2.5, PM10, NO2, SO2, CO, O3), recorded city/date, batch CSV upload with validation
3. **Prediction Result** — predicted AQI, tier badge, error band, SHAP top contributing pollutants, health advisory, review flag
4. **Pollution Map** — Folium heatmap by city, filterable by pollutant, years and season
5. **Evaluation** — MAE, RMSE, R², actual vs predicted, residual distribution, tier confusion, error cases
6. **Error Analysis** — missed severe events, large residuals, boundary-tier flips, missing-data errors
7. **Export** — PDF health advisory, batch predictions CSV, evaluation report JSON

Each prediction moves through `SUBMITTED → PROCESSING → PREDICTED → TIER_MAPPED → COMPLETED / REVIEW_REQUIRED`
(or `PROCESSING_FAILED` on invalid input). Predictions within ±½ MAE of a tier boundary, with missing inputs,
or outside the training range are flagged `REVIEW_REQUIRED`.

## Project layout

- `sylearn/` — Linear Regression, Decision Tree and Random Forest written from scratch with NumPy
- `data.py` — loading, cleaning, outlier capping, median imputation and the city-based train/test split
- `train.py` — trains every model and records test metrics in `models/results.json`
- `app.py` — Streamlit entry point and navigation
- `dashboard/tiers.py` — rule-based CPCB health-tier mapper (if/elif thresholds) and advisory text
- `dashboard/core.py` — prediction pipeline: validation, imputation, prediction, tiering, SHAP, error band, review flags, batch scoring
- `dashboard/report.py` — PDF report
- `dashboard/views/` — the seven screens
- `notebooks/01_eda.ipynb` — EDA and model experiments

## Data cleaning and its effect

Chosen by a controlled experiment: same six held-out cities, same 5,331 valid test days, only the preprocessing changes.

| Setup | Scratch RF R² | Scratch RF MAE | Scratch LR R² | XGBoost R² |
|---|---|---|---|---|
| No cleaning, 5 pollutants | 0.883 | 23.2 | 0.816 | 0.849 |
| Drop AQI > 500, CO = 0 → missing, cap at train 99.5th pct | 0.888 | 21.8 | 0.848 | 0.847 |
| … and also delete CO > 10 mg/m³ | 0.864 | 25.3 | 0.791 | 0.807 |
| **Cleaning + PM10 (final)** | **0.914** | **19.9** | **0.885** | **0.892** |

- **AQI > 500** (543 rows, mostly Ahmedabad) is off the CPCB scale, so those rows are dropped.
- **CO = 0** (768 rows with valid AQI) is a missing-value code, not a measurement.
- **High CO** is capped, not deleted — deleting it made every model worse, so those readings carry real signal.
- **PM10** is kept despite 37.7% missingness: CPCB AQI is the maximum pollutant sub-index, and PM10 is the
  deciding sub-index on 65% of days that report both particulates.
- **Split by city** (Mumbai, Patna, Shillong, Talcher, Thiruvananthapuram, Visakhapatnam held out). The earlier
  row-order split cut Mumbai in half across train and test.
- **Known limitation:** the March–May 2020 COVID lockdown lowered mean AQI from 149 to 88 versus the same months in 2019.
