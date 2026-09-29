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
