# AQI Lab: interactive web frontend

A hand-built web app (FastAPI + HTML/CSS/JavaScript, no Streamlit) for the same models as the Streamlit dashboard.
It puts **sylearn** front and centre.

## Run

From the project folder (models must exist: run `python train.py` once):

```bash
.venv/bin/pip install fastapi uvicorn      # once, if not installed
.venv/bin/python web/server.py
```

Open http://localhost:8000. Start-up takes a few seconds because it replays the sylearn demos and benchmarks.

## Sections

- **Predict**: pollutant sliders (with CPCB safe limits), a city's typical day or a real day from a city the models never saw, an animated AQI gauge, and every model's prediction side by side.
- **sylearn**: watch `partial_fit` learn 2,009 days one at a time against sklearn's `SGDRegressor`, merge any set of city models with `merge()` and compare with training on all their raw data, the actual code, and a sylearn vs scikit-learn table.
- **Inside the models**: the linear regression equation and its terms, the forest's 10 tree votes, and the decision tree with the path your input takes.
- **Model lab**: leaderboard, random-forest benchmark (size, single-prediction speed, training time), actual vs predicted, residuals and error per test city.
- **Cities** and **Data**: AQI history, seasons, missing values and correlations.

All numbers are computed from the saved models and the data when the server starts; nothing is hard-coded.
