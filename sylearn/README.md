# sylearn

Regression models written from scratch with only NumPy. No scikit-learn inside.

- `syLinearRegression`: ordinary least squares, solved with the pseudo-inverse
- `syDecisionTreeRegressor`: a regression tree that picks splits by mean squared error
- `syRandomForestRegressor`: bagging (bootstrap samples) over `syDecisionTreeRegressor`

They were built to learn how these models work inside. On an air-quality (AQI) dataset they reach the same R² as scikit-learn's models with the same settings.

## Install

```bash
pip install sylearn
```

## Use

```python
import numpy as np
from sylearn import syLinearRegression, syDecisionTreeRegressor, syRandomForestRegressor

X = np.random.rand(200, 3)
y = 4 * X[:, 0] - 2 * X[:, 1] + np.random.normal(0, 0.1, 200)

lr = syLinearRegression()
weights, intercept = lr.fit(X, y)   # fit returns the learned weights and intercept
print(lr.predict(X[:5]))

tree = syDecisionTreeRegressor(max_depth=5, min_samples_split=2).fit(X, y)
print(tree.predict(X[:5]))

np.random.seed(42)                  # the forest uses np.random, so seed it for repeatable results
forest = syRandomForestRegressor(n_trees=10, max_depth=5).fit(X, y)
print(forest.predict(X[:5]))
```

`X` can be a NumPy array, a list of lists or a pandas DataFrame. `y` is a 1-D array.

## Learn as data arrives, and merge models (new in 0.2)

scikit-learn's `LinearRegression` can only be trained on all the data at once. `syLinearRegression` can also:

- **`partial_fit(X, y)`**: learn from a new batch (a new day, a new file) without keeping or re-reading old data.
- **`merge(other)`**: combine two models trained on different data into the model you would get by training on both together. The raw data never has to be moved.

Both are **exact**: the weights equal a normal `fit()` on all the data. Linear regression only needs two running
sums of the data (XᵀX and Xᵀy), which can be updated batch by batch and added across models. Memory stays at
(features + 1)² numbers however much data you have.

```python
daily = syLinearRegression()
for X_day, y_day in stream_of_days:          # e.g. one batch of readings per day
    daily.partial_fit(X_day, y_day)          # no retraining, no stored history

delhi = syLinearRegression().partial_fit(X_delhi, y_delhi)
mumbai = syLinearRegression().partial_fit(X_mumbai, y_mumbai)
combined = delhi.merge(mumbai)               # same as training on both cities' data together
```

On the AQI data (18,976 readings, 20 cities, 2,009 days) learning each day takes about 16 µs and matches one-shot
training to within 1e-11. scikit-learn's incremental regressors (`SGDRegressor`, `PassiveAggressiveRegressor`,
`MLPRegressor`) learn by gradient steps, so they are approximate and need tuning: `SGDRegressor` reached R² 0.880
with feature scaling and diverged without it; `syLinearRegression` reaches 0.885 with no tuning.
`partial_fit` and `merge` are available for linear regression only.

## Good to know

- The decision tree tries every possible split, so training gets slow on large datasets (tens of thousands of rows).
- The models do not handle missing values. Fill or drop NaN values first.

## License

MIT
