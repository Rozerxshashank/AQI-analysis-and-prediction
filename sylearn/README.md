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

## Good to know

- The decision tree tries every possible split, so training gets slow on large datasets (tens of thousands of rows).
- The models do not handle missing values. Fill or drop NaN values first.

## License

MIT
