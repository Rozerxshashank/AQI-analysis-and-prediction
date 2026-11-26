import numpy as np


class syLinearRegression:
    """Linear Regression from scratch using OLS: w = pinv(X) @ y.

    Besides fit(), it can also learn incrementally and combine models, which
    scikit-learn's LinearRegression cannot do:

    - partial_fit(X, y): learn from a new batch without keeping or re-reading old data.
    - merge(other): combine two models trained on different data (e.g. two cities)
      into the model you would get by training on both datasets together.

    Both are exact, not approximations. OLS only needs two running sums of the data,
    A = Xb^T Xb and b = Xb^T y (Xb is X with a column of ones for the intercept).
    Sums can be updated batch by batch and added across models, then solved again:
    w = pinv(A) @ b. The memory needed is (features + 1)^2 numbers, whatever the data size.
    """

    def __init__(self):
        self.weights = None
        self.intercept_ = None
        self.xtx_ = None              # running sum Xb^T Xb
        self.xty_ = None              # running sum Xb^T y
        self.n_samples_seen_ = 0

    def fit(self, X, y):
        npX = np.array(X)
        npX = np.c_[np.ones(npX.shape[0]), npX]
        npy = np.array(y)
        # self.weights= np.linalg.inv((npX.T@ npX)) @ npX.T @ npy
        mixed = np.linalg.pinv(npX) @ npy

        self.intercept_ = mixed[0]
        self.weights = mixed[1:]
        # keep the running sums too, so partial_fit() / merge() can continue from here
        self.xtx_ = npX.T @ npX
        self.xty_ = npX.T @ npy
        self.n_samples_seen_ = npX.shape[0]
        return self.weights, self.intercept_

    def predict(self, X):
        npX = np.array(X)
        return npX @ self.weights + self.intercept_

    # ------------------------------------------------------------------ incremental learning
    def partial_fit(self, X, y):
        """Update the model with a new batch of rows. Exact: after any number of batches the
        weights equal fit() on all rows seen so far. Returns self."""
        npX = np.array(X, dtype=float)
        npy = np.array(y, dtype=float).ravel()
        if npX.ndim != 2:
            raise ValueError("X must be 2-D: one row per sample, one column per feature.")
        if npX.shape[0] != npy.shape[0]:
            raise ValueError(f"X has {npX.shape[0]} rows but y has {npy.shape[0]} values.")
        if npX.shape[0] == 0:
            return self
        if self.weights is not None and self.xtx_ is None:
            raise ValueError("This model was saved before partial_fit() existed. Call fit() once to rebuild "
                             "its running sums, then continue with partial_fit().")
        Xb = np.c_[np.ones(npX.shape[0]), npX]
        if self.xtx_ is not None and self.xtx_.shape[0] != Xb.shape[1]:
            raise ValueError(f"X has {npX.shape[1]} features but the model was trained with {self.xtx_.shape[0] - 1}.")
        self.xtx_ = Xb.T @ Xb if self.xtx_ is None else self.xtx_ + Xb.T @ Xb
        self.xty_ = Xb.T @ npy if self.xty_ is None else self.xty_ + Xb.T @ npy
        self.n_samples_seen_ += npX.shape[0]
        self._solve()
        return self

    def merge(self, other):
        """Return a new model equal to training on this model's data plus other's data.
        Neither model's raw data is needed, only their running sums."""
        for m in (self, other):
            if m.xtx_ is None:
                raise ValueError("Both models must be trained with fit() or partial_fit() before merging.")
        if self.xtx_.shape != other.xtx_.shape:
            raise ValueError("Models were trained on a different number of features and cannot be merged.")
        merged = syLinearRegression()
        merged.xtx_ = self.xtx_ + other.xtx_
        merged.xty_ = self.xty_ + other.xty_
        merged.n_samples_seen_ = self.n_samples_seen_ + other.n_samples_seen_
        merged._solve()
        return merged

    def _solve(self):
        mixed = np.linalg.pinv(self.xtx_) @ self.xty_
        self.intercept_ = mixed[0]
        self.weights = mixed[1:]
