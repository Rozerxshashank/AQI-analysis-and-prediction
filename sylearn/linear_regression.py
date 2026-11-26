import numpy as np


class syLinearRegression:
    def __init__(self):
        self.weights = None
        self.intercept_ = None
        self.xtx = None      # X^T X, saved so the model can keep learning later
        self.xty = None      # X^T y
        self.n_seen = 0      # how many rows the model has learned from

    def fit(self, X, y):
        npX = np.array(X)
        npX = np.c_[np.ones(npX.shape[0]), npX]
        npy = np.array(y)
        # self.weights= np.linalg.inv((npX.T@ npX)) @ npX.T @ npy
        mixed = np.linalg.pinv(npX) @ npy

        self.intercept_ = mixed[0]
        self.weights = mixed[1:]

        self.xtx = npX.T @ npX
        self.xty = npX.T @ npy
        self.n_seen = npX.shape[0]
        return self.weights, self.intercept_

    def predict(self, X):
        npX = np.array(X)
        return npX @ self.weights + self.intercept_

    # learn from new data without retraining on the old data
    def partial_fit(self, X, y):
        npX = np.array(X)
        npX = np.c_[np.ones(npX.shape[0]), npX]
        npy = np.array(y)

        if self.xtx is None:
            self.xtx = npX.T @ npX
            self.xty = npX.T @ npy
        else:
            self.xtx = self.xtx + npX.T @ npX
            self.xty = self.xty + npX.T @ npy
        self.n_seen = self.n_seen + npX.shape[0]

        mixed = np.linalg.pinv(self.xtx) @ self.xty
        self.intercept_ = mixed[0]
        self.weights = mixed[1:]
        return self

    # combine two models trained on different data (e.g. two cities)
    def merge(self, other):
        new_model = syLinearRegression()
        new_model.xtx = self.xtx + other.xtx
        new_model.xty = self.xty + other.xty
        new_model.n_seen = self.n_seen + other.n_seen

        mixed = np.linalg.pinv(new_model.xtx) @ new_model.xty
        new_model.intercept_ = mixed[0]
        new_model.weights = mixed[1:]
        return new_model
