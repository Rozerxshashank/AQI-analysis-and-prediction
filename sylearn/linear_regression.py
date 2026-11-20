import numpy as np


class syLinearRegression:
    """Linear Regression from scratch using OLS: w = pinv(X) @ y."""

    def __init__(self):
        self.weights = None
        self.intercept_ = None

    def fit(self, X, y):
        npX = np.array(X)
        npX = np.c_[np.ones(npX.shape[0]), npX]
        npy = np.array(y)
        # self.weights= np.linalg.inv((npX.T@ npX)) @ npX.T @ npy
        mixed = np.linalg.pinv(npX) @ npy

        self.intercept_ = mixed[0]
        self.weights = mixed[1:]
        return self.weights, self.intercept_

    def predict(self, X):
        npX = np.array(X)
        return npX @ self.weights + self.intercept_
