import numpy as np

from sylearn.decision_tree import syDecisionTreeRegressor


class syRandomForestRegressor:
    """Random Forest from scratch: bagging over my own decision trees."""

    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.trees = []
        for i in range(self.n_trees):
            ids = np.random.choice(len(X), len(X), replace=True)
            X_new = X[ids]
            y_new = y[ids]
            tree = syDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_new, y_new)
            self.trees.append(tree)
        return self

    def predict(self, X):
        results = []
        for tree in self.trees:
            pred = tree.predict(X)
            results.append(pred)
        final = np.mean(results, axis=0)
        return final
