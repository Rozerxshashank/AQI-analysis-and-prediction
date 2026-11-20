import numpy as np


class Node:
    def __init__(self, feature=None,threshold=None,left=None,right=None,prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

class syDecisionTreeRegressor:

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self.build_me_a_tree(X, y)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            predictions.append(self.predict_one(x, self.root))
        return np.array(predictions)

    def build_me_a_tree(self, X, y, depth=0):
        if depth>=self.max_depth or len(y)< self.min_samples_split:
            prediction =np.mean(y)
            return Node(prediction=prediction)

        feature, threshold = self.find_best_split(X, y)

        if feature is None:
            prediction = np.mean(y)
            return Node(prediction=prediction)

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        X_left = X[left_mask]
        X_right = X[right_mask]
        y_left = y[left_mask]
        y_right = y[right_mask]

        left_node = self.build_me_a_tree(X_left, y_left, depth + 1)
        right_node = self.build_me_a_tree(X_right, y_right, depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def find_best_split(self, X, y):
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for i in range(X.shape[1]):
            values = np.unique(X[:, i])
            thresholds = (values[:-1] + values[1:])/2
            #This part is not optimized by me. For now I am goining with this unomptimized version
            for threshold in thresholds:
                left = X[:, i] <= threshold
                right = X[:, i] > threshold

                y_left = y[left]
                y_right = y[right]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                mse_left = self.calculate_mse(y_left)
                mse_right = self.calculate_mse(y_right)

                weighted_mse = (len(y_left)/len(y))*mse_left + (len(y_right)/len(y))*mse_right

                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = i
                    best_threshold = threshold

        return best_feature, best_threshold

    def calculate_mse(self, y):
        mean = np.mean(y)
        return np.mean((y-mean)**2)

    def predict_one(self, x, node):
        if node.prediction is not None:
            return node.prediction

        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)
