"""My from-scratch ML models (no sklearn inside)."""
from sylearn.linear_regression import syLinearRegression
from sylearn.decision_tree import Node, syDecisionTreeRegressor
from sylearn.random_forest import syRandomForestRegressor

__version__ = "0.2.0"
__all__ = ["syLinearRegression", "Node", "syDecisionTreeRegressor", "syRandomForestRegressor"]
