import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from .base import BaseSearcher, EstimatorWrapper

class Regressor(BaseSearcher):
    """
        Class for regression models
        Inherits from EstimatorWrapper
        Includes DecisionTreeWrapper, RandomForestWrapper, LinearRegressionWrapper, LassoWrapper, ElasticNetWrapper
        Methods:
            fit(X,y) - fits the model
            predict(X) - predicts the target variable
    """
    def __init__(self):
        super().__init__(
            "neg_mean_squared_log_error",
            [
                DecisionTreeWrapper(),
                RandomForestWrapper(),
                LinearRegressionWrapper(),
                LassoWrapper(),
                ElasticNetWrapper()
            ]
        )

    @staticmethod
    def measure_importances(X, y):
        X = X.copy()
        X["really_random_variable"] = np.random.rand(X.shape[0])
        feature_names = X.columns
        forest = RandomForestRegressor()
        forest.fit(X, y)
        importances = forest.feature_importances_
        return pd.Series(importances, index=feature_names)

class DecisionTreeWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "max_depth": randint(1, 31),
            "min_samples_split": randint(2, 61),
            "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            "min_samples_leaf": randint(1, 61),
        }
        super().__init__(DecisionTreeRegressor(), param_distributions, "DecisionTreeRegressor", 100)

class RandomForestWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "n_estimators": randint(100, 501),      
            "min_samples_leaf": randint(1, 251),    
            "max_samples": uniform(0.5, 0.5),        
            "max_features": uniform(1e-6, 1 - 1e-6),
        }
        super().__init__(RandomForestRegressor(), param_distributions, "RandomForestRegressor", 50)

class LinearRegressionWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "fit_intercept": [True, False],
            "copy_X": [True]
        }
        super().__init__(LinearRegression(), param_distributions, "LinearRegression", None) # None means GridSearchCV

class LassoWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "alpha": uniform(0.1, 100),
            "fit_intercept": [True, False],
            "copy_X": [True]
        }
        super().__init__(Lasso(), param_distributions, "Lasso", 100)

class ElasticNetWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "alpha": uniform(0.1, 100),
            "l1_ratio": uniform(0.1, 0.9),
            "fit_intercept": [True, False],
            "copy_X": [True]
        }
        super().__init__(ElasticNet(), param_distributions, "ElasticNet", 100)