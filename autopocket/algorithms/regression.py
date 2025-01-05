from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from algorithms.base import BaseSearcher, EstimatorWrapper

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
            "neg_mean_squared_error",
            [
                DecisionTreeWrapper(),
                RandomForestWrapper(),
                LinearRegressionWrapper(),
                LassoWrapper(),
                ElasticNetWrapper()
            ]
        )

class DecisionTreeWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "max_depth": randint(1, 31),
            "min_samples_split": randint(2, 61),
            "criterion": ["gini", "entropy"],
            "min_samples_leaf": randint(1, 61),
        }
        super().__init__(DecisionTreeRegressor(), param_distributions, "DecisionTreeRegressor", 10)

class RandomForestWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "n_estimators": randint(100, 501),      
            "min_samples_leaf": randint(1, 251),    
            "max_samples": uniform(0.5, 0.5),        
            "max_features": uniform(1e-6, 1 - 1e-6),
        }
        super().__init__(RandomForestRegressor(), param_distributions, "RandomForestRegressor", 10)

class LinearRegressionWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "copy_X": [True]
        }
        super().__init__(LinearRegression(), param_distributions, "LinearRegression", 10)

class LassoWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "alpha": uniform(0.1, 1.0),
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "copy_X": [True]
        }
        super().__init__(Lasso(), param_distributions, "Lasso", 10)

class ElasticNetWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "alpha": uniform(0.1, 1.0),
            "l1_ratio": uniform(0.1, 1.0),
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "copy_X": [True]
        }
        super().__init__(ElasticNet(), param_distributions, "ElasticNet", 10)