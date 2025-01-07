import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from algorithms.base import BaseSearcher, EstimatorWrapper


class Classifier(BaseSearcher):
    """
        Class for classification models
        Inherits from EstimatorWrapper
        Includes RandomForestWrapper, LogisticRegressionWrapper, DecisionTreeWrpaper
        Methods:
            fit(X,y) - fits the model
            predict(X) - predicts the target variable
    """
    def __init__(self):
        super().__init__(
            "roc_auc",
            [
                RandomForestWrapper(),
                LogisticRegressionWrapper(),
                DecisionTreeWrapper()
            ]
        )

    @staticmethod
    def measure_importances(X, y):
        X = X.copy()
        X["really_random_variable"] = np.random.rand(X.shape[0])
        feature_names = X.columns
        forest = RandomForestClassifier()
        forest.fit(X, y)
        importances = forest.feature_importances_
        return pd.Series(importances, index=feature_names)
    
class RandomForestWrapper(EstimatorWrapper):
    def __init__(self):
        super().__init__(
            RandomForestClassifier(),
            {
                "n_estimators": randint(100, 501),      
                "min_samples_leaf": randint(1, 251),    
                "max_samples": uniform(0.5, 0.5),        
                "max_features": uniform(1e-6, 1 - 1e-6),
            },
            "RandomForestClassifier",
            10
        )

class LogisticRegressionWrapper(EstimatorWrapper):
    def __init__(self, big_data=False):
        self.big_data = big_data
        super().__init__(
            LogisticRegression(),
            None,
            "LogisticRegression",
            10
        )

    @property
    def param_distributions_(self):
        params = {
                "penalty": ["l2"],
                "C": uniform(0.01, 1000),
                "solver": ['saga'] if self.big_data else ['liblinear'],
                "fit_intercept": [True, False],
                "class_weight": ["balanced", None],
                "l1_ratio": [None],
                "max_iter": [5000] if self.big_data else [1000],
            }
        if "saga" in params["solver"]:
            params["penalty"] = ["elasticnet", "l1", "l2" ,"none"] 
        elif "liblinear" in params["solver"]:
            params["penalty"] = ["l1", "l2"]

        if "elasticnet" in params["penalty"]:
            params["l1_ratio"] = uniform(0.1,0.9)

        print("Using", params["solver"], "solver", end=".")    
        return params


class DecisionTreeWrapper(EstimatorWrapper):
    def __init__(self):
        super().__init__(
            DecisionTreeClassifier(),
            {
                "max_depth": randint(1, 31),
                "min_samples_split": randint(2, 61),
                "criterion": ["gini", "entropy"],
                "min_samples_leaf": randint(1, 61),
            },
            "DecisionTreeClassifier",
            10
        )
