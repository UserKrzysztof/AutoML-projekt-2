import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from autopocket.algorithms.base import BaseSearcher, EstimatorWrapper


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
                DecisionTreeWrapper(),
                RidgeClassifierWrapper()
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
            50
        )

class LogisticRegressionWrapper(EstimatorWrapper):
    def __init__(self, big_data=False):
        self.big_data = big_data
        super().__init__(
            Pipeline(
                [
                    ("scaler",StandardScaler()),
                    ("model",LogisticRegression())
                ]
            ),
            None,
            "LogisticRegression",
            20
        )

    @property
    def param_distributions_(self):
        params = {
                "model__penalty": ["l2"],
                "model__C": [1e-8,1e-6,1e-4,1e-3,0.01,0.1, 1,10,100,500,1000,5000,10000],
                "model__solver": ['saga'] if self.big_data else ['lbfgs'],
                "model__fit_intercept": [True, False],
                "model__class_weight": ["balanced", None],
                "model__l1_ratio": [None],
                "model__max_iter": [800],
            }
        if "saga" in params["model__solver"]:
            params["model__penalty"] = ["elasticnet", "l1", "l2" ,None] 
        elif "lbfgs" in params["model__solver"]:
            params["model__penalty"] = ["l2", None]

        if "elasticnet" in params["model__penalty"]:
            params["model__l1_ratio"] = uniform(0.1,0.9)

        print("Using", params["model__solver"], "solver", end=".")    
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
            100
        )

class RidgeClassifierWrapper(EstimatorWrapper):
    def __init__(self):
        super().__init__(
            RidgeClassifier(),
            {
                "alpha": uniform(0, 1),
                "fit_intercept": [True, False],
                "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                "max_iter": [800],
                "class_weight": ["balanced", None]
            },
            "RidgeClassifier",
            100
        )
