from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

import numpy as np

from algorithms.base import EstimatorWrapper

class Regressor():
    """
        Class for regression models
        Inherits from EstimatorWrapper
        Includes DecisionTreeWrapper, RandomForestWrapper, LinearRegressionWrapper, LassoWrapper, ElasticNetWrapper
        Methods:
            fit(X,y) - fits the model
            predict(X) - predicts the target variable
    """
    def __init__(self):
        self.best_model_ = None
        self.best_score_ = None
        self.best_params = None
        self.metric_ = "neg_mean_squared_error"
        self.estimators_ = [
            DecisionTreeWrapper(),
            RandomForestWrapper(),
            LinearRegressionWrapper(),
            LassoWrapper(),
            ElasticNetWrapper()
        ]
        self.n_estimators_ = len(self.estimators_)
        self.results_ = {}

    def fit(self,X,y):
        check_X_y(X,y)

        self.classes_ = unique_labels(y)
        self.best_score_ = np.inf
        print("Fitting", self.n_estimators_ ,"models")

        for i,wrapper in enumerate(self.estimators_):
            print(i+1,"/",self.n_estimators_," | Fitting:", wrapper.name_, end=". ")
            rs = RandomizedSearchCV(wrapper.estimator_, 
                                      wrapper.param_distributions_,
                                      cv=5,
                                      scoring=self.metric_,
                                      random_state=420,
                                      n_iter=wrapper.n_iter_
                                    )
            rs.fit(X,y)
            print("Best score:", rs.best_score_, self.metric_)

            self.results_[wrapper.name_] = {
                "estimator": rs.best_estimator_,
                "score": rs.best_score_,
                "params": rs.best_params_
            }

            if rs.best_score_ < self.best_score_:
                self.best_score_ = rs.best_score_
                self.best_model_ = rs.best_estimator_
                self.best_params = rs.best_params_

        return self
    
    def predict(self, X):
        check_is_fitted(self)
        return self.best_model_.predict(X)
    
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