from scipy.stats import randint, uniform
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from algorithms.base import EstimatorWrapper


class Classifier(BaseEstimator):
    def __init__(self):
        self.best_model_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.metric_ = "roc_auc"
        self.estimators_ = [
            RandomForestWrapper(),
            LogisticRegressionWrapper(),
            DecisionTreeWrpaper()
        ]
        self.n_estimators_ = len(self.estimators_)
        self.results_ = {}
    
    def fit(self,X,y):
        check_X_y(X,y)

        self.classes_ = unique_labels(y)
        self.best_score_ = -np.inf
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

            if rs.best_score_ > self.best_score_:
                self.best_score_ = rs.best_score_
                self.best_model_ = rs.best_estimator_
                self.best_params_ = rs.best_params_

        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.best_model_.predict(X)
    

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
    def __init__(self):
        super().__init__(
            LogisticRegression(),
            {
                "penalty": ["l1","l2", "elasticnet"],
                "C": [0.001,0.01,0.1,1,10,100,1000],
                "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                "fit_intercept": [True, False],
            },
            "LogisticRegression",
            10
        )

class DecisionTreeWrpaper(EstimatorWrapper):
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
