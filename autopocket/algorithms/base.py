from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class BaseSearcher(BaseEstimator):
    """
        Abstract class for model selection
    """
    def __init__(self, metric, estimators):
        self.best_model_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.metric_ = metric
        self.estimators_ = estimators
        self.n_estimators_ = len(self.estimators_)
        self.results_ = {}
    
    def fit(self,X,y):
        check_X_y(X,y)

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


class EstimatorWrapper(BaseEstimator):
    """
        Abstract class for estimators creation
    """
    def __init__(self, estimator, param_distributions, name, n_iter):
        super().__init__()
        self.estimator_ = estimator
        self.param_distributions_ = param_distributions
        self.name_ = name
        self.n_iter_ = n_iter

    def fit(self, X,y):
        return self.estimator_.fit(X,y)
    
    def predict(self, X):
        return self.estimator_.predict(X)
    
    def predict_proba(self,X,y):
        assert hasattr(self.estimator_, "predict_proba")
        return self.estimator_.predic_proba(X)