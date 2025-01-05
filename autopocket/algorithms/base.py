from sklearn.base import BaseEstimator


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