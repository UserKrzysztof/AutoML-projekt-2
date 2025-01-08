from abc import abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import json
import os

class BaseSearcher(BaseEstimator):
    """
        Abstract class for model selection
    """
    results_dir = os.path.join(os.getcwd(), 'results', 'algorithms_results')

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
        X = X.copy()
        y = y.copy()

        print("Measuring importances")
        importances = self.__class__.measure_importances(X,y)
        top_3_features = importances.nlargest(3)
        print("Top 3 features by importance:")
        print(top_3_features)
        
        # print("Dropping unimportant features")
        # X = self.__class__.drop_unimportant_features(X, importances)
        
        # assert X.shape[1] > 0, "No features left after dropping unimportant features. Please check your data"

        self.best_score_ = -np.inf
        print("Fitting", self.n_estimators_ ,"models")

        for i,wrapper in enumerate(self.estimators_):
            print(i+1,"/",self.n_estimators_," | Fitting:", wrapper.name_, end=". ")

            if hasattr(wrapper, "big_data"):
                wrapper.big_data = X.shape[0] > 6000

            if wrapper.n_iter_ is None:
                rs = GridSearchCV(wrapper.estimator_,
                                    wrapper.param_distributions_,
                                    cv=5,
                                    scoring=self.metric_
                                    )
            else:
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

        self.save_results()
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.best_model_.predict(X)
    
    def save_results(self):
        """
            Save the results to json files
        """

        os.makedirs(os.path.join(os.getcwd(), 'results', 'algorithms_results'), exist_ok=True)
        results_dir = os.path.join(os.getcwd(), 'results', 'algorithms_results')

        for wrapper_name, result in self.results_.items():
            print(f"Saving results for {wrapper_name} to results/algorithms_results/{wrapper_name}_results.json")
            result_to_save = {
                "score": result["score"],
                "params": result["params"]
            }
            with open(os.path.join(results_dir, f'{wrapper_name}_results.json'), 'w') as f:
                json.dump(result_to_save, f)
    
    def read_results(self):
        results_dir = self.__class__.results_dir
        results = {}
        for wrapper_name in self.estimators_:
            with open(os.path.join(results_dir, f'{wrapper_name}_results.json'), 'r') as f:
                results[wrapper_name] = json.load(f)
        return results
    
    def create_model_from_json(self, wrapper_name):
        """
            Create a model from the best parameters found in the search
            Not in use yet, but leaved an option for future improvements
        """
        results = self.read_results()
        best_params = results[wrapper_name]["params"]
        estimator = None
        for wrapper in self.estimators_:
            if wrapper.name_ == wrapper_name:
                estimator = wrapper.estimator_
            break
        if estimator is None:
            raise ValueError(f"No estimator found with name {wrapper_name}")
        model = estimator.set_params(**best_params)
        return model
    
    @staticmethod
    @abstractmethod
    def measure_importances(X,y):
        """
            Abstract method for measuring importances
            Should return a pandas Series with feature importances
            Should add a really_random_variable to the dataset
            Should be implemented in the child class
        """
        pass

    @staticmethod
    def drop_unimportant_features(X, importances):
        really_random_importance = importances.get('really_random_variable', None)
        if really_random_importance is not None:
            columns_to_drop = importances[importances < really_random_importance].index.tolist()
            X.drop(columns=columns_to_drop, inplace=True)
            if len(columns_to_drop) > 5:
                print(f"Dropped columns: {columns_to_drop[:5]} ...")
            else:
                print(f"Dropped columns: {columns_to_drop}")
        important_features = [col for col in X.columns if col not in columns_to_drop]

        print("Saving important features to algorithms_results/important_features.json")

        results_dir = BaseSearcher.results_dir
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'important_features.json'), 'w') as f:
            json.dump(important_features, f)
        return X

class EstimatorWrapper(BaseEstimator):
    """
        Abstract class for estimators creation
    """
    def __init__(self, estimator, param_distributions, name, n_iter):
        super().__init__()
        self.estimator_ = estimator
        self.param_distributions = param_distributions
        self.name_ = name
        self.n_iter_ = n_iter

    @property
    def param_distributions_(self):
        return self.param_distributions
    
    def fit(self, X,y):
        return self.estimator_.fit(X,y)
    
    def predict(self, X):
        return self.estimator_.predict(X)
    
    def predict_proba(self,X,y):
        assert hasattr(self.estimator_, "predict_proba")
        return self.estimator_.predict_proba(X)