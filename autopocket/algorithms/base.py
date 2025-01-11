from abc import abstractmethod
from time import strftime, gmtime
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import json
import os

from autopocket.algorithms.utils import ResultsReader

class BaseSearcher(BaseEstimator):
    """
        Abstract class for model selection
        Fields:
            best_model_ - the best model found
            best_score_ - the best score found
            best_params_ - the best parameters found
            metric_ - the metric to optimize
            estimators_ - list of EstimatorWrappers (models to fit)
            n_estimators_ - number of models to fit
            results_ - dictionary with results of the search
        Methods:
            fit(X,y) - fits the models and finds the best one
            predict(X) - predicts the target variable
            save_results() - saves the results to json files
            read_results() - reads the results from json files
            create_model_from_json(wrapper_name) - creates a model from the best parameters found in the search
            measure_importances(X,y) - abstract method for measuring importances
            drop_unimportant_features(X, importances) - drops unimportant features from the dataset
    """
    def __init__(self, metric, estimators, dummy_estimator = None, dummy_strategy = None, additional_estimators=None):
        self.best_model_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.metric_ = metric
        self.estimators_ = estimators
        self.n_estimators_ = len(self.estimators_)
        self.results_ = {}
        now = strftime("%Y%m%d_%H%M%S", gmtime())
        self.results_dir = os.path.join(os.getcwd(), 'results', f'algorithms_results_{now}')
        self.dummy_estimator = dummy_estimator
        self.dummy_strategy = dummy_strategy
        self.additional_estimators = additional_estimators
    
    def fit(self,X,y):
        """
            Finds the best model using GridSearchCV or RandomizedSearchCV
            Parameters:
                X - input features
                y - target variable
        """
        check_X_y(X,y)
        X = X.copy()
        y = y.copy()

        if self.dummy_estimator is not None:
            print("Fitting dummy estimator")
            self.dummy_estimator.fit(X,y)
            scorer = get_scorer(self.metric_)
            print(f"Dummy score (strategy: {self.dummy_strategy}):", scorer(self.dummy_estimator, X, y), self.metric_)

        # Depracted in latest version  
        #print("Measuring importances")
        #importances = self.__class__.measure_importances(X,y)
        #top_3_features = importances.nlargest(3)
        #print("Top 3 features by importance:")
        #print(top_3_features)

        self.best_score_ = -np.inf
        print("Fitting", self.n_estimators_ ,"models")

        self.fit_on(self.estimators_, self.n_estimators_, X,y)

        if self.additional_estimators is not None:
            print(f"Fitting {len(self.additional_estimators)} additional estimator{'s' if len(self.additional_estimators) > 1 else ''}")
            self.fit_on(self.additional_estimators, len(self.additional_estimators), X,y)

        self.save_results()
        return self
    
    def fit_on(self, estimators, n_estimators, X,y):
        for i,wrapper in enumerate(estimators):
            print(i+1,"/",n_estimators," | Fitting:", wrapper.name_, end=". ")

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

    @abstractmethod
    def get_baseline_prediction(self, y):
        """
            Abstract method for getting the baseline prediction
            Should be implemented in the child class
        """
        pass

    def predict(self, X):
        """
            Predicts the target variable using the best model
            Parameters:
                X - input features
        """
        check_is_fitted(self)
        return self.best_model_.predict(X)
    
    def save_results(self):
        """
            Save the results to json files
        """
        os.makedirs(self.results_dir, exist_ok=True)
        results_dir = self.results_dir

        for wrapper_name, result in self.results_.items():
            result_to_save = {
                "score": result["score"],
                "params": result["params"]
            }
            with open(os.path.join(results_dir, f'{wrapper_name}_results.json'), 'w') as f:
                json.dump(result_to_save, f)
        print(f"Saving results to results/algorithms_results")
    
    def read_results(self):
        """
            Read the results from json files
        """
        reader = ResultsReader(self.results_dir)
        return reader.results
    
    def create_model_from_json(self, wrapper_name: str):
        """
            Create a model from the best parameters found in the search
            Not in use yet, but leaved an option for future improvements
        """
        reader = ResultsReader(self.results_dir)
        return reader.create_model_from_json(wrapper_name)
    
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
    def drop_unimportant_features(X, importances: pd.Series):
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
        """
            Getter for param_distributions
        """
        return self.param_distributions
    
    def fit(self, X,y):
        """
            Fits the estimator
        """
        return self.estimator_.fit(X,y)
    
    def predict(self, X):
        """
            Predicts the target variable
        """
        return self.estimator_.predict(X)
    
    def predict_proba(self,X,y):
        """
            Predicts the probabilities of the target variable
        """
        assert hasattr(self.estimator_, "predict_proba")
        return self.estimator_.predict_proba(X)
    
def create_wrapper(estimator, param_distributions, name, n_iter):
    """
        Creates an EstimatorWrapper
        Parameters:
            estimator - the estimator to wrap
            param_distributions - the parameters distributions for RandomizedSearchCV
            name - the name of the estimator
            n_iter - the number of iterations for RandomizedSearchCV
    """
    return EstimatorWrapper(estimator, param_distributions, name, n_iter)

    
