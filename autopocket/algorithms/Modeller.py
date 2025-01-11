import warnings

from sklearn.exceptions import ConvergenceWarning
from autopocket.algorithms.classification import Classifier
from autopocket.algorithms.regression import Regressor
from scipy.linalg import LinAlgWarning

class Modeller():
    def __init__(self):
        """
        Porządny init.
        """
        pass

    def model(self, X, y, ml_type):
        """
        Porządny model.
        """
        if ml_type == "BINARY_CLASSIFICATION":
            m = Classifier()
            print("Performing binary classification")
        else:
            m = Regressor()
            print("Performing regression")
        metric = m.get_metric() ####
        estimators = m.get_estimators() ####
        with warnings.catch_warnings():
            warnings.simplefilter('always', LinAlgWarning)
            warnings.simplefilter('always', ConvergenceWarning)
            warnings.showwarning = custom_warning_handler

            m.fit(X, y)
            
        return m.best_model_, metric, estimators ####

shown_warnings = set()

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    if category == UserWarning:
        return
    if category in shown_warnings:
        return
    shown_warnings.add(category)
    if category == LinAlgWarning:
        print("Your data may contain colinear features", end=". ")
        return
    if category == ConvergenceWarning:
        print("Some models did not converge", end=". ")
        return
    print(f"{category.__name__}: {message}", end=". ")
    return