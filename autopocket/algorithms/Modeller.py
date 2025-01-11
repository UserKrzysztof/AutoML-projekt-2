import warnings

from sklearn.exceptions import ConvergenceWarning
from autopocket.algorithms.classification import Classifier
from autopocket.algorithms.regression import Regressor
from scipy.linalg import LinAlgWarning

class Modeller():
    def __init__(self, additional_estimators=None):
        """
        Porządny init.
        """
        self.additional_estimators = additional_estimators
        pass

    def model(self, X, y, ml_type):
        """
        Porządny model.
        """
        if ml_type == "BINARY_CLASSIFICATION":  
            m = Classifier(additional_estimators=self.additional_estimators)
            print("Performing binary classification")
        else:
            m = Regressor(additional_estimators=self.additional_estimators)
            print("Performing regression")

        with warnings.catch_warnings():
            warnings.simplefilter('always', LinAlgWarning)
            warnings.simplefilter('always', ConvergenceWarning)
            warnings.showwarning = custom_warning_handler
            m.fit(X, y)
            
        return m.best_model_, m.results_dir ####

shown_warnings = set()

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    if category == UserWarning:
        return
    if category in shown_warnings:
        return
    shown_warnings.add(category)
    if category == LinAlgWarning:
        print(message, end=". ")
        return
    if category == ConvergenceWarning:
        print("Some models did not converge", end=". ")
        return
    print(f"{category.__name__}: {message}", end=". ")
    return