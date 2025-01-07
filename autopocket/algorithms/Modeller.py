import warnings
from algorithms.classification import Classifier
from algorithms.regression import Regressor

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

        with warnings.catch_warnings():
            #warnings.filterwarnings("ignore")
            m.fit(X,y)
            
        return m.best_model_
