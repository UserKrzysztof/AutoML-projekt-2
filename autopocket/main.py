import pandas as pd

from autopocket.preprocessing.Preprocessor import  Preprocessor
from autopocket.algorithms.Modeller import Modeller
from autopocket.postprocessing.Postprocessor import Postprocessor



class AutoPocketor():
    def __init__(self):
        """
        Porządny init.



        Bardzo długi.
        """
        pass

    def doJob(self, path, target):
        """
        Porządny doJob.
        Bardzo krótki.
        """
        X, y, ml_type = Preprocessor().preprocess(path, target="is_claim")
        print("Preprocessing done")
        best_model = Modeller().model(X, y, ml_type)
        print("Modelling done")
        print(best_model.__class__.__name__)
        X = pd.DataFrame(X)
        Postprocessor().postprocess(best_model, X, y, ml_type)

if __name__ == "__main__":
    AutoPocketor().doJob("example/data/car_insurance.csv", "is_claim")









