import os
import pandas as pd

from preprocessing.Preprocessor import  Preprocessor
from algorithms.Modeller import Modeller
from postprocessing.Postprocessor import Postprocessor



class AutoPocketor():
    def __init__(self):
        """
        Porządny init.



        Bardzo długi.
        """
        pass

    def doJob(self, *args):
        """
        Porządny doJob.
        Bardzo krótki.
        """
        X, y, ml_type = Preprocessor().preprocess(path=os.path.join(".","example","data","car_insurance.csv"), target="is_claim")
        print("Preprocessing done")
        best_model = Modeller().model(X, y, ml_type)
        print("Modelling done")
        print(best_model.__class__.__name__)
        X = pd.DataFrame(X)
        Postprocessor().postprocess(best_model, X, y, ml_type)

if __name__ == "__main__":
    AutoPocketor().doJob()









