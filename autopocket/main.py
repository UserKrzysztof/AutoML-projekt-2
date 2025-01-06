
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


    def doJob(self, path, target, model_file_path=None):

        """
        Porządny doJob.
        Bardzo krótki.
        """

        X, y, ml_type = Preprocessor().preprocess(path, target=target)
        print("Preprocessing done")
        print(X.shape)
        best_model = Modeller().model(X, y, ml_type)
        print("Modelling done")
        print(f"Chosen model: {best_model.__class__.__name__}")
        X = pd.DataFrame(X)
        Postprocessor().postprocess(best_model, X, y, ml_type, model_file_path)
        print("Postprocessing done")


if __name__ == "__main__":
    AutoPocketor().doJob('example/data/car_insurance.csv', 'is_claim', ".")

