import pandas as pd

from autopocket.preprocessing.Preprocessor import  Preprocessor
from autopocket.algorithms.Modeller import Modeller
from autopocket.postprocessing.Postprocessor import Postprocessor


class AutoPocketor():
    def __init__(self):
        """
        Class for performing AutoML tasks.
        """
        pass

    def doJob(self, path, target, model_file_path=None, display=True):
        """
        Method for performing AutoML tasks.
        """

        print("Performing preprocessing...")
        X, y, ml_type = Preprocessor().preprocess(path = path, target=target)
        print("X shape:", X.shape)
        print("Preprocessing done.")


        print("\nPerforming modelling...")
        best_model = Modeller().model(X, y, ml_type)
        print(f"Chosen model: {best_model.__class__.__name__}")
        print("Modelling done.")

        X = pd.DataFrame(X)

        print("\nPerforming postprocessing...")
        Postprocessor().postprocess(best_model, X, y, ml_type)
        print("Postprocessing done.")