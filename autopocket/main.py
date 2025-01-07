
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


    def doJob(self, path, target, model_file_path=None):

        """
        Porządny doJob.
        Bardzo krótki.
        """

        X, y, ml_type = Preprocessor().preprocess(path, target=target)
        print("Preprocessing done")

        import matplotlib.pyplot as plt
        plt.hist(y)
        plt.show()


        print(X.shape)
        best_model = Modeller().model(X, y, ml_type)
        print("Modelling done")
        print(f"Chosen model: {best_model.__class__.__name__}")
        X = pd.DataFrame(X)
        Postprocessor().postprocess(best_model, X, y, ml_type, model_file_path)
        print("Postprocessing done")


if __name__ == "__main__":
    AutoPocketor().doJob('../../walmart.csv', 'Weekly_Sales', ".")

