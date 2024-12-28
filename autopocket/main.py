from preprocessing.Preprocessor import  Preprocessor
from algorithms.Modeller import  Modeller
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

        X, y, ml_type = Preprocessor().preprocess(path = "../example/data/car_insurance.csv", target = "is_claim")
        print("Preprocessing done")
        Modeller().model(X, y, ml_type)
        print("Modelling done")

        # Postprocessor().postprocess()

if __name__ == "__main__":
    AutoPocketor().doJob()









