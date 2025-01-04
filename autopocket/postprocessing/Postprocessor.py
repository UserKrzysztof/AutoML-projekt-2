import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shap
from sklearn.model_selection import train_test_split
from autopocket.postprocessing.shap import ShapPLOT

class Postprocessor():
    def __init__(self):
        """
        Porządny init.
        """
        pass
    
    def postprocess(self, best_model, X, y, ml_task, model_file_path, results=None):
        
        """
        Porządny postprocess.
        """
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

        with PdfPages("shap_explanations_combined.pdf") as pdf:
            try:
                ShapPLOT.explain_with_shap(best_model, X_train, y_train, X_test, y_test, ml_task, model_file_path, pdf=pdf)
            except ValueError as e:
                print(f"ValueError in postprocess: {e}")

        print("PDF saved successfully.")



