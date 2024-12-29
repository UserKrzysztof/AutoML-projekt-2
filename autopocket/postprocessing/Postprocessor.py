from autopocket.postprocessing.LimePostProcessor import LimePostprocessor
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

class Postprocessor():
    def __init__(self):
        """
        Porządny init.
        """
        pass


    def postprocess(self, best_model, X, y, ml_type, results=None):
        """
        Postprocessing logic, including LIME integration.
        """
        print("Starting postprocessing...")
        X_train = X.sample(frac=0.7, random_state=42)
        X_test = X.drop(X_train.index)

        lime_processor = LimePostprocessor()
        with PdfPages("lime_explanations_combined.pdf") as pdf:
            explanations = lime_processor.explain_top_observations_with_lime(
                model=best_model,  
                X_train=X_train,
                X_test=X_test,
                ml_type=ml_type,  
                num_features=10,
                pdf=pdf
            )
            lime_processor.lime_summary_plot(
                explanations=explanations,
                max_features=15,
                pdf=pdf
            )

        print("All plots have been saved to lime_explanations_combined.pdf")
        print("Postprocessing completed.")
        

