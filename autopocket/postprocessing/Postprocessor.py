
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shap
from sklearn.model_selection import train_test_split
from postprocessing.shap import ShapPLOT
from postprocessing.LimePostProcessor import LimePostprocessor

from postprocessing.PartialDependencePlotter import PartialDependencePlotter
from postprocessing.ICEPlotter import IndividualConditionalExpectationPlotter
import pandas as pd

class Postprocessor():
    def __init__(self):
        """
        Porządny init.
        """
        self.pdp_plotter = PartialDependencePlotter()
        self.ice_plotter = IndividualConditionalExpectationPlotter()
        self.lime_processor = LimePostprocessor()
        pass

    
    def postprocess(self, best_model, X, y, ml_task, model_file_path, results=None):

        """
        Postprocessing logic, including LIME integration.
        """

        if ml_task == "BINARY_CLASSIFICATION":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model_name = best_model.__class__.__name__
        output_file = f"explanations_{model_name}.pdf"

        with PdfPages(output_file) as pdf:
            try:
                ShapPLOT.explain_with_shap(best_model, X_train, y_train, X_test, y_test, ml_task, model_file_path, pdf=pdf)
                if not isinstance(y, pd.Series):
                    y = pd.Series(y, name="target")
                explanations = self.lime_processor.explain_top_observations_with_lime(
                    model=best_model,  
                    X_train=X_train,
                    X_test=X_test,
                    ml_type=ml_task,  
                    num_features=10,
                    pdf=pdf
                )
                self.lime_processor.lime_summary_plot(
                    explanations=explanations,
                    max_features=15,
                    pdf=pdf
                )
                print("Selecting top features based on LIME Feature Importance...")
                top_non_binary_features, top_all_features = self.lime_processor.top_features_by_lime_importance(
                    explanations=explanations,
                    X=X,
                    top_n_non_binary=3,
                    top_m_all=8
                )

                print("Generating Partial Dependence Plots...")
                self.pdp_plotter.generate_pdp(best_model, X, top_non_binary_features, top_all_features, pdf=pdf)

                print("Generating Individual Conditional Expectation (ICE) Plots...")
                self.ice_plotter.generate_ice(best_model, X, top_non_binary_features, top_all_features, pdf=pdf)

            except ValueError as e:
                print(f"ValueError in postprocess: {e}")

     


        print(f"All plots have been saved to {output_file}")
        print("Postprocessing completed.")
        
