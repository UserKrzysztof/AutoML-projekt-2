import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shap
from sklearn.model_selection import train_test_split
from autopocket.algorithms.utils import ResultsReader
from autopocket.postprocessing.ModelsLeaderboard import ModelsLeaderboard
from autopocket.postprocessing.shap import ShapPLOT
from autopocket.postprocessing.LimePostProcessor import LimePostprocessor

from autopocket.postprocessing.PartialDependencePlotter import PartialDependencePlotter
from autopocket.postprocessing.ICEPlotter import IndividualConditionalExpectationPlotter
import pandas as pd
from autopocket.postprocessing.utils import validate_features_for_displaying, plot_features_with_explanations


class Postprocessor():
    def __init__(self, results_dir): ####
        """
        Porządny init.
        """
        self.pdp_plotter = PartialDependencePlotter()
        self.ice_plotter = IndividualConditionalExpectationPlotter()
        self.lime_processor = LimePostprocessor()
        self.reader = ResultsReader(results_dir) #####
        self.models_leaderboard = ModelsLeaderboard(self.reader) #####
        pass

    

    
    def postprocess(self, best_model, X, y, ml_task, features_for_displaying_plots=None, subset_fraction_for_ICE_plot=None):
        """
        Perform postprocessing to analyze and explain the performance of the best-trained model.

        This method generates explanations and visualizations to interpret the behavior of the 
        best-performing machine learning model. It includes SHAP and LIME analyses, feature importance 
        summaries, and generates Partial Dependence and ICE plots for selected features.

        Parameters:
            best_model: object
                The trained machine learning model to be explained.
            X: pd.DataFrame
                Input features used in the model.
            y: pd.Series or np.ndarray
                Target variable corresponding to X.
            ml_task: str
                Type of machine learning task. Options are:
                - "BINARY_CLASSIFICATION"
                - "REGRESSION"
            display_plots: bool, optional, default=True
                Whether to display plots during postprocessing.
            features_for_displaying_plots: list, optional, default=None
                Specific features for which Partial Dependence and ICE plots should be displayed.
                It might contain only non-binary columns. ICE plots and PDP plots will not be generated for 
                binary features.
                If there are no non-binary features specified or the parameter is set to None, the method 
                will automatically select top features based on LIME feature importance.
            subset_fraction_for_ICE_plot: float, optional, default=None
                Fraction of rows from X to use for ICE plot generation. If None, use all rows.

        Workflow:
            1. Splits the input data into training and test sets based on the task type.
            2. Creates SHAP explanations for global and local interpretability.
            3. Generates LIME explanations for selected instances and aggregates feature importance.
            4. If `features_for_displaying_plots` is None:
                - Automatically selects top features for PDP and ICE plots using LIME feature importance.
            5. Generates Partial Dependence and ICE plots for specified or automatically selected features.
            6. Saves all results, including plots and explanations, in a structured PDF file.

        Outputs:
            - A PDF file saved to `results/explanations/`, containing:
                - SHAP summary plots and feature dependence plots.
                - LIME explanations for influential instances.
                - PDP and ICE plots for selected features.
            - All plots and explanations are also displayed if `display_plots=True`.

        Raises:
            ValueError: If there is an issue during the postprocessing step, such as incorrect input data or model type.

        """

        if ml_task == "BINARY_CLASSIFICATION":
            X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        else:
            X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.3)
        model_name = best_model.__class__.__name__
        os.makedirs(os.path.join(os.getcwd(), 'results', 'explanations'), exist_ok=True)
        output_file = os.path.join(os.getcwd(), 'results', 'explanations', f"explanations_{model_name}.pdf")
        
        try: #####
            leaderboard = self.models_leaderboard.create_leaderboard()
            self.models_leaderboard.save_leaderboard_to_csv(leaderboard)
        except Exception as e:
            print(f"Error during leaderboard creation: {e}") #####
        
        with PdfPages(output_file) as pdf:
            try:
                ShapPLOT.explain_with_shap(best_model, X_train, X_test, y_test, ml_task, pdf=pdf)
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

                features_for_displaying_plots = validate_features_for_displaying(features_for_displaying_plots, X)

                plot_features_with_explanations(
                    best_model=best_model,
                    X=X,
                    explanations=explanations,
                    lime_processor=self.lime_processor,
                    pdp_plotter=self.pdp_plotter,
                    ice_plotter=self.ice_plotter,
                    pdf=pdf,
                    features_for_displaying_plots=features_for_displaying_plots,
                    subset_fraction_for_ICE_plot=subset_fraction_for_ICE_plot
                )

            except ValueError as e:
                print(f"ValueError in postprocess: {e}")

        print(f"All plots have been saved to {output_file}")
        
