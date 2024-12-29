import warnings
import matplotlib.pyplot as plt

import shap
from sklearn.model_selection import train_test_split
from autopocket.postprocessing.shap import ShapPLOT

class Postprocessor():
    def __init__(self):
        """
        Porządny init.
        """
        pass

    

    def postprocess(self, best_model, X, y, ml_task, results=None):
        """
        Porządny postprocess.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

        
        explainer = ShapPLOT.get_explainer(best_model, X_test)

        shap_values = explainer.shap_values(X_test)

        fig = plt.figure(figsize=(12, 6))  # Ustaw rozmiar figury

        shap.dependence_plot(
            "age_of_policyholder", shap_values[:, :, 1], X_test,
            interaction_index="population_density"
        )

        shap.dependence_plot(
            "age_of_policyholder", shap_values[:, :, 0], X_test,
            interaction_index="population_density"
        )


