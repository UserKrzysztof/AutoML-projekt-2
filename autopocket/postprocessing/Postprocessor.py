import warnings
import matplotlib.pyplot as plt

import shap
from sklearn.model_selection import train_test_split


class Postprocessor():
    def __init__(self):
        """
        Porządny init.
        """
        pass

    @staticmethod
    def get_explainer(best_model, X_train):
        model_name = best_model.__class__.__name__
        shap_explainer = None
        if model_name in [
            "DecisionTreeClassifier",
            "RandomForestClassifier"
        ]:
            shap_explainer = shap.TreeExplainer(best_model)
        else:
            shap_explainer = shap.LinearExplainer(best_model, X_train)

        return shap_explainer

    def postprocess(self, best_model, X, y, ml_task, results=None):
        """
        Porządny postprocess.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

        if X_train.shape[1] > 400:
            warnings.warn(
                "Too many columns for SHAP explanations"
            )
            return False
        if X_train.shape[0] < 30:
            warnings.warn(
                "Not enough columns for SHAP explanations"
            )
            return False
        explainer = Postprocessor.get_explainer(best_model, X_test)
        shap_values = explainer.shap_values(X_test)
        # for i in range shap_values.shape[0]:
        #     for j in range shap_values.shape[0]:
        #         for k in range shap_values.shape[0]:
        # plots = [
        #     shap.dependence_plot("age_of_policyholder", shap_values[:, :, 1], X_test, interaction_index="population_density"),
        #     shap.dependence_plot("age_of_policyholder", shap_values[:, :, 0], X_test, interaction_index="population_density")
        #     ] #X.colnames[0]?
        # fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        # axs[0].plot(plots[0])
        # axs[1].plot(plots[1])
        fig = plt.figure(figsize=(12, 6))  # Ustaw rozmiar figury

        # Dodanie pierwszego subplotu
        ax1 = fig.add_subplot(1, 2, 1)  # 1 wiersz, 2 kolumny, pierwszy subplot
        shap.dependence_plot(
            "age_of_policyholder", shap_values[:, :, 1], X_test,
            interaction_index="population_density", ax=ax1
        )
        ax1.set_title("SHAP Dependence Plot - Value 1")

        # Dodanie drugiego subplotu
        ax2 = fig.add_subplot(1, 2, 2)  # 1 wiersz, 2 kolumny, drugi subplot
        shap.dependence_plot(
            "age_of_policyholder", shap_values[:, :, 0], X_test,
            interaction_index="population_density", ax=ax2
        )
        ax2.set_title("SHAP Dependence Plot - Value 0")

        # Optymalizacja układu i wyświetlenie
        fig.tight_layout()
        plt.show()

