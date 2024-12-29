import warnings
import matplotlib.pyplot as plt

import shap
from sklearn.model_selection import train_test_split

class ShapPLOT:
    
    @staticmethod
    def is_available(best_model, X_train):
        if best_model.__class__.__name__ in ["CatBoost","Neural Network", "Baseline"]:
            return False
        if X_train.shape[1] > 400:
            warnings.warn(
                "Too many columns for SHAP explanations"
            )
            return False
        
        if X_train.shape[0] < 30:
            warnings.warn(
                "Not enough records for SHAP explanations"
            )
            return False
        return True
    
    @staticmethod
    def get_explainer(best_model, X_train):
        model_name = best_model.__class__.__name__
        shap_explainer = None
        if model_name in [
            "DecisionTreeClassifier", #do dodania potencjalnie inne modele
            "RandomForestClassifier"
        ]:
            shap_explainer = shap.TreeExplainer(best_model)
        else:
            shap_explainer = shap.LinearExplainer(best_model, X_train)

        return shap_explainer
    
    @staticmethod
    def limit_df(X_validation, y_validation):
        SAMPLES_LIMIT = 1000
        if X_validation.shape[0] > SAMPLES_LIMIT:
            X_validation.reset_index(inplace=True, drop=True)
            y_validation.reset_index(inplace=True, drop=True)
            X_vald = X_validation.sample(SAMPLES_LIMIT)
            y_vald = y_validation[X_vald.index]
        else:
            X_vald = X_validation
            y_vald = y_validation
        return X_vald, y_vald