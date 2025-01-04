import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
import os

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
    def limit_df(X_test, y_test):
        ROW_LIMIT = 1000
        if X_test.shape[0] > ROW_LIMIT:
            X_test.reset_index(inplace=True, drop=True)
            X_test.reset_index(inplace=True, drop=True)
            X_test_lim = X_test.sample(ROW_LIMIT)
            y_test_lim = y_test[X_test_lim.index]
        else:
            X_test_lim = X_test
            y_test_lim = y_test
        return X_test_lim, y_test_lim
    
    
    def get_predictions(best_model, X_test_lim, y_test_lim):
        
        predictions = best_model.predict(X_test_lim)
        residuals = np.abs(np.array(y_test_lim) - predictions)
        pred_dataframe = pd.DataFrame(
                    {"res": residuals, "lp": range(residuals.shape[0]), "target": np.array(y_test_lim)},
                        index=X_test_lim.index,
                    )
        pred_dataframe = pred_dataframe.sort_values(by="res", ascending=False)
        return pred_dataframe
    
    @staticmethod
    def shap_summary_plot(shap_values, X_test_lim, best_model, model_file_path=None, pdf=None):
        try: 
            #do usuniecia
            print(f"shap_values type: {type(shap_values)}")
            
            if isinstance(shap_values, np.ndarray):
                print(f"shap_values shape: {shap_values.shape}")
            
            assert not np.isnan(shap_values).any(), "shap_values contains NaN values"
            
            shap.summary_plot(
                shap_values, X_test_lim, plot_type="bar", show=False
            )
            fig = plt.gcf()
            fig.tight_layout()  
            if pdf:
                pdf.savefig(fig)  
            plt.close(fig)

            vals = None
            if isinstance(shap_values, list):
                vals = None
                for sh in shap_values:
                    v = np.abs(sh).mean(0)
                    if vals is None:
                        vals = v
                    else:
                        vals += v
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3 and shap_values.shape[2] == 2:
                    shap_values = shap_values[:, :, 1]
                vals = np.abs(shap_values).mean(0)
            else:
                raise ValueError(f"Unexpected type for shap_values: {type(shap_values)}")

            feature_importance = pd.DataFrame(
                list(zip(X_test_lim.columns, vals)), columns=["feature", "shap_importance"]
            )
            feature_importance.sort_values(
                by=["shap_importance"], ascending=False, inplace=True
            )
            print(type(best_model.__class__.__name__))
            if model_file_path:
                feature_importance.to_csv(
                    os.path.join(model_file_path, f"{best_model.__class__.__name__}_shap_importance.csv"),
                    index=False,
                )
        except Exception as e:
            print(f"Error in shap_summary_plot: {e}")
    
    @staticmethod    
    def explain_with_shap(best_model, X_train, y_train, X_test, y_test, ml_task, model_file_path=None, pdf=None):
        
        if not ShapPLOT.is_available(best_model, X_train):
            return
        
        explainer = ShapPLOT.get_explainer(best_model, X_train)
        X_test_lim, y_test_lim = ShapPLOT.limit_df(X_test, y_test)
        
        #do usuniecia
        print(f"X_test_lim: {X_test_lim.shape}")
        
        # Dodaj sprawdzenie dla X_test_lim
        assert not X_test_lim.isnull().values.any(), "X_test_lim contains NaN values"

        shap_values = explainer.shap_values(X_test_lim)
        
        #do usuniecia
        print(f"shap_values: {np.array(shap_values).shape}")
        
        # Dodaj sprawdzenie dla shap_values
        assert not pd.isnull(shap_values).any(), "shap_values contains NaN values"
        
        expected_value = explainer.expected_value
        if ml_task == "BINARY_CLASSIFICATION" and isinstance(shap_values, list):
            shap_values = shap_values[1]
            expected_value = explainer.expected_value[1]
        
        ShapPLOT.shap_summary_plot(shap_values, X_test_lim, best_model, model_file_path, pdf)
