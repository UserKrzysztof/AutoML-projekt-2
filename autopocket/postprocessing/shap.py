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
        #     return False
        
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
            "RandomForestClassifier",
            "DecisionTreeRegressor"
        ]:
            shap_explainer = shap.TreeExplainer(best_model)
        else:
            shap_explainer = shap.LinearExplainer(best_model, X_train)

        return shap_explainer
    
    @staticmethod
    def limit_df(X_test, y_test):
        print(y_test.shape, type(y_test))
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test, name="target")
        ROW_LIMIT = 1000
        if X_test.shape[0] > ROW_LIMIT:
            X_test.reset_index(inplace=True, drop=True)
            y_test.reset_index(inplace=True, drop=True)
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
            
            shap.summary_plot(
                shap_values, X_test_lim, show=False
            )
            fig = plt.gcf()
            fig.tight_layout()  
            plt.title(f"{best_model.__class__.__name__} SHAP summary plot")
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
            else:    
                vals = np.abs(shap_values).mean(0)

            feature_importance = pd.DataFrame(
                list(zip(X_test_lim.columns, vals)), columns=["feature", "shap_importance"]
            )
            feature_importance.sort_values(
                by=["shap_importance"], ascending=False, inplace=True
            )

            if model_file_path:
                feature_importance.to_csv(
                    os.path.join(model_file_path, f"{best_model.__class__.__name__}_shap_importance.csv"),
                    index=False,
                )
        except Exception as e:
            print(f"Error in shap_summary_plot: {e}")
            
    @staticmethod
    def shap_dependence(shap_values, X_test_lim, best_model, model_file_path = None, pdf=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = plt.figure(figsize=(14, 7))
            plots_counter = np.min([9, X_test_lim.shape[1]])
            cols_cnt = 3
            rows_cnt = 3
            if plots_counter < 4:
                rows_cnt = 1
            elif plots_counter < 7:
                rows_cnt = 2
                    
            for i in range(plots_counter):
                ax = fig.add_subplot(rows_cnt, cols_cnt, i + 1)
                shap.dependence_plot(
                    f"rank({i})",
                    shap_values,
                    X_test_lim,
                    show=False,
                    title=f"Importance #{i+1}",
                    ax=ax,
                )

            fig.tight_layout(pad=2.0)
            fig.savefig(
                os.path.join(
                    model_file_path, f"{best_model.__class__.__name__}_shap_dependence.png"
                )
            )
            plt.close("all")
    
    @staticmethod    
    def explain_with_shap(best_model, X_train, y_train, X_test, y_test, ml_task, model_file_path=None, pdf=None):
        
        if not ShapPLOT.is_available(best_model, X_train):
            return
        
        explainer = ShapPLOT.get_explainer(best_model, X_train)
        
        X_test_lim, y_test_lim = ShapPLOT.limit_df(X_test, y_test)

        shap_values = explainer.shap_values(X_test_lim)
        
        expected_value = explainer.expected_value
        
        if ml_task == "BINARY_CLASSIFICATION" and isinstance(shap_values, np.ndarray):
            shap_values = shap_values[:,:,1]
            expected_value = expected_value[1]
        
        ShapPLOT.shap_summary_plot(shap_values, X_test_lim, best_model, model_file_path, pdf)
        
        ShapPLOT.shap_dependence(shap_values, X_test_lim, best_model, model_file_path, pdf) #bez pdfa poki co
        
        df_preds = ShapPLOT.get_predictions(best_model, X_test_lim, y_test_lim)
        
        if ml_task == "BINARY_CLASSIFICATION":
            ShapPLOT.decisions_binary(df_preds, shap_values, expected_value, X_test_lim, y_test_lim, best_model, model_file_path, pdf) #bez pdfa poki co
        else:
            ShapPLOT.decisions_regression(df_preds, shap_values, expected_value, X_test_lim, best_model, model_file_path, pdf) #bez pdfa poki co



    @staticmethod
    def decisions_binary(
        df_preds,
        shap_values,
        expected_value,
        x_test_lim,
        y_test_lim,
        best_model,
        model_file_path,
        pdf,
    ):
        for t in np.unique(y_test_lim):
            fig = plt.gcf()
            shap.decision_plot(
                expected_value,
                shap_values[df_preds[df_preds.target == t].lp[:10], :],
                x_test_lim.loc[df_preds[df_preds.target == t].index[:10]],
                show=False,
            )
            fig.tight_layout(pad=2.0)
            fig.savefig(
                os.path.join(
                    model_file_path,
                    f"{best_model.__class__.__name__}_shap_class_{t}_worst_decisions.png",
                )
            )
            plt.close("all")

            fig = plt.gcf()
            shap.decision_plot(
                expected_value,
                shap_values[df_preds[df_preds.target == t].lp[-10:], :],
                x_test_lim.loc[df_preds[df_preds.target == t].index[-10:]],
                show=False,
            )
            fig.tight_layout(pad=2.0)
            fig.savefig(
                os.path.join(
                    model_file_path, f"{best_model.__class__.__name__}_shap_class_{t}_best_decisions.png"
                )
            )
            plt.close("all")
    
    @staticmethod
    def decisions_regression(
        df_preds,
        shap_values,
        expected_value,
        x_test_lim,
        best_model,
        model_file_path,
        pdf,
    ):
        fig = plt.gcf()
        shap.decision_plot(
            expected_value,
            shap_values[df_preds.lp[:10], :],
            x_test_lim.loc[df_preds.index[:10]],
            show=False,
        )
        fig.tight_layout(pad=2.0)
        fig.savefig(
            os.path.join(model_file_path, f"{best_model.__class__.__name__}_shap_worst_decisions.png")
        )
        plt.close("all")

        fig = plt.gcf()
        shap.decision_plot(
            expected_value,
            shap_values[df_preds.lp[-10:], :],
            x_test_lim.loc[df_preds.index[-10:]],
            show=False,
        )
        fig.tight_layout(pad=2.0)
        fig.savefig(
            os.path.join(model_file_path, f"{best_model.__class__.__name__}_shap_best_decisions.png")
        )
        plt.close("all")