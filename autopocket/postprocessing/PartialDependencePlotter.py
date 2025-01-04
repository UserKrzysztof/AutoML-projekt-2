from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

class PartialDependencePlotter:
    def __init__(self):
        """
        Initialize the PartialDependencePlotter class.
        """
        pass

    def generate_pdp(self, model, X, features_non_binary, features_top, pdf=None):
        """
        Generate Partial Dependence Plots for the specified features.
        Display PDP for top non-binary features and save all plots (4 per page) to the PDF.

        Parameters:
            model: The trained model to explain.
            X: pd.DataFrame, input features.
            features_non_binary: list, top non-binary features for displaying plots.
            features_top: list, top features (binary and non-binary) for saving to the PDF.
            pdf: PdfPages object to save plots. If None, no plots are saved.
        """
        print("Displaying PDP plots for non-binary features...")
        for feature in features_non_binary:
            fig, ax = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(
                model,
                X,
                [feature], 
                ax=ax
            )
            plt.title(f"Partial Dependence Plot for {feature}")
            plt.tight_layout()
            plt.show()
            plt.close(fig)

        if pdf:
            print("Saving PDP plots to PDF...")
            for i in range(0, len(features_top), 4):
                n_features = len(features_top[i:i + 4]) 

                if n_features == 1: 
                    fig, ax = plt.subplots(figsize=(8, 6))
                    PartialDependenceDisplay.from_estimator(
                        model,
                        X,
                        [features_top[i]],
                        ax=ax
                    )
                    ax.set_title(f"Partial Dependence Plot for {features_top[i]}")
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                else: 
                    rows = (n_features + 1) // 2  
                    fig, axes = plt.subplots(rows, 2, figsize=(16, 6 * rows))
                    axes = axes.flatten() 

                    for j, feature in enumerate(features_top[i:i + 4]):
                        PartialDependenceDisplay.from_estimator(
                            model,
                            X,
                            [feature], 
                            ax=axes[j]
                        )
                        axes[j].set_title(f"Partial Dependence Plot for {feature}")

                    for k in range(len(features_top[i:i + 4]), len(axes)):
                        axes[k].set_visible(False)

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)