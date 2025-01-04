from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class Modeller():
    def __init__(self):
        """
        Porządny init.
        """
        pass

    def model(self, X, y, ml_type):
        """
        Porządny model.
        """
        if ml_type == "BINARY_CLASSIFICATION":
            m = DecisionTreeClassifier()

            param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 20]
            }

        else:
            m = DecisionTreeRegressor()
            param_grid = { 'criterion': ['friedman_mse'],
                           'max_depth': [20, 30, 40, 50],
                           'min_samples_split': [2, 5],
                           'min_samples_leaf': [1, 2, 5, 10, 15],

                           }

        gs = GridSearchCV(estimator=m, param_grid=param_grid, cv=3, scoring="neg_mean_squared_error")
        gs.fit(X, y)
        best_model = gs.best_estimator_

        print("Best score (MSE):", -gs.best_score_)

        return best_model
