import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


class Preprocessor():
    def __init__(self):
        """
        Porządny init.
        """
        pass

    def preprocess(self, path, target):
        """
        Porządny preprocess.
        """

        data = pd.read_csv(path, sep=',')
        X = data.drop(columns=[target])
        y = data[target]

        unique_counts = X.nunique()
        cols_to_drop = unique_counts[unique_counts == len(X)].index
        X = X.drop(columns=cols_to_drop)
        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(exclude=np.number).columns

        if len(num_cols) > 0:
            X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

        if len(cat_cols) > 0:
            X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])
            X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=False)

        X = pd.DataFrame(X)

        if self.analyze_column_type(y) == "BINARY_CLASSIFICATION":
            y = LabelEncoder().fit_transform(y)

        return X, y, self.analyze_column_type(y)


    def analyze_column_type(self, column):
        """
        Analyzes a column from a DataFrame and returns the data type:
        - BINARY_CLASSIFICATION if the column contains only 2 unique values.
        - LINEAR_REGRESSION if the column contains numeric data and more than 2 unique values.

        Parameters:
            column (pd.Series): Column from a DataFrame.

        Returns:
            str: Data type (BINARY_CLASSIFICATION or LINEAR_REGRESSION).
        """

        unique_values = np.unique(column).size

        if unique_values == 2:
            return "BINARY_CLASSIFICATION"
        elif np.issubdtype(column.dtype, np.number):
            return "LINEAR_REGRESSION"
        else:
            raise ValueError("The column is neither binary nor numeric for regression.")

