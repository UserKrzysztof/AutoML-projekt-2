import numpy as np
from sklearn.preprocessing import LabelEncoder


class ColumnTypeAnalyzer:
    """
    Class to analyze the type of a target column for machine learning tasks.
    """

    def analyze_column_type(self, column):
        """
        Analyze the type of the target column (binary classification or linear regression).
        :param column: pandas Series - target column
        :return: string - "BINARY_CLASSIFICATION" or "LINEAR_REGRESSION"
        """
        unique_values = np.unique(column).size
        if unique_values == 2:
            return "BINARY_CLASSIFICATION", LabelEncoder().fit_transform(column)
        elif np.issubdtype(column.dtype, np.number):
            return "LINEAR_REGRESSION", column
        else:
            raise ValueError("The column is neither binary nor numeric for regression.")
