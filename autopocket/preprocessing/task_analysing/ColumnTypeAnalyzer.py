import numpy as np
from sklearn.preprocessing import LabelEncoder


class ColumnTypeAnalyzer:
    """
    Class to analyze the type of a target column for machine learning tasks.
    It determines whether the column is suitable for binary classification or linear regression.
    """

    def analyze_column_type(self, column):
        """
        Analyze the type of the target column (binary classification or linear regression).

        Parameters:
        - column: pandas Series - The target column to analyze.

        Returns:
        - string: "BINARY_CLASSIFICATION" or "LINEAR_REGRESSION" indicating the type of machine learning task.
        - pandas Series: Transformed target column for further modeling.
          - For binary classification, the column is label-encoded.
          - For linear regression, the column is returned as is.

        Steps:
        1. If the column has exactly 2 unique values, it is considered for binary classification.
        2. If the column is numeric, it is considered for linear regression.
        3. If the column has more than 2 unique values and is not numeric, it raises an exception.

        Notes:
        - The `LabelEncoder` is used to transform the column for binary classification.
        """
        # Count unique values in the column
        unique_values = np.unique(column).size

        # Check for binary classification
        if unique_values == 2:
            return "BINARY_CLASSIFICATION", LabelEncoder().fit_transform(column)

        # Check for linear regression
        elif np.issubdtype(column.dtype, np.number):
            return "LINEAR_REGRESSION", column

        # Raise an error if the column type doesn't match expected formats
        else:
            raise ValueError("The column is neither binary nor numeric for regression.")
