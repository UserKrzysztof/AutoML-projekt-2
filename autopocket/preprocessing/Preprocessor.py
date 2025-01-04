import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from .DateHandler import DateHandler


class Preprocessor():
    def __init__(self):
        """
        Initialize Preprocessor class.
        Creates a DateHandler object for date handling.
        """

        self.date_handler = DateHandler()

    def preprocess(self, path, target):
        """
        Main preprocessing function.
        """
        data = pd.read_csv(path, sep=';')
        X = data.drop(columns=[target])
        y = data[target]

        # 1. Remove spaces from the beginning and end of string columns
        X = X.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

        # 2. Fix different number formats (comma vs dot)
        X = X.apply(lambda x: x.apply(self.fix_number_format) if x.dtype == 'object' else x)

        # 3. Convert all string columns to lowercase
        X = X.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)

        # 4. Convert yes/no, true/false to binary (only for appropriate columns)
        for col in X.select_dtypes(include=[object]):
            if self.is_binary_column(X[col]):
                X[col] = X[col].apply(self.convert_yes_no_to_binary)

        # 5. Convert date columns to a consistent format (dd-MM-yyyy)
        for col in X.select_dtypes(include=[object]):
            if self.date_handler.is_date_column(X[col]):
                X[col] = X[col].apply(self.date_handler.fix_date_format)

        # Drop constant columns
        unique_counts = X.nunique()
        cols_to_drop = unique_counts[unique_counts == len(X)].index
        X = X.drop(columns=cols_to_drop)

        # Separate numerical and categorical columns
        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(exclude=np.number).columns

        # Handle missing data
        if len(num_cols) > 0:
            X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

        if len(cat_cols) > 0:
            X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])
            X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=False)

        X = pd.DataFrame(X)

        # Analyze column type
        ml_task = self.analyze_column_type(y)
        if ml_task == "BINARY_CLASSIFICATION":
            y = LabelEncoder().fit_transform(y)

        return X, y, ml_task

    def convert_yes_no_to_binary(self, value):
        """
        Convert 'yes', 'no', 'true', 'false' to binary 1 and 0 respectively.
        Checks for common variations in text representation.
        """
        if isinstance(value, str):
            value = value.strip().lower()  # Remove any spaces and convert to lowercase
            if value in ['true', 'yes', '1']:
                return 1
            elif value in ['false', 'no', '0']:
                return 0
        return value

    def fix_number_format(self, value):
        """
        Fix number format (comma vs dot) in string columns.
        """
        if isinstance(value, str):
            # Replace comma with dot for decimal numbers
            value = value.replace(',', '.')
            try:
                return float(value)
            except ValueError:
                return value  # Return as is if it's not a valid number
        return value


    def is_binary_column(self, column):
        """
        Check if a column contains only values that can be converted to binary (1 or 0).
        It looks for values like: 'true', 'false', 'yes', 'no', '1', '0'.
        """
        # Get unique values from the column before applying the conversion
        unique_values = column.dropna().unique()
        # Define the possible binary values before conversion
        binary_values = {'true', 'yes', '1', 'false', 'no', '0'}

        # Check if all unique values in the column can be mapped to binary values
        if all(val.strip().lower() in binary_values for val in unique_values):
            return True
        return False


    def analyze_column_type(self, column):
        """
        Analyze the type of the target column (binary classification or linear regression).
        """
        unique_values = np.unique(column).size
        if unique_values == 2:
            return "BINARY_CLASSIFICATION"
        elif np.issubdtype(column.dtype, np.number):
            return "LINEAR_REGRESSION"
        else:
            raise ValueError("The column is neither binary nor numeric for regression.")
