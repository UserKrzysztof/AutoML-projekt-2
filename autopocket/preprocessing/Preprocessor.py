import pandas as pd

from .data_cleaning.DataCleaner import DataCleaner
from .task_analysing.ColumnTypeAnalyzer import ColumnTypeAnalyzer


class Preprocessor():
    def __init__(self):
        """
        porządny init

        """

        self.dataCleaner = DataCleaner()
        self.columnTypeAnalyzer = ColumnTypeAnalyzer()

    def preprocess(self, path, target, num_strategy='mean', cat_strategy='most_frequent', fill_value=None):
        """
        Main preprocessing function.
        """
        data = pd.read_csv(path, sep=';')
        X = data.drop(columns=[target])
        y = data[target]

        # Clean data
        X = self.dataCleaner.clean(X, num_strategy, cat_strategy, fill_value)

        # Analyze task type
        ml_task, y = self.columnTypeAnalyzer.analyze_column_type(y)

        return X, y, ml_task