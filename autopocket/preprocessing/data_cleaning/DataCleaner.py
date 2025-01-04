from .cleaner_tools.DateHandler import DateHandler
from .cleaner_tools.StringStripper import StringStripper
from .cleaner_tools.StringToLowerConverter import StringToLowerConverter
from .cleaner_tools.BinaryColumnHandler import BinaryColumnHandler
from .cleaner_tools.RedundantColumnRemover import RedundantColumnRemover
from .cleaner_tools.DataImputer import DataImputer
from .cleaner_tools.NumberFormatFixer import NumberFormatFixer


class DataCleaner():
    def __init__(self):
        """
        Initialize the DataCleaner class.
        """

        self.dateHandler = DateHandler()
        self.stringStripper = StringStripper()
        self.stringToLowerConverter = StringToLowerConverter()
        self.binaryColumnHandler = BinaryColumnHandler()
        self.redundantColumnRemover = RedundantColumnRemover()
        self.dataImputer = DataImputer()
        self.numberFormatFixer = NumberFormatFixer()

    def clean(self, X, num_strategy='mean', cat_strategy='most_frequent', fill_value=None):
        """
        Perform data cleaning on the input data X.
        """

        # 1. Remove spaces from the beginning and end of string columns
        X = X.apply(self.stringStripper.strip_strings)

        # 2. Fix different number formats (comma vs dot)
        X = X.apply(lambda x: self.numberFormatFixer.fix_column_format(x) if x.dtype == 'object' else x)

        # 3. Convert all string columns to lowercase
        X = X.apply(self.stringToLowerConverter.to_lowercase)

        # 4. Drop redundant columns
        X = self.redundantColumnRemover.drop_redundant_columns(X)

        # 5. Convert yes/no, true/false to binary (only for appropriate columns)
        for col in X.select_dtypes(include=[object]):
            if self.binaryColumnHandler.is_binary_column(X[col]):
                X[col] = X[col].apply(self.binaryColumnHandler.convert_yes_no_to_binary)

        # 6. Convert date columns to a consistent format (yyyyMMdd)
        for col in X.select_dtypes(include=[object]):
            if self.dateHandler.is_date_column(X[col]):
                X[col] = X[col].apply(self.dateHandler.fix_date_format)

        # 7. Handle missing data using DataImputer with provided strategies
        X = self.dataImputer.impute(X, num_strategy, cat_strategy, fill_value)

        return X