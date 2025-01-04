import re


class NumberFormatFixer:
    """
    Class to fix number formats in columns by replacing commas with dots,
    and ensuring the column is considered numeric if it contains valid numbers.
    """

    def fix_column_format(self, column):
        """
        Check the first 3 values of a column to determine if it's numeric,
        then replace commas with dots and leave the column as a string to be handled later.

        :param column: pandas Series - the column to be processed
        :return: pandas Series - the column with the fixed number format
        """
        # Check if three random values are valid numbers
        values_to_check = column.dropna().sample(n=min(3, len(column.dropna())), random_state=34)
        if all(self.is_number(value) for value in values_to_check):
            # Replace commas with dots for all values in the column
            column = column.apply(lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else x)
        return column

    def is_number(self, value):
        """
        Helper function to check if a value is a valid number (either integer or float),
        allowing for commas in the decimal part.

        :param value: any type - value to check
        :return: bool - True if the value is a valid number, False otherwise
        """
        # Check if value is a number using regular expressions
        if isinstance(value, str):
            # Allow for an optional comma or dot for decimal point
            return bool(re.match(r'^-?\d+(\.|,)\d+$', value.strip())) or bool(re.match(r'^\d+$', value.strip()))
        return False
