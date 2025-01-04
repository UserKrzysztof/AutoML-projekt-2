class BinaryColumnHandler:
    """
    Class responsible for handling binary columns, including detection and conversion to binary format.
    """

    def __init__(self):
        """
        Initialize the BinaryColumnHandler class.
        """
        self.true_values = {'true', 'yes', '1'}
        self.false_values = {'false', 'no', '0', 'n0'}

    def is_binary_column(self, column):
        """
        Check if a column contains only values that can be converted to binary (1 or 0).
        It looks for values like: 'true', 'false', 'yes', 'no', '1', '0'.
        """
        # Get unique values from the column before applying the conversion
        unique_values = column.dropna().unique()
        # Define the possible binary values before conversion
        binary_values =  self.true_values | self.false_values

        # Check if all unique values in the column can be mapped to binary values
        return all(
            isinstance(val, str) and val.strip().lower() in binary_values
            for val in unique_values
        )

    def convert_yes_no_to_binary(self, value):
        """
        Convert 'yes', 'no', 'true', 'false' to binary 1 and 0 respectively.
        Checks for common variations in text representation.
        """
        if isinstance(value, str):
            value = value.strip().lower()  # Remove any spaces and convert to lowercase
            if value in self.true_values:
                return 1
            elif value in self.false_values:
                return 0
        return value
