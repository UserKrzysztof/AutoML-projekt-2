class StringToLowerConverter:
    """
    Class responsible for converting strings in a DataFrame column to lowercase.
    """

    def __init__(self):
        """
        Initialize StringToLowerConverter class.
        """

    def to_lowercase(self, column):
        """
        Convert strings in the provided column to lowercase.
        """
        return column.str.lower() if column.dtype == 'object' else column
