class StringStripper:
    """
    Class responsible for stripping spaces from the beginning and end of strings in a DataFrame column.
    """

    def __init__(self):
        """
        Initialize the StringStripper class.
        """

    def strip_strings(self, column):
        """
        Strip spaces from strings in the provided column.
        """
        return column.str.strip() if column.dtype == 'object' else column