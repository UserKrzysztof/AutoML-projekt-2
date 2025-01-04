import re
import pandas as pd


class DateHandler:
    """
    Class responsible for handling and formatting dates.
    """
    def __init__(self):
        # Define date patterns as an instance attribute
        self.date_patterns = [
            (r"(\d{2})-(\d{2})-(\d{2})", "%d-%m-%y"),  # 07-05-20
            (r"(\d{2})\.(\d{2})\.(\d{2})", "%d.%m.%y"),  # 28.05.10
            (r"(\d{2})\s([a-z]+)\s(\d{2})", "%d %B %y"),  # 11 june 20
            (r"(\d{2})/(\d{2})/(\d{2})", "%d/%m/%y"),  # 25/06/10
            (r"(\d{1,2})\s([a-z]+)\s(\d{2})", "%d %B %y"),  # 4 june 20
            (r"(\d{1,2})/(\d{2})/(\d{2})", "%d/%m/%y"),  # 2/03/20
            (r"(\d{1,2})\.(\d{2})\.(\d{2})", "%d.%m.%y"),  # 3.08.12
            (r"(\d{1,2})\s([a-z]+)\s(\d{2}|\d{4})", "%d %b %Y"),  # 2 Apr 2010 or 2 Apr 10
            (r"(\d{2})[-./](\d{2})[-./](\d{2}|\d{4})", "%d-%m-%Y"),  # 02-04-2010 or 02/04/2010
            (r"(\d{2})\s([a-z]+)\s(\d{2}|\d{4})", "%d %B %Y"),  # 2 April 2010
            (r"(\d{1,2})\.(\d{1,2})\.(\d{2}|\d{4})", "%d.%m.%Y"),  # 02.04.2010 or 2.04.2010
            (r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", "%d/%m/%Y"),  # 02/04/2010 or 2/04/2010
            (r"(\d{1,2})-(\d{2})-(\d{4})", "%d-%m-%Y")  # 1-04-2011
        ]

    def fix_date_format(self, value):
        """
        Fix date format in various string formats to dd-MM-yyyy.
        """
        if isinstance(value, str):
            # First, normalize common formats to a standardized form
            value = value.strip().lower()

            # Try to match the date patterns
            for pattern, date_format in self.date_patterns:
                match = re.match(pattern, value)
                if match:
                    try:
                        # Try to parse the matched date using the appropriate format
                        parsed_date = pd.to_datetime(match.group(0), format=date_format)
                        return parsed_date.strftime('%d-%m-%Y')  # Return in the format dd-MM-yyyy
                    except Exception:
                        continue
        return value  # Return original if no match was found

    def is_date_column(self, column):
        """
        Check if a column contains dates (i.e., can be converted to datetime).
        We handle custom date formats using regex and pd.to_datetime.
        """
        try:
            sample_value = column.dropna().iloc[0]  # Take the first non-null value to check the format
            if isinstance(sample_value, str):
                # Check against each of the stored date patterns
                for pattern, _ in self.date_patterns:
                    if re.match(pattern, sample_value):
                        return True  # Recognize date column by regex pattern
            return False  # Return false if it doesn't match any known format
        except Exception:
            return False
