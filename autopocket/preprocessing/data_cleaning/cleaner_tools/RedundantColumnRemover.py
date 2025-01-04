class RedundantColumnRemover:
    """
    Class responsible for identifying and removing redundant columns (constant or fully unique).
    """

    def drop_redundant_columns(self, df):
        """
        Remove columns where all values are constant or unique.
        :param df: pandas DataFrame
        :return: DataFrame with redundant columns removed
        """
        # Identify columns with constant values (only one unique value)
        unique_counts = df.nunique()
        cols_to_drop = unique_counts[unique_counts == 1].index
        df.drop(columns=cols_to_drop, inplace=True)

        # Identify columns with unique values (equal to the number of rows)
        cols_to_drop = unique_counts[(unique_counts == len(df)) & (df.dtypes == 'object')].index
        df.drop(columns=cols_to_drop, inplace=True)

        return df
