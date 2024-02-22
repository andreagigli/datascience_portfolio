import pandas as pd


def downcast(df):
    """
    Optimizes a DataFrame's memory usage by downcasting numeric types and converting strings to categories where appropriate.

    This function iterates over each column in the DataFrame and downcasts numeric types to the most
    memory-efficient type possible. For object types, it attempts to convert strings to categories if
    the number of unique values is less than 50% of the total values in the column. Dates are converted
    to datetime if a column is named 'date'.

    Args:
        df (pd.DataFrame): The DataFrame to optimize.

    Returns:
        pd.DataFrame: The optimized DataFrame with downcasted data types.
    """
    # If the input is a series, convert it to a dataframe
    series_as_input = False
    if isinstance(df, pd.Series):
        series_as_input = True
        df = pd.DataFrame(df, columns=["value"])

    # Iterate over each column in the DataFrame
    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_object_dtype(col_type):
            if col == 'date':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                # Attempt to convert to category only if the number of unique values is less than 50% of total values
                if len(df[col].unique()) / len(df[col]) < 0.5:
                    df[col] = df[col].astype('category')

    # If the input was a series, return it to a series
    if series_as_input:
        df = df["value"]

    return df
