import pandas as pd


def downcast(df):
    """
    Downcast the data types of a Pandas DataFrame to more memory-efficient types.

    Parameters:
    - df: pandas.DataFrame to downcast.

    Returns:
    - df: The same DataFrame with downcast data types for numeric and object columns.
    """
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

    return df
