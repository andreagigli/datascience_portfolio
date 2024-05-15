from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tabulate import tabulate


def convert_df_to_sparse_matrix(df: pd.DataFrame, fill_value: Union[int, float] = 0) -> csr_matrix:
    """
    Converts a pandas DataFrame with mixed dense and sparse columns into a scipy CSR sparse matrix.

    Args:
        df: A pandas DataFrame with any mix of dense and sparse columns.
        fill_value: The value used to fill "empty" entries in the dense columns before conversion. This is typically
                    0 or any other value that represents missingness in your dataset.

    Returns:
        A scipy CSR (Compressed Sparse Row) matrix representation of the input DataFrame.

    Raises:
        AttributeError: If the provided DataFrame does not contain numeric data types.
    """
    # Ensure all columns are numeric
    if not all(pd.api.types.is_numeric_dtype(col) for col in df.dtypes):
        raise AttributeError("DataFrame contains non-numeric columns that cannot be converted to a sparse matrix.")

    # Convert dense columns to SparseArrays with the specified fill_value
    for col in df.columns:
        if not pd.api.types.is_sparse(df[col].dtype):
            df[col] = pd.arrays.SparseArray(df[col], fill_value=fill_value)

    # Convert the entire DataFrame to a COO matrix, then to CSR format
    sparse_matrix = csr_matrix(df.sparse.to_coo())

    return sparse_matrix


def convert_to_dataframe(data: Union[np.ndarray, pd.DataFrame, pd.Series], prefix: str) -> pd.DataFrame:
    """
    Converts numpy arrays or pandas Series to a DataFrame with generated column names.

    Args:
        data: The input data to convert.
        prefix: The prefix for column names ('X' for features, 'Y' for targets).

    Returns:
        A pandas DataFrame representation of the input data.
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        col_names = [f'{prefix}{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=col_names)
    elif isinstance(data, pd.Series):
        return pd.DataFrame(data, columns=[f'{prefix}0'])
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise ValueError("Unsupported data type for conversion.")


def custom_info(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Generates a summary DataFrame containing the number of nans, the type, that number of unique values for each column in the provided DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame for which the summary is to be generated.

    Returns:
        pd.DataFrame: A new DataFrame with columns for the name of each column in `df`, the data type of each column,
                      the number of NaN values in each column, and the number of distinct values in each column.
    """
    # Creating the summary DataFrame
    summary_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes,
        "Number of NaNs": df.isna().sum(),
        "Number of Distinct Values": df.nunique()
    })

    # Resetting index to make it more readable
    summary_df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"Custom information about the dataframe: \n{summary_df}")

    return summary_df


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
            max_val = df[col].max()
            min_val = df[col].min()
            # Scale the values by 10 to account for future compatibility
            max_val_scaled = max_val * 10
            min_val_scaled = min_val * 10
            # Determine appropriate integer type
            if min_val_scaled > np.iinfo(np.int8).min and max_val_scaled < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif min_val_scaled > np.iinfo(np.int16).min and max_val_scaled < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif min_val_scaled > np.iinfo(np.int32).min and max_val_scaled < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        elif pd.api.types.is_float_dtype(col_type):
            max_val = df[col].max()
            min_val = df[col].min()
            # Scale the values by 10 to account for future compatibility
            max_val_scaled = max_val * 10
            min_val_scaled = min_val * 10
            # Determine appropriate float type
            # Check if float16 can handle the range safely
            if min_val_scaled > np.finfo(np.float16).min and max_val_scaled < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif min_val_scaled > np.finfo(np.float32).min and max_val_scaled < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

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


def pprint_db(df: pd.DataFrame, title: [str] = None) -> None:
    """
    Prints dataframe in a custom formatted way.

    Args:
        df (pd.DataFrame): The data to be printed
        title (str, optional): Title for the print section.
    """
    print("\n============================================================")
    print(f"{'DATAFRAME' if title is None else title.upper()}")
    print("============================================================")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print("============================================================\n")


def subsample_regular_interval(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Subsample a DataFrame by selecting rows at regular intervals based on the desired sample size.

    Args:
        df (pd.DataFrame): The DataFrame to subsample.
        sample_size (int): The number of samples to select.

    Returns:
        pd.DataFrame: A subsampled DataFrame with the specified number of samples.
    """
    total_rows = len(df)
    step = max(1, total_rows // sample_size)  # Calculate step size, ensure it's at least 1
    subsampled_df = df.iloc[::step]  # Select rows at intervals of the step size

    return subsampled_df.head(sample_size)  # Return exactly `sample_size` elements
