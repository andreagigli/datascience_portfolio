from typing import Tuple

import pandas as pd
from pandas import DataFrame


def load_data(fpath: str) -> Tuple[DataFrame, DataFrame]:
    """
    Loads data from a specified CSV file into DataFrames for features and target variables.

    Args:
        fpath (str): The file path to the CSV file containing the dataset.

    Returns:
        X (DataFrame): A DataFrame containing the features (predictor variables). This includes columns
                       such as median income, housing median age, average rooms, average bedrooms, block
                       population, average occupancy, latitude, and longitude for different blocks.
        Y (DataFrame): A DataFrame containing the target variable, which is the median house value for
                       different blocks.
    """

    # Load the dataset

    data = pd.read_csv(fpath)

    # Assuming the target variable is named 'median_house_value'
    Y = data[["MedHouseVal"]]
    X = data.drop("MedHouseVal", axis=1)

    return X, Y
