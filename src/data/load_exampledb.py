from typing import Tuple

import pandas as pd
from pandas import DataFrame


def load_data(fpath: str) -> Tuple[DataFrame, DataFrame]:
    """
    Loads the California Housing dataset from a CSV file.

    This dataset contains data collected from the 1990 California census. It includes information such as median income,
    housing median age, average rooms, average bedrooms, block population, average occupancy, latitude, and longitude
    for different blocks. The target variable is the median house value.

    Parameters:
    fpath (str): Path to the CSV file containing the dataset.

    Returns:
    X (pd.DataFrame): DataFrame containing the features (predictor variables).
    Y (pd.DataFrame): DataFrame containing the target variable (median house value).
    """

    # Load the dataset

    data = pd.read_csv(fpath)

    # Assuming the target variable is named 'median_house_value'
    Y = data[["MedHouseVal"]]
    X = data.drop("MedHouseVal", axis=1)

    return X, Y
