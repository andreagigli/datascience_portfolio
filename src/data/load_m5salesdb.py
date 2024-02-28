import os

import numpy as np
import pandas as pd

from pandas import DataFrame
from typing import Tuple


def load_data(dpath: str, debug: bool = False) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Loads the M5 sales dataset, aimed at forecasting Walmart product sales for the next 28 days.

    The sales_train_evaluation.csv file includes historical daily sales data over 1941 days
    (columns d_1 to d_1941), suitable for splitting into training (d_1 to d_1913) and validation (d_1914 to d_1941) sets.
    The dataset includes item ID, category, department, store, and state.
    According to the Kaggle challenge instructions, one should add 28 extra columns (d_1942 to d_1790) for the test
    predictions.

    The sell_prices.csv file provides selling prices for each item, and calendar.csv contains date-related
    information for each of the 1941 days (e.g., day of the week, special events).

    Reference: [M5 Forecasting - Accuracy on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

    Parameters:
    dpath (str): Path to the dataset CSV files.
    debug (str): If true eliminates a number of time points for faster processing.

    Returns:
    sales (pd.DataFrame): Dataframe containing main feature and historical sales for all the items.
    sell_prices (pd.DataFrame): Dataframe containing sell price of each item.
    calendar (pd.DataFrame): Dataframe containing the selling data for datapoint (d_* column in sales).
    """
    # Load the dataset
    sales = pd.read_csv(os.path.join(dpath, "sales_train_evaluation.csv"))
    sell_prices = pd.read_csv(os.path.join(dpath, "sell_prices.csv"))
    calendar = pd.read_csv(os.path.join(dpath, "calendar.csv"))

    # Add zero sales for the remaining days 1942-1969 (they are present in calendar and sell_prices (as weeks) but not in sales)
    days = ['d_' + str(d) for d in range(1942, 1970)]
    sales[days] = 0
    sales[days] = sales[days].astype(np.int16)

    # Cast all the object columns into string columns
    sales = sales.astype({col: pd.StringDtype() for col in sales.select_dtypes('object').columns})
    sell_prices = sell_prices.astype({col:  pd.StringDtype()for col in sell_prices.select_dtypes('object').columns})
    calendar = calendar.astype({col:  pd.StringDtype() for col in calendar.select_dtypes('object').columns})

    # Eliminate part of the items for faster computation while in debug
    if debug:
        sales = sales.groupby(['store_id', "cat_id"]).head(5).reset_index(drop=True)
        unique_item_ids = sales["item_id"].unique()
        sell_prices = sell_prices[sell_prices["item_id"].isin(unique_item_ids)].reset_index(drop=True)

    return sales, sell_prices, calendar
