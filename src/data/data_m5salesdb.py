import numpy as np
import os
import pandas as pd
import re
import warnings

from pandas import DataFrame
from typing import Tuple, Optional, List

from src.utils.my_dataframe import downcast


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
        keep_n_items_per_category = 10
        # Sample unique items within each category
        sampled_items = sales.groupby('cat_id')['item_id'].apply(
            lambda x: x.drop_duplicates()
            .sample(n=min(len(x.drop_duplicates()), keep_n_items_per_category), random_state=0)
        ).reset_index(drop=True)
        # Filter based on the sampled items
        sales = sales[sales['item_id'].isin(sampled_items)]
        sell_prices = sell_prices[sell_prices["item_id"].isin(sampled_items)].reset_index(drop=True)

    return sales, sell_prices, calendar


def preprocess_data(sales: pd.DataFrame, sell_prices: pd.DataFrame, calendar: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses sales, sell prices, and calendar datasets for time series analysis.

    Steps:
    1. Inspect datasets for content, structure, and data integrity.
    2. Correct anomalies and standardize formats.
    3. Merge datasets into a single long-format dataframe.

    Inputs:
    - sales: DataFrame of daily sales data.
    - sell_prices: DataFrame of item prices.
    - calendar: DataFrame of date information.

    Parameters:
    - sales (pd.DataFrame): Daily sales data.
    - sell_prices (pd.DataFrame): Item price data.
    - calendar (pd.DataFrame): Date information data.

    Returns:
    - sales (pd.DataFrame): A unified dataset ready for feature engineering and subsequent analysis.
    """
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', None)

    # region Inspect dataset 'sales'
    print("\n\n\n*******************************************")
    print("************* Inspect 'sales' *************")
    print("*******************************************")
    print("\n************* Dataset structure *************")
    print("Shape:")
    print(sales.shape)
    print("There are more than 10 columns to be printed: attempt grouping patterned columns with same dtype.")
    # I know that this dataset has a group of patterned columns with the same type.
    # TODO: the presence of patterned columns with same dtype should be automatic
    patterned_columns = [col for col in sales.columns if re.match('d_\d+', col)]
    # same_type = all(sales[col].dtype == sales[patterned_columns[0]].dtype for col in patterned_columns)
    print("All patterned columns have the same type, let's group them for the purpose of understanding the dataset.\n")
    independent_cols = [col for col in sales.columns if col not in patterned_columns]
    sales_view = sales[independent_cols + [patterned_columns[0]]]
    sales_view = sales_view.rename(
        columns={patterned_columns[0]: f"PATT: {patterned_columns[0]} ... {patterned_columns[-1]}"})
    del independent_cols, patterned_columns
    print("Head:")
    print(sales_view.head())
    print("Info:")
    print(sales_view.info(show_counts=True))

    print("\n************* Missing data *************")
    print(f"Missing values: None, (see db.info() above)")

    print("\n************* Data description *************")
    print("Number of unique values:")
    print(sales_view.nunique())
    print("Value counts of categorical column:")
    print(sales_view['id'].value_counts())
    print("Value counts of categorical column:")
    print(sales_view['item_id'].value_counts())
    print("Value counts of categorical column:")
    print(sales_view['store_id'].value_counts())
    print("Description of numerical columns:")
    print(sales_view.describe())

    # endregion

    # region Inspect dataset 'sell_prices'
    print("\n\n\n*******************************************")
    print("************* Inspect 'sell_prices' *************")
    print("*******************************************")
    print("\n************* Dataset structure *************")
    print("Shape:")
    print(sell_prices.shape)
    print("Head:")
    print(sell_prices.head())
    print("Info:")
    print(sell_prices.info(show_counts=True))

    print("\n************* Missing data *************")
    print("Missing values: None, (see db.info() above)")

    print("\n************* Data description *************")
    print("Number of unique values:")
    print(sell_prices.nunique())
    print("Value counts of categorical column:")
    print(sell_prices['item_id'].value_counts())
    print("Value counts of categorical column:")
    print(sell_prices['store_id'].value_counts())
    print("Description of numerical columns:")
    print(sell_prices.describe())

    # endregion

    # region Inspect dataset 'calendar'
    print("\n\n\n*******************************************")
    print("************* Inspect 'calendar' *************")
    print("*******************************************")
    print("\n************* Dataset structure *************")
    print("Shape:")
    print(calendar.shape)
    print("Head:")
    print(calendar.head())
    print("Info:")
    print(calendar.info(show_counts=True))

    print("\n************* Missing data *************")
    print(
        "The db.info() above reveals suspect missing values in cols ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']. Remember: the db has",
        calendar.shape[0], "entries in total.")

    print("Study the NaN entries with db.isna().sum():")
    print(calendar.isna().sum())
    print("Study the NaN entries with db.unique() and db.value_counts():")
    print("Unique values in 'event_type_1':")
    print(calendar['event_type_1'].unique())
    print("Value counts for 'event_type_1':")
    print(calendar['event_type_1'].value_counts())
    print("Unique values in 'event_type_2':")
    print(calendar['event_type_2'].unique())
    print("Value counts for 'event_type_2':")
    print(calendar['event_type_2'].value_counts())

    print("Are the detected NaN entries supposed to be NaN?")
    # Some NaN entries may convey useful information (Missing Not At Random, MNAR), as opposed to just being
    # missing datapoints. For example, they may convey the absence of a special condition, a default state, or else.
    # In this case, the NaN must be expressed with a usable value of the same type of the other entries in the column.
    print("I find that the '<NA>' entries in the calendar actually represent a default event class 'NoEvent',"
          "and therefore must NOT be expressed as a NaN.")
    calendar[[
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2"]] = calendar[["event_name_1", "event_type_1", "event_name_2", "event_type_2"]].fillna("NoEvent")

    print("Are there entries that must become NaN?")
    # Online datasets may express NaN entries in non-standard ways such as "Not available" for strings or
    # -999 for numbers. In that case, pandas might miss those entries and explicit casting to pd.NA
    # (this is the current standard) is necessary. If so, df = df.replace({'Not available': pd.NA, -999: pd.NA}).

    print("Standardize all NaN entries.")
    # NaN values should be expressed in a standard way such as '<NA>' for strings or object columns
    # (although 'None' or 'NaN' are also used) or NaN for numeric columns. For maximum compatibility,
    # redefine all the NaN entries with pd.NA. If so, df = df.fillna(pd.NA).

    print("\n************* Data description *************")
    print("Number of unique values:")
    print(calendar.nunique())
    print("Value counts of categorical column:")
    print(calendar['weekday'].value_counts())
    print("Description of numerical columns:")
    print(calendar.describe())

    print("\n************* Sanity of temporal information *************")
    # Cast the date field into datetime format
    calendar['date'] = pd.to_datetime(calendar['date'])

    print("Check for duplicate date entries:")
    print(f"There are {'no ' if not calendar['date'].duplicated().any() else ''}duplicates in the 'date' column.")
    print(f"There are {'no ' if not calendar['d'].duplicated().any() else ''}duplicates in the 'd' column.\n")

    print("Ensure that fields 'date' and 'd' are monotonically increasing:")
    if not calendar['date'].is_monotonic_increasing:
        print("The 'date' column is not monotonically increasing. Sorting the DataFrame...")
        calendar.sort_values('date', inplace=True)
    else:
        print("The 'date' column is monotonically increasing.")
    assert calendar['d'].str.extract('(\d+)').astype(int).diff().dropna().gt(0).all().all(), \
        "The 'd' column is not monotonically increasing. This column does not logically correspond to the dates."

    print("\n************* Characteristics of temporal information *************")
    print(
        f"The start and end dates in the calendar are {calendar['date'].min()} and {calendar['date'].max()}, respectively.")

    print("Time gaps:")
    date_gaps = calendar['date'].diff().dt.days[1:]
    if date_gaps.nunique() != 1:
        print(f"The entries of calendar['date'] are equispaced. The gap is always {date_gaps.unique()[0]} day(s).")
    else:
        print(f"The entries of calendar['date'] are not equispaced. Here's some info about the gap distribution:")
        print(date_gaps.describe())

        """ One could also get a visual representation of the date information with these plots.

        # Plot data sample date on x axis
        plt.figure(figsize=(10, 6))
        plt.title('Date of each data sample')
        plt.plot(calendar['date'], np.zeros(len(calendar['date'])), marker='o', linestyle='none')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=False)

        # Histogram of date gaps
        plt.figure(figsize=(10, 6))
        date_gaps.plot(kind='hist', bins=30, figsize=(10, 6), title='Distribution of Date Gaps')
        plt.xlabel('Gap to next day (days)')
        plt.ylabel('Frequency')
        plt.show(block=False)

        # Cumulative distribution of date gaps
        cumulative_distribution = np.cumsum(date_gaps)
        plt.figure(figsize=(10, 6))
        plt.plot(calendar['date'][1:], cumulative_distribution, marker='.', linestyle='-')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Gap Days')
        plt.title('Cumulative Distribution of Date Gaps Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=False)

        """

    # endregion

    # region Join datasets

    print("\n\nPrepare to join sales, sell_prices, calendar datasets")

    print("Downcast data to save memory")
    sales = downcast(sales)
    calendar = downcast(calendar)
    sell_prices = downcast(sell_prices)

    """
    Joining together sell price and date information for multiple items at multiple time points involves transforming 
    all dataframes into a long format. In this format, each row corresponds to one item at one time point, organizing 
    the data to effectively represent temporal sequences for each item with both static and dynamic features:
        item1, staticFeature1, ..., sf10, timePoint1
        item1, staticFeature1, ..., sf10, timePoint2
        ...
        item1, staticFeature1, ..., sf10, timePoint1946
        item2, staticFeature1, ..., sf10, timePoint1
        ...
        item3000, staticFeature1, ..., sf10, timePoint1945
        item3000, staticFeature1, ..., sf10, timePoint1946

    For Sklearn models, feature engineering transforms the temporal data into a format where each row represents a 
    comprehensive set of features for one item, incorporating lagged and rolling window calculations to encapsulate 
    temporal dependencies within a single observation. This approach facilitates the use of regression or 
    classification algorithms that do not inherently process sequential data:
        item1, staticFeature1, ..., sf10, lagFeature1, ..., lf1946, rollingFeature1, rf2
        ...
        item3000, staticFeature1, ..., sf10, lagFeature1, ..., lf1946, rollingFeature1, rf2
    In this case, data rows encapsulate all necessary historical context, allowing for rows shuffling as each sample
    is treated independently.

    Partitioning the data into smaller sequences for Sklearn is an option for focusing on shorter temporal dependencies,
    leading to multiple rows per item, each encapsulating a portion of the item's temporal sequence:
        item1, staticFeature1, ..., sf10, lagFeature1, ..., lf900, rollingFeature1_1-900, rf2_1-900
        item1, staticFeature1, ..., sf10, lagFeature901, ..., lf1946, rollingFeature1_901-1946, rf2_901-1946
        ...
        item3000, staticFeature1, ..., sf10, lagFeature1, ..., lf900, rollingFeature1_1-900, rf2_1-900
        item3000, staticFeature1, ..., sf10, lagFeature901, ..., lf1946, rollingFeature1_901-1946, rf2_901-1946
    In this case, shuffling the rows requires caution as each row is part of a larger temporal segments. One must not 
    shuffle rows pertaining to the same item as they belong to a common temporal segment. Nonetheless, one can shuffle
    blocks of rows pertaining to different items. 

    For LSTMs and other sequence models, the data must maintain its sequential integrity, formatted into a 3D array 
    (n_items x n_timePoints x n_features) to allow these models to leverage temporal order and dependencies directly.
    In this case, it is possible to shuffle the rows, as each row pertains to a different item.
    """
    # Ensure all dataframes are in long format with each row representing a unique item-timePoint
    print("Melt sales dataframe")
    sales = pd.melt(sales,
                    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                    var_name="d",
                    value_name="sold",
                    ignore_index=True,
                    )

    # Integrate the calendar dates into (notice "left") the sales dataframe based on d
    print("Merge sales and calendar dataframes")
    sales = pd.merge(sales, calendar, on="d", how="left")

    # Integrate the sell prices into the sales dataframe based on the common fields
    print("Merge sales and sell_prices dataframes")
    sales = pd.merge(sales, sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # endregion

    # region Data processing

    print("\n\nPreprocess dataframes")

    print("Simplify sales['id']")
    if sales['id'].str.endswith('_evaluation').all():
        sales['id'] = sales['id'].str.replace('_evaluation', '')

    print("Convert the day into a number")
    sales["d"] = sales["d"].apply(lambda s: s.split("_")[1]).astype(np.int16)

    print("Align weekday and wday columns (monday=1, tuesday=2, ...)")
    weekday_to_num = {'Monday': 1,
                      'Tuesday': 2,
                      'Wednesday': 3,
                      'Thursday': 4,
                      'Friday': 5,
                      'Saturday': 6,
                      'Sunday': 7}
    sales['wday'] = sales['weekday'].map(weekday_to_num)

    print("Cast specific columns to categorical type")
    columns_to_convert = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1',
                          'event_name_2', 'event_type_2']  # Note that binary columns must be left numerical
    sales[columns_to_convert] = sales[columns_to_convert].astype('category')

    # Cast data to most adequate types in order to save memory
    print("Downcast data to save memory")
    sales = downcast(sales)

    # Check residual NaN values and handle them through elimination or samples or imputation
    print("Check if there are any remaining columns with NaN: ")
    sales.isna().any()
    print(f"Impute NaN values for the column sell_price as the median sell_price for the item")
    # Step 1: Attempt group-wise Median Imputation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sales['sell_price'] = sales.groupby("id")['sell_price'].transform(lambda x: x.fillna(x.median()))
    # Step 2: Global Median Imputation for remaining NaNs (for whom the sell_price might be NaN for the entire group)
    sales['sell_price'] = sales['sell_price'].fillna(sales['sell_price'].median())

    # Sort the dataframe so that the temporal evolution of the target variable is immediately evident for each item
    sales = sales.sort_values(by=["id", "d"]).reset_index(drop=True)  # Sort by the item and then by the day

    # endregion

    return sales


def split_data(X: DataFrame, Y: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, Optional[List[Tuple[np.ndarray, np.ndarray]]]]:
    """
    Splits the dataset into training, validation, and test sets based on the sample's day.
    The training set includes data from day 1 until 139, the validation set data from

    Args:
        X (DataFrame): DataFrame containing the features.
        Y (DataFrame): DataFrame containing the target variable.

    Returns:
        X_train (DataFrame): Training set features.
        Y_train (DataFrame): Training set target variable.
        X_val (DataFrame): Validation set features.
        Y_val (DataFrame): Validation set target variable.
        X_test (DataFrame): Test set features.
        Y_test (DataFrame): Test set target variable.
        cv_indices (List[Tuple[np.ndarray, np.ndarray]]): Placeholder for cross-validation indices, indicating no cross-validation indices are provided in this function.
    """
    idx_train = X["d"] < 1912
    idx_val = (X["d"] >= 1912) & (X["d"] < 1941)
    idx_test = X["d"] >= 1941

    X_train = X.loc[idx_train].reset_index(drop=True)
    Y_train = Y.loc[idx_train].reset_index(drop=True)
    X_val = X.loc[idx_val].reset_index(drop=True)
    Y_val = Y.loc[idx_val].reset_index(drop=True)
    X_test = X.loc[idx_test].reset_index(drop=True)
    Y_test = Y.loc[idx_test].reset_index(drop=True)

    cv_indices = None

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices


def subsample_items(X: pd.DataFrame, Y: pd.DataFrame, cv_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None, subsampling_rate: float = 1.0, random_seed: int = 42) -> Tuple[
    pd.DataFrame, pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Randomly selects a specified number of unique items (item_id) and includes all samples for those items.

    Args:
        X (pd.DataFrame): The features DataFrame containing an 'item_id' column.
        Y (pd.DataFrame): The target DataFrame with the same indices as X.
        cv_indices (Optional[List[Tuple[np.ndarray, np.ndarray]]]): CV indices before subsampling.
        subsampling_rate (float): A number in [0, 1] reflecting the proportion of the original dataset that is retained.
        random_seed (int): The seed used for reproducibility of the random selection.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]: A tuple containing the sampled feature and target DataFrames, and optionally the adjusted CV indices.
    """
    if subsampling_rate >= 1:  # Skip the whole function is the sampling rate is 1 (or erroneously higher)
        sampled_X = X
        sampled_Y = Y
        adjusted_cv_indices = cv_indices

    else:
        # Ensure Y's index aligns with X
        if not X.index.equals(Y.index):
            raise ValueError("Indices of X and Y must match.")

        # Get unique item_ids
        unique_items = X['item_id'].unique()

        # Calculate the number of items to sample
        num_items_to_sample = max(1, int(len(unique_items) * subsampling_rate))
        num_items_to_sample = min(num_items_to_sample, len(unique_items))  # Safeguard

        # Randomly select unique item_ids
        rng = np.random.default_rng(random_seed)  # Use numpy's random generator for a seed
        selected_items = rng.choice(unique_items, size=num_items_to_sample, replace=False)

        # Filter X and Y to include only rows with the selected item_ids
        sampled_X = X[X['item_id'].isin(selected_items)]
        sampled_Y = Y.loc[sampled_X.index]

        # Before resetting the index, adjust CV indices if provided
        adjusted_cv_indices = None
        if cv_indices is not None:
            """Example: 
            X.index [0 1 2 3 4 5 6 7 8 9] (dataset index before sampling)
            sampled_X.index: [2 3 5 7 8] (dataset index after sampling)
            cv_indices: [([0 1 2 3 4],[5 6 7 8 9 10]),...] (cv_indices before sampling)
            """
            # Convert original indices to a mask
            max_index = X.index.max()  # 9
            all_indices_mask = np.zeros(max_index + 1, dtype=bool)
            all_indices_mask[sampled_X.index] = True  # Mask [False False True True False True False True True False] (indicates preserved samples, refers to original X.index)

            adjusted_cv_indices = []
            for train_idx, test_idx in cv_indices:  # train_idx: [0 1 2 3 4] test_idx: [5 6 7 8 9]
                # Apply the mask to keep only the indices that exist in the subsampled dataset
                valid_train_idx = train_idx[np.in1d(train_idx, sampled_X.index)]  # valid_train_idx: [2 3] (refers to mask, i.e. original X.index)
                valid_test_idx = test_idx[np.in1d(test_idx, sampled_X.index)]  # valid_test_idx: [5 7 8] (refers to mask, i.e. original X.index)

                # Convert valid indices to positions within the subsampled dataset
                train_positions = np.where(np.in1d(sampled_X.index, valid_train_idx))[0]  # train_positions: [0 1] (cv_indices adapted to sampled_X.index)
                test_positions = np.where(np.in1d(sampled_X.index, valid_test_idx))[0]  # test_positions: [2 3 4] (cv_indices adapted to sampled_X.index)

                adjusted_cv_indices.append((train_positions, test_positions))

        # Now, reset the indices of sampled_X and sampled_Y
        sampled_X = sampled_X.reset_index(drop=True)
        sampled_Y = sampled_Y.reset_index(drop=True)

    return sampled_X, sampled_Y, adjusted_cv_indices
