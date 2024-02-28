import numpy as np
import pandas as pd

def extract_features(sales: pd.DataFrame):
    """
    Extract and generate a set of features for the sales DataFrame, including lag and robust lag features,
    mean encodings, rolling-window and expanding-window statistics, detrended features, one-hot encoding of categorical
    variables. Finally, separate the processed DataFrame into input features (X) and the target variable (Y).


    Args:
        sales (pd.DataFrame): The input sales DataFrame. Must contain "id", "item_id", "cat_id", "dept_id",
                              "store_id", "state_id", and "sold" columns, among others.

    Returns:
        X (pd.DataFrame): The DataFrame containing the input features for modeling, excluding the target variable.
        Y (pd.Series): The Series containing the target variable "sold".
    """
    # Lag features
    print("Compute lag features")
    lags = [1, 2, 3, 7, 14, 30, 100, 365]
    for lag in lags:
        # Create the lagged column
        sales[f"sold_lag_{lag}"] = sales.groupby("id")["sold"].shift(lag).astype(np.float16)
        # Backfill the NaN values in the lagged columns for each group
        sales[f"sold_lag_{lag}"] = sales.groupby("id")[f"sold_lag_{lag}"].bfill()  # IMPORTANT: do not use .transform(lambda x: x.bfill()) if the dtype is different from np.float32

    # Robust lag features: obtained as a lagged rolling window (e.g. a rolling window of size 3 computed 7 days before the current value)
    print("Compute robust lag features")
    lags = [7, 30, 365]
    window_size_factor = 0.5  # size of the lagged window as a ration of the lag
    for lag in lags:
        window_size = int(lag * window_size_factor)
        window_size = max(window_size, 1)  # Ensure window_size is at least 1
        sales[f"sold_robustlag_{lag}"] = sales.groupby("id")["sold"].transform(
            lambda x: x.shift(lag)  # Shift the "sold" column by the specified lag to get the lagged data
            .rolling(window=window_size, min_periods=1)  # Apply rolling mean to the lagged data
            .mean()  # IMPORTANT: when you want to compute aggregated values over a group and pass it to the original, not grouped dataframe, use .transform(lambda x: x.mean()) and not .apply() or .mean()
        ).astype(np.float16)
        sales[f"sold_robustlag_{lag}"] = sales.groupby("id")[f"sold_robustlag_{lag}"].bfill()

    # Compute mean encodings
    print("Compute target encodings")
    sales["sold_avg_item"] = sales.groupby("item_id", observed=True)["sold"].transform(lambda x: x.mean()).astype(np.float16)
    sales["sold_avg_cat"] = sales.groupby("cat_id", observed=True)["sold"].transform(lambda x: x.mean()).astype(np.float16)
    sales["sold_avg_dept"] = sales.groupby("dept_id", observed=True)["sold"].transform(lambda x: x.mean()).astype(np.float16)
    sales["sold_avg_store"] = sales.groupby("store_id", observed=True)["sold"].transform(lambda x: x.mean()).astype(np.float16)
    sales["sold_avg_state"] = sales.groupby("state_id", observed=True)["sold"].transform(lambda x: x.mean()).astype(np.float16)

    sales["sold_avg_item_store"] = sales.groupby(["item_id", "store_id"], observed=True)["sold"].transform(lambda x: x.mean()).astype(np.float16)
    sales["sold_avg_item_state"] = sales.groupby(["item_id", "state_id"], observed=True)["sold"].transform(lambda x: x.mean()).astype(np.float16)
    sales["sold_avg_cat_store"] = sales.groupby(["cat_id", "store_id"], observed=True)["sold"].transform(lambda x: x.mean()).astype(np.float16)
    sales["sold_avg_cat_state"] = sales.groupby(["cat_id", "state_id"], observed=True)["sold"].transform(lambda x: x.mean()).astype(np.float16)

    # Rolling-window statistics
    print("Compute rolling-window statistics")
    window_sizes = [7, 30, 365]
    for window_size in window_sizes:
        sales[f"sold_avg_{window_size}d"] = sales.groupby("id", observed=True)["sold"].transform(lambda x: x.rolling(window=window_size).mean()).astype(np.float16)
        sales[f"sold_avg_{window_size}d"] = sales.groupby("id", observed=True)[f"sold_avg_{window_size}d"].bfill()  # Note: the alternative .transform(lambda x: x.bfill()) would crush if the col type is not np.float32!

    sales["sold_max_14d"] = sales.groupby("id", observed=True)["sold"].transform(lambda x: x.rolling(window=14).max()).astype(np.float16)
    sales[f"sold_max_14d"] = sales.groupby("id", observed=True)[f"sold_max_14d"].bfill()

    # Expanding window statistics
    print("Compute expanding-window statistics")
    sales["sold_avg_expanding"] = sales.groupby("id", observed=True)["sold"].transform(lambda x: x.expanding().mean()).astype(np.float16)

    # Detrended features
    print("Compute detrended features")
    sold_avg_item_day = sales.groupby("id", observed=True)["sold"].transform(lambda x: x.mean()).astype(np.float16)
    sales["sold_daily_deviation_from_avg"] = sold_avg_item_day - sales["sold_avg_item"]  # Divergence between avg daily sale of an item (across stores) and overall avg of that item

    sales["sold_daily_deviation_from_avg_30d"] = sold_avg_item_day - sales["sold_avg_30d"]  # Divergence between avg daily sale of an item (across stores) and 30d avg of that item
    # Backfill the NaN values in the daily deviation from the 30d avg
    sales[f"sold_daily_deviation_from_avg_30d"] = sales.groupby("id", observed=True)[f"sold_daily_deviation_from_avg_30d"].bfill()

    # # Trend indicators
    # print("Compute trend indicators")
    # sales["sold_linear_slope_30d"] = sales.groupby("item_id", observed=True)["sold"].rolling(window=30).apply(lambda x: linear_slope(x, downsampling_factor=5), raw=False)  # This can be very computational intensive

    # # Seasonality features commonly include day of the week, week number, year, and special events, with lagged
    # # features capturing autocorrelation related to seasonality. For cycles not aligned with calendar periods (e.g.,
    # # 4-day cycles) or when dealing with multiple seasonal patterns, Fourier Series Components are useful. They
    # # simplify encoding cyclical patterns by adding sine and cosine functions for identified cycles (e.g., 3-day,
    # # 50-day periods), with frequencies matching the cycles" periods. The phase is not explicitly encoded because the
    # # model will adjust the weights of the sine and cosine features to reflect the phase shifts.
    # print("Compute seasonality features")
    # dominant_period = 7  # A weekly periodicity was observed in eda. In this case the Fourier Series Components are an overkill becasue this periodicity matched the calendar periodicity (weekly, monthly, and yearly...)
    # sales["sold_sin_dominant_cycle"] = np.sin(2 * np.pi * sales["d"] / dominant_period)
    # sales["cos_dominant_cycle"] = np.cos(2 * np.pi * sales["d"] / dominant_period)

    # Convert binary columns to booleans
    print("Convert binary columns to integers")
    sales["snap_CA"] = sales["snap_CA"].astype(bool)
    sales["snap_TX"] = sales["snap_TX"].astype(bool)
    sales["snap_WI"] = sales["snap_WI"].astype(bool)
    """ 
    Strategies for dealing with high-cardinality categorical features like "item_id":
     1. Replace with category codes: Converts categories to their integer codes to reduce storage space. As a note, it"s 
        essential to map and store category-code pairs for later use in predictions.
     2. Target encoding: Useful if a correlation is suspected between the feature and target. Each category is replaced
        with the mean target value for that category, enabling a numerical representation that retains predictive value.
        As a note, it"s essential to account for regularization to avoid overfitting.
     3. One-hot encoding and dimensionality reduction: Feasible for moderately high cardinalities. One-hot encode the
        feature and then apply dimensionality reduction techniques (e.g., PCA) to the encoded data to manage the feature space.
     4. Clustering: Group data based on other features and replace the high-cardinality feature with cluster labels. This
        method reduces dimensionality while potentially capturing intrinsic patterns in the data unrelated to the target.
     5. Dropping the feature: Consider if the feature"s predictive value does not justify the complexity it adds to the model.
     """
    # In this case, simple category codes replacement for high-cardinality columns
    print("Handle high-cardinality non-ordinal categorical features")
    sales["id"] = sales["id"].astype("category").cat.codes
    sales["item_id"] = sales["item_id"].astype("category").cat.codes

    # One-hot encode the categorical features
    print("One-hot encode the categorical features")
    cat_cols = sales.select_dtypes(include=["category"]).columns
    sales = pd.get_dummies(sales, columns=cat_cols, sparse=True, dtype=bool)
    # Drop the "NoEvent" columns that arise from the one-hot encoding of the "event" columns (they are redundant)
    sales = sales.drop(columns=[col for col in sales.columns if "NoEvent" in col])

    # Zero-out the "sales-related" features that were computed for days 1942-1969 (test days). Since we will be
    # performing sequential inference, those features will be computed at inference time using the model prediction,
    # one day at a time.
    columns_to_zero = [col for col in sales.columns if col.startswith('sold_')]
    sales.loc[sales["d"] > 1941, columns_to_zero] = 0

    # Drop useless features
    print("Drop useless columns")
    sales = sales.drop(columns=["date", "wm_yr_wk", "weekday"])

    # Divide dataframe into X and Y
    print("Separate data into features X and targets Y")
    Y = sales["sold"]
    X = sales.drop(columns=["sold"])

    return X, Y
