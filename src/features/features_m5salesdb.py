import numpy as np
import pandas as pd

def extract_features(sales):
    """
    """

    # # One-hot encode the categorical values
    # cat_cols = sales.select_dtypes(include=['category']).columns
    # sales = pd.get_dummies(sales, columns=cat_cols)  # Necessary, but will make the number of columns explode and render the db less interpretable

    # Compute lag features and deal with the introduced nan values (drop or impute)
    lags = [1, 2, 3, 7, 14, 21, 28]
    for lag in lags:
        # Create the lagged column
        sales[f"sold_lag_{lag}"] = sales.groupby("id")["sold"].shift(lag)
        # Compute and fill NaNs with mean sales per item
        # sales[f"sold_lag_{lag}"] = sales.groupby("id")["sold"].transform(lambda x: x.fillna(x.mean()))

    # Compute mean encodings
    sales['sold_avg_item'] = sales.groupby('item_id')['sold'].transform('mean').astype(np.float16)  # IMPORTANT: when you want to compute aggregated values over a group and pass it to the original, not grouped dataframe, use .transform() and not .apply().
    sales['sold_avg_cat'] = sales.groupby('cat_id')['sold'].transform('mean').astype(np.float16)
    sales['sold_avg_dept'] = sales.groupby('dept_id')['sold'].transform('mean').astype(np.float16)
    sales['sold_avg_store'] = sales.groupby('store_id')['sold'].transform('mean').astype(np.float16)
    sales['sold_avg_state'] = sales.groupby('state_id')['sold'].transform('mean').astype(np.float16)

    sales['sold_avg_item_store'] = sales.groupby(['item_id', 'store_id'])['sold'].transform('mean').astype(np.float16)
    sales['sold_avg_item_state'] = sales.groupby(['item_id', 'state_id'])['sold'].transform('mean').astype(np.float16)
    sales['sold_avg_cat_store'] = sales.groupby(['cat_id', 'store_id'])['sold'].transform('mean').astype(np.float16)
    sales['sold_avg_cat_state'] = sales.groupby(['cat_id', 'state_id'])['sold'].transform('mean').astype(np.float16)

    # Windowed statistics
    sales["sold_rolling_7_mean"] = sales.groupby("id")["sold"].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)
    sales["sold_rolling_14_mean"] = sales.groupby("id")["sold"].transform(lambda x: x.rolling(window=14).mean()).astype(np.float16)
    sales["sold_rolling_28_mean"] = sales.groupby("id")["sold"].transform(lambda x: x.rolling(window=28).mean()).astype(np.float16)
    sales["sold_rolling_14_max"] = sales.groupby("id")["sold"].transform(lambda x: x.rolling(window=14).max()).astype(np.float16)

    sales['sold_expanding_mean'] = sales.groupby(["id"])["sold"].transform(lambda x: x.expanding().mean()).astype(np.float16)

    # Trend features

    X = Y = None
    return X, Y