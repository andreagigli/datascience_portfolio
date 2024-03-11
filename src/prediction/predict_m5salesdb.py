import numpy as np
import pandas as pd
from typing import Callable, Optional, Union


def predict(model: Callable,
            X_test: pd.DataFrame,
            Y_test: Optional[Union[pd.DataFrame, pd.Series]],
            start_day: int,
            extract_features_fn: Callable,
            *args, **kwargs) -> pd.DataFrame:
    """
    Splits the dataset into training, validation, and an extended test set to accommodate sequential prediction needs.
    The training data covers days 1 - 1911, the validation set spans days 1912 - 1940, and the test set extends from
    day 1941 to day 1969. The "sold" column for the first day of the test set (day 1941) includes actual sales figures,
    while the following days (1942 - 1969) initially have their "sold" values set to zero, pending prediction.

    The training set is utilized for model training, the validation set for model tuning and validation, and the
    extended test set for conducting sequential predictions. Each day's forecast within the prediction period can
    be informed by the actual or predicted sales data from preceding days, incorporating dynamic feature recalculation.

    Because the prediction problem requires a sequential approach, the test set is extended backward to include a
    specified number of look-back days. This extension facilitates the computation of features for days 1942 - 1969
    that depend on both the model's "sold" predictions and historical sales data, enabling the use of lagged or
    rolling window features.

    Args:
        X (pd.DataFrame): DataFrame containing features, with 'd' indicating the day.
        Y (pd.DataFrame): DataFrame containing the target variable, aligned with X.
        look_back_days (int): Number of days prior to the test period included in the extended test set for feature computation.

    Returns:
        X_train (pd.DataFrame): Features for the training set.
        Y_train (pd.DataFrame): Target variable for the training set.
        X_val (pd.DataFrame): Features for the validation set.
        Y_val (pd.DataFrame): Target variable for the validation set.
        X_test_extended (pd.DataFrame): Features for the extended test set, including the look-back period for feature computation and the actual prediction period.
        Y_test (pd.DataFrame): Target variable for the actual test period, excluding the look-back days.
        aux_split_params (Optional[Dict[str, any]]): Additional parameters like 'start_day_for_prediction' that may be useful for prediction.
    """
    if not {'id', 'd', 'sold'}.issubset(X_test.columns):
        raise ValueError("X_test must include 'id', 'd', and 'sold' columns.")

    if not kwargs.get("start_day_for_prediction"):
        raise ValueError("The kwargs must include the argument 'start_day_for_prediction'")

    Y_pred = pd.DataFrame(columns=['id', 'd', 'sold_pred'])
    start_day_for_prediction = kwargs["start_day_for_prediction"]

    # Iterate over days from start_day to the last day in X_test
    for day in sorted(X_test.loc[X_test["d"] >= start_day_for_prediction, "d"].unique())[:-1]:
        # Select rows for the current day
        day_data = X_test[X_test['d'] == day]

        # Make predictions for all items on the current day
        predictions = model.predict(day_data.squeeze())

        # Determine the next available day in the dataset
        current_day_index = list(sorted(X_test["d"].unique())).index(day)
        next_day = sorted(X_test["d"].unique())[current_day_index + 1]
        next_day_data = X_test[X_test['d'] == next_day]

        # Check if the items are in the same order for both day and next_day
        if (not next_day_data.empty and
                len(next_day_data) == len(predictions) and
                all(day_data['id'].values == next_day_data['id'].values)
        ):
            # Directly update the 'sold' values for next_day in X_test
            X_test.loc[X_test['d'] == next_day, 'sold'] = predictions
        else:
            # If the items are not aligned or if the lengths do not match, raise an error
            raise ValueError(f"Items are not aligned between day {day} and next day {next_day}")

        # if not next_day_data.empty and len(next_day_data) == len(predictions):
        #     # Attempt to align the items based on 'id', ensuring predictions match the correct items
        #     try:
        #         # Create a temporary DataFrame with predictions and corresponding 'id's from day_data
        #         pred_df = pd.DataFrame({'id': day_data['id'], 'sold_pred': predictions})
        #
        #         # Reindex the pred_df to match the order of 'id's in next_day_data
        #         pred_df = pred_df.set_index('id').reindex(index=next_day_data['id']).reset_index()
        #
        #         # Verify that the reindexing covers all ids by checking for NaNs in 'sold_pred'
        #         if pred_df['sold_pred'].isnull().any():
        #             raise ValueError(
        #                 f"Reindexing failed to cover all items for day {next_day}. Missing predictions for some items.")
        #
        #         # Update 'sold' values in X_test for next_day using the aligned predictions
        #         X_test.loc[X_test['d'] == next_day, 'sold'] = pred_df['sold_pred'].values
        #     except Exception as e:
        #         # Handle exceptions related to reindexing or updating
        #         print(f"An error occurred while processing day {day} to {next_day}: {e}")
        # else:
        #     # If lengths do not match or next_day_data is empty, skipping update
        #     raise ValueError(f"Skipping update from day {day} to {next_day} due to mismatched lengths or empty data.")

        # # TODO: Extract features for all items on the current day
        # next_day_features, _ = extract_features_fn(X_test.loc[X_test["d"] <= next_day, :], *args, **kwargs)
        # X_test[X_test['d'] == next_day] = next_day_features


    return Y_pred.reset_index(drop=True)