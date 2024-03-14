import numpy as np
import pandas as pd
from typing import Callable, Optional, Union, Tuple, Dict


def predict(model: Callable,
            X_test: pd.DataFrame,
            Y_test: Optional[Union[pd.DataFrame, pd.Series]],
            X_train: pd.DataFrame,
            Y_train: Optional[Union[pd.DataFrame, pd.Series]],
            *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, any]]]:
    """
    Performs sequential predictions on the test dataset, with optional predictions for training and validation sets.
    The function uses the model to compute the 'sold' for days greater than the specified start day. It then
    recalculates features on the next day based on the predicted 'sold' value. The function also supports making
    predictions on the training and validation datasets if provided via kwargs.

    Args:
        model (Callable): The trained prediction model.
        X_test (pd.DataFrame): Test set features with 'id' and 'd' columns, and possibly 'sold' values to be updated.
        Y_test (Optional[Union[pd.DataFrame, pd.Series]]): Actual target values for the test set, used for evaluation if provided.
        X_train (pd.DataFrame): Training set features.
        Y_train (Optional[Union[pd.DataFrame, pd.Series]]): Actual target values for the training set.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments, which may include:
                  - "start_day_for_prediction" (int): The first day in X_test to start making predictions for.
                  - "extract_features_fn" (Callable): Function to dynamically extract features after each prediction.
                  - "X_val" (pd.DataFrame): Optional validation set features.

    Returns:
        Y_pred (np.ndarray): Array of predicted 'sold' values for the test set.
        Y_train_pred (np.ndarray): Array of predicted 'sold' values for the training set.
        optional_predictions (Optional[Dict[str, np.ndarray]]): Dictionary containing additional prediction arrays,
                                                                e.g., predictions for the validation set ("Y_val_pred").

    """
    if not {'id', 'd', 'sold'}.issubset(X_test.columns):
        raise ValueError("X_test must include 'id', 'd', and 'sold' columns.")

    if kwargs.get("start_day_for_prediction") is None:
        raise ValueError("The kwargs must include the argument 'start_day_for_prediction'.")

    if kwargs.get("extract_features_fn") is None:
        raise ValueError("The kwargs must include the argument 'extract_features_fn'.")

    # Initialize variables for sequential prediction
    start_day_for_prediction = kwargs["start_day_for_prediction"]
    extract_features_fn = kwargs["extract_features_fn"]
    Y_pred = X_test.loc[X_test["d"] >= start_day_for_prediction, ["id", "d"]].copy()  # Y_pred has auxiliary fields to align with X_test
    Y_pred["sold"] = 0

    # Perform sequential prediction starting from start_day_for_prediction until the second-last day in X_test
    for day in sorted(X_test.loc[X_test["d"] >= start_day_for_prediction, "d"].unique())[:-1]:
        print(f"Performing sequential prediction for day {day}")

        # Select rows for the current day
        day_data = X_test[X_test['d'] == day]

        # Make predictions for all items on the current day
        predictions = model.predict(day_data.squeeze())
        predictions = np.rint(np.clip(predictions, a_min=0, a_max=None))

        # Determine the next available day in the dataset
        next_day = min(d for d in sorted(X_test['d'].unique()) if d > day)
        next_day_data = X_test[X_test['d'] == next_day]

        # Attribute the predicted "sold" value to Y_pred.loc[Y_pred["d"]==day, "sold"] and to X_test.loc[X_test["d"]==next_day, "sold"].
        # Note that Y_pred will have a zero on the last day because the "sold" relative to the last day is actually stored in Y_pred.loc[Y_pred["d"]==second_last_day, "sold"]
        if any(day_data['id'].values != next_day_data['id'].values):
            # Preliminary check if the items "id" appears in the same order for both day and next_day. If not, it won't
            # be possible to assign the predicted sold to the next_day (TODO: implement a proper mapping to do so, in case).
            raise ValueError(f"Items are not aligned between day {day} and next day {next_day}")
        else:
            Y_pred.loc[Y_pred['d'] == day, 'sold'] = predictions
            X_test.loc[X_test['d'] == next_day, 'sold'] = predictions

        # Extract features for "next_day"
        next_day_features, _ = extract_features_fn(X_test.loc[X_test["d"] <= next_day, :], extract_features_only_for_these_days=[next_day])

        # Override the newly computed features for "next_day" to the values currently in X_test (only the values that
        # depend on "sold" will be changed).
        if any(X_test.loc[X_test['d'] == next_day, "id"] != next_day_features['id'].values):
            # Ensure the indices match between 'X_test' for 'next_day' and 'next_day_features' before updating
            raise ValueError(f"Items are not aligned between the original features of day {next_day} and the newly computed ones")
        else:
            # Align the indices of the newly computed features with the rows they will be replacing within X_test
            next_day_features.index = X_test.loc[X_test['d'] == next_day, :].index
            X_test.loc[X_test['d'] == next_day, :] = next_day_features

    # Strip the Y_pred of the accessory columns
    Y_pred = Y_pred["sold"].values

    # For model assment purposes, also predict the training set and the validation sets (the Kaggle challenge doesn't provide Y_test)
    Y_train_pred = model.predict(X_train.squeeze())
    Y_train_pred = np.rint(np.clip(Y_train_pred, a_min=0, a_max=None))

    optional_predictions = None
    if kwargs.get("X_val") is not None and kwargs.get("Y_val") is not None:
        Y_val_pred = model.predict(kwargs.get("X_val").squeeze())
        Y_val_pred = np.rint(np.clip(Y_val_pred, a_min=0, a_max=None))
        optional_predictions = {"Y_val_pred": Y_val_pred, "Y_val": kwargs.get("Y_val")}

    return Y_pred, Y_train_pred, optional_predictions