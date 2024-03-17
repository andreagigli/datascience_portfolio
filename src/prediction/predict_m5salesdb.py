import warnings

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
    if 'start_day_for_prediction' not in kwargs:
        raise ValueError("The 'start_day_for_prediction' argument is required.")
    if 'extract_features_fn' not in kwargs:
        raise ValueError("The 'extract_features_fn' argument is required.")
    start_day_for_prediction = kwargs["start_day_for_prediction"]
    extract_features_fn = kwargs["extract_features_fn"]

    # Initialize variables for sequential prediction
    Y_pred = pd.DataFrame(
        index=X_test.loc[X_test.index.get_level_values('d') >= start_day_for_prediction].index,
        data={"sold_next_day": 0},
        columns=["sold_next_day"],
    )

    # Perform sequential prediction starting from start_day_for_prediction until the second-last day in X_test
    for day in sorted(X_test.index.get_level_values('d').unique()):
        if day < start_day_for_prediction:
            continue  # Skip days before the start day for prediction

        print(f"Performing sequential prediction for day {day}")

        # Select rows for the current day
        day_data = X_test.loc[(slice(None), day), :]

        # Make predictions for all items on the current day
        predictions = model.predict(day_data.squeeze())
        predictions = np.rint(np.clip(predictions, a_min=0, a_max=None))

        # Find the next day to predict
        future_days = X_test.index.get_level_values('d').unique()
        future_days = future_days[future_days > day]  # Days after the current day
        if len(future_days) == 0:
            break  # No more days to predict
        next_day = future_days[0]

        # Assign predictions to Y_pred and update 'sold' in X_test for next_day
        Y_pred.loc[(slice(None), day), 'sold_next_day'] = predictions
        X_test.loc[(slice(None), next_day), 'sold'] = predictions

        # Extract features for next_day and update X_test
        next_day_features, _ = extract_features_fn(X_test.loc[X_test.index.get_level_values('d') <= next_day, :], extract_features_only_for_these_days=[next_day])
        # Before updating X_test with next_day_features, adjust dtypes to match exactly (to avoid minor type mismatches between pandas and numpy datatypes)
        with warnings.catch_warnings():  # Now update X_test
            warnings.simplefilter("ignore", FutureWarning)  # Filtering out the following because resistant to explicit type checking AND casting. "FutureWarning: Setting an item of incompatible dtype is deprecated and will raise in a future error of pandas. Value ... has dtype incompatible with int8, please explicitly cast to a compatible dtype first."
            X_test.update(next_day_features)

    # For model assment purposes, also predict the training set and the validation sets (the Kaggle challenge doesn't provide Y_test)
    Y_train_pred = model.predict(X_train.squeeze())
    Y_train_pred = np.rint(np.clip(Y_train_pred, a_min=0, a_max=None))
    if Y_train is not None:
        Y_train_pred = pd.DataFrame(
            index=Y_train.index,
            data={"sold_next_day": Y_train_pred},
            columns=["sold_next_day"],
        )

    optional_predictions = None
    if kwargs.get("X_val") is not None and kwargs.get("Y_val") is not None:
        Y_val_pred = model.predict(kwargs.get("X_val").squeeze())
        Y_val_pred = np.rint(np.clip(Y_val_pred, a_min=0, a_max=None))
        if kwargs.get("Y_val") is not None:
            Y_val_pred = pd.DataFrame(
                index=kwargs.get("Y_val").index,
                data={"sold_next_day": Y_val_pred},
                columns=["sold_next_day"],
            )
        optional_predictions = {"Y_val_pred": Y_val_pred, "Y_val": kwargs.get("Y_val")}

    return Y_pred, Y_train_pred, optional_predictions
