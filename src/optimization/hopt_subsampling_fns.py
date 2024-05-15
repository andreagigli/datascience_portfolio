from typing import Tuple, Optional, List

import numpy as np
import pandas as pd


def hopt_subsampling_passthrough(X, Y, **kwargs):
    return X, Y, kwargs.get('cv_indices', None)


def hopt_subsampling_m5salesdb(X: pd.DataFrame,
                               Y: pd.DataFrame,
                               cv_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                               subsampling_rate: float = 1.0,
                               random_seed: int = 42,
                               ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]:
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
            all_indices_mask[
                sampled_X.index] = True  # Mask [False False True True False True False True True False] (indicates preserved samples, refers to original X.index)

            adjusted_cv_indices = []
            for train_idx, test_idx in cv_indices:  # train_idx: [0 1 2 3 4] test_idx: [5 6 7 8 9]
                # Apply the mask to keep only the indices that exist in the subsampled dataset
                valid_train_idx = train_idx[np.in1d(train_idx,
                                                    sampled_X.index)]  # valid_train_idx: [2 3] (refers to mask, i.e. original X.index)
                valid_test_idx = test_idx[np.in1d(test_idx,
                                                  sampled_X.index)]  # valid_test_idx: [5 7 8] (refers to mask, i.e. original X.index)

                # Convert valid indices to positions within the subsampled dataset
                train_positions = np.where(np.in1d(sampled_X.index, valid_train_idx))[
                    0]  # train_positions: [0 1] (cv_indices adapted to sampled_X.index)
                test_positions = np.where(np.in1d(sampled_X.index, valid_test_idx))[
                    0]  # test_positions: [2 3 4] (cv_indices adapted to sampled_X.index)

                adjusted_cv_indices.append((train_positions, test_positions))

        # Now, reset the indices of sampled_X and sampled_Y
        # sampled_X = sampled_X.reset_index(drop=True)  # Not necessary to reset the index if Y has ("id", "d") as index
        # sampled_Y = sampled_Y.reset_index(drop=True)  # Not necessary to reset the index if Y has ("id", "d") as index

    return sampled_X, sampled_Y, adjusted_cv_indices
