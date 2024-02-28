"""
This script is designed for flexible machine learning workflows. It allows for dynamic loading of data, preprocessing, feature extraction, model training, and evaluation based on specified function identifiers. Users can specify model parameters, choose to train a new model or use a pre-trained one, and control data splitting for training and testing.

Example shell calls:

1. Basic usage with required arguments (no hypopt, split train-test)
python analysis_exampledb.py
--data_path ../../data/external/exampledb/m5salesdb.csv
--precomputed_features_path ../../data/processed/m5salesdb/
--data_loading_fn load_exampledb
--model sklearn_RandomForestRegressor
--model_hparams "{\"n_estimators\": 100, \"max_depth\": 10}"
--preprocessing_fn preprocess_passthrough
--feature_extraction_fn features_passthrough
--split_fn split_train_test
--split_ratio "80 20"
--evaluation_fn evaluate_exampledb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ../../data/processed/
--output_model_dir ../../models/
--output_reports_dir ../../outputs/reports/
--output_figures_dir ../../outputs/figures/

2. Basic usage with minimal function call (no hypopt, split train-test)
python analysis_exampledb.py
--data_path ../../data/external/exampledb/california_housing.csv
--data_loading_fn load_exampledb
--model sklearn_RandomForestRegressor
--split_fn split_train_test
--split_ratio "80 20"
--evaluation_fn evaluate_exampledb
--log_level INFO
--random_seed 0

3. Basic usage with minimal function call (no hypopt, assuming loaded data is already split)
python analysis_exampledb.py
--data_path ../../data/external/exampledb/california_housing.csv
--data_loading_fn load_exampledb
--model sklearn_RandomForestRegressor
--evaluation_fn evaluate_exampledb
--log_level INFO
--random_seed 0

4. Usage with hyperparameters optimization (hypopt, split train-test + kfold)
python analysis_exampledb.py
--data_path ../../data/external/exampledb/california_housing.csv
--data_loading_fn load_exampledb
--model sklearn_RandomForestRegressor
--model_hparams "{\"n_estimators\": \"randint(10, 100)\", \"max_depth\": 5}"
--preprocessing_fn preprocess_passthrough
--feature_extraction_fn features_passthrough
--split_fn split_train_test
--split_ratio "80 20"
--n_folds 3
--evaluation_fn evaluate_exampledb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ../../data/processed
--output_model_dir ../../models
--output_reports_dir ../../outputs/reports
--output_figures_dir ../../outputs/figures

5. Usage with hyperparameters optimization (hypopt, split train-val-test)
python analysis_exampledb.py
--data_path ../../data/external/exampledb/california_housing.csv
--data_loading_fn load_exampledb
--model sklearn_RandomForestRegressor
--model_hparams "{\"n_estimators\": \"randint(10, 100)\", \"max_depth\": 5}"
--preprocessing_fn preprocess_passthrough
--feature_extraction_fn features_passthrough
--split_fn split_train_val_test
--split_ratio "70 15 15"
--evaluation_fn evaluate_exampledb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ../../data/processed
--output_model_dir ../../models
--output_reports_dir ../../outputs/reports
--output_figures_dir ../../outputs/figures

6. Usage with tensorflow model
python analysis_exampledb.py
--data_path ../../data/external/exampledb/california_housing.csv
--data_loading_fn load_exampledb
--model tensorflow_mynet
--preprocessing_fn preprocess_passthrough
--feature_extraction_fn features_passthrough
--split_fn split_train_val_test
--split_ratio "70 15 15"
--evaluation_fn evaluate_exampledb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ../../data/processed
--output_model_dir ../../models
--output_reports_dir ../../outputs/reports
--output_figures_dir ../../outputs/figures

7. Using a pre-trained model: ...

8. Custom data splitting function: ...

"""

import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import sys

from datetime import datetime
from scipy.stats import loguniform, randint, uniform, rv_continuous, rv_discrete
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from typing import Dict, Any, Union, Optional, Type, Callable

import src.models.custom_linear_regressor
import src.data.load_exampledb
import src.data.load_m5salesdb
import src.data.preprocess_m5salesdb
import src.data.preprocess_passthrough
import src.data.split_m5salesdb
import src.data.split_train_val_test
import src.data.split_train_test
import src.data.split_passthrough
import src.eda.eda_m5salesdb
import src.eda.eda_passthrough
import src.evaluation.evaluate_exampledb
import src.evaluation.evaluate_passthrough
import src.features.features_passthrough
import src.features.features_m5salesdb
from src.optimization.custom_sk_validators import PredefinedSplit

from src.utils.my_os import ensure_dir_exists


# Dictionaries for mapping identifiers to strings representing sklearn or custom functions
MODELS: Dict[str, Type[BaseEstimator]] = {
    "sklearn_LinearRegression": LinearRegression,
    "sklearn_SVC": SVC,
    "sklearn_RandomForestRegressor": RandomForestRegressor,
    "mymodel": src.models.custom_linear_regressor.CustomModel,
}
DATA_LOADING_FNS: Dict[str, Callable] = {
    "load_exampledb": src.data.load_exampledb.load_data,
    "load_m5salesdb": src.data.load_m5salesdb.load_data,
}
PREPROCESSING_FNS: Dict[str, Callable] = {
    "preprocess_passthrough": src.data.preprocess_passthrough.preprocess_data,
    "preprocess_m5salesdb": src.data.preprocess_m5salesdb.preprocess_data,
}
EDA_FNS: Dict[str, Callable] = {
    "eda_passthrough": src.eda.eda_passthrough.eda,
    "eda_m5salesdb": src.eda.eda_m5salesdb.eda,
}
FEATURE_EXTRACTION_FNS: Dict[str, Callable] = {
    "features_passthrough": src.features.features_passthrough.extract_features,
    "features_m5salesdb": src.features.features_m5salesdb.extract_features,
}
SPLITTING_FNS: Dict[str, Callable] = {
    "split_passthrough": src.data.split_passthrough.split_data,
    "split_train_val_test": src.data.split_train_val_test.split_data,
    "split_train_test": src.data.split_train_test.split_data,
    "split_m5salesdb": src.data.split_m5salesdb.split_data,
}
RAND_DISTR_FNS: Dict[str, Type[Union[rv_continuous, rv_discrete]]] = {
    'loguniform': loguniform,
    'randint': randint,
    'uniform': uniform,
}
EVALUATION_FNS: Dict[str, Callable] = {
    "evaluate_passthrough": src.evaluation.evaluate_passthrough.evaluate,
    "evaluate_exampledb": src.evaluation.evaluate_exampledb.evaluate,
}


def check_split_args(split_fn: str, split_ratio: str, model: str) -> None:
    """"
    Check the consistency and validity of split function arguments.

    Args:
        split_fn (str): Identifier for the data splitting function.
        split_ratio (str): A string representing the ratio for splitting data, e.g., "80 20".
        model (str): Model identifier to check compatibility with the splitting strategy.

    Raises:
        ValueError: If any inconsistency or incompatibility is found among the arguments.
    """
    if split_fn not in SPLITTING_FNS.keys():
        raise ValueError(f"split_fn must be one among {SPLITTING_FNS.keys()}.")

    if (split_fn == "split_train_test" or split_fn == "split_train_val_test") and split_ratio is None:
        raise ValueError("split_ratio must be provided for splitting functions 'split_train_test' and 'split_train_val_test'.")

    if split_ratio is not None:
        # Check that split ratio is a string of numbers separated by spaces
        try:
            ratios = [float(num) for num in split_ratio.split()]
        except ValueError:
            raise ValueError("split_ratio must be a string of numbers separated by spaces.")
        # Check that the correct number of split_ratio(s) is provided for the desired split_fn
        if split_fn == "split_train_test" and len(ratios) != 2:
            raise ValueError("Two numbers should be given as split_ratios for splitting function 'split_train_test'.")
        elif split_fn == "split_train_val_test" and len(ratios) != 3:
            raise ValueError("Three numbers should be given as split_ratios for splitting function 'split_train_val_test'.")

    # Validation based on the chosen model's library
    model_module = MODELS.get(model).__module__
    # if "sklearn" in model_module:
    #     if split_fn == "split_train_val_test":
    #         raise ValueError("Splitting function 'split_train_val_test' incompatible with sklearn models.")
    #     if len(ratios) != 2:
    #         raise ValueError("Split ratio for sklearn models must contain exactly two numbers.")

    if "tensorflow" in model_module or "torch" in model_module:
        if split_fn == "split_train_test":
            raise ValueError("Splitting function 'split_train_test' incompatible with sklearn models.")
        if len(ratios) != 3:
            raise ValueError("Split ratio for TensorFlow/PyTorch models must contain exactly three numbers.")


def check_hparams_opt_args(model_hparams: Optional[str], split_fn: str, n_folds: Optional[int]) -> None:
    """
    Validates the format of model hyperparameters and checks their consistency with the data split function.
    This function first checks if the model hyperparameters are in the correct JSON format and then validates
    whether the values are appropriate (scalar or specified distribution strings for hyperparameter optimization).
    It also checks if the 'n_folds' argument is consistent with the chosen 'split_fn' based on whether hyperparameter
    optimization is necessary.

    Args:
        model_hparams (Optional[str]): JSON string of model hyperparameters.
        split_fn (str): Identifier for the data split function.
        n_folds (Optional[int]): Number of folds for k-fold cross-validation, if applicable.

    Raises:
        ValueError: If there's an inconsistency in hyperparameter optimization requirements.
        argparse.ArgumentTypeError: If the hyperparameters format is incorrect.
    """
    # First, validate the format of model hyperparameters

    if model_hparams is None:
        hparams = {}
    else:
        try:
            hparams = json.loads(model_hparams)
        except json.JSONDecodeError:
            raise argparse.ArgumentTypeError("Invalid JSON string for hyperparameters.")
        for key, value in hparams.items():
            if isinstance(value, str):
                if not re.match(r'^(uniform|loguniform|randint)\(\d+(\.\d+)?(e[+\-]?\d+)?, \d+(\.\d+)?(e[+\-]?\d+)?\)$', value):
                    raise argparse.ArgumentTypeError(
                        f"Invalid value for hyperparameter {key}: must be a specific distribution string.")
            elif not isinstance(value, (int, float)):
                raise argparse.ArgumentTypeError(f"Invalid value for hyperparameter {key}: must be a scalar.")

    # Check if hyperparameter optimization is necessary
    optimization_needed = any(isinstance(value, str) for value in hparams.values())

    # Conditions based on split function
    if optimization_needed:
        if split_fn == 'split_train_val_test' and n_folds is not None:
            raise ValueError("'n_folds' should not be specified for splitting function 'split_train_val_test'.")
        elif split_fn == 'split_train_test' and n_folds is None:
            raise ValueError("For 'split_train_test', 'n_folds' must be specified when doing hyperparameter optimization.")
        else:
            # Custom split function case, placeholder for further specifications
            pass

    else:  # Optimization is not needed
        if n_folds is not None:
            raise ValueError("Argument n_folds should not be specified when hyperparameter optimization is not required.")


def check_output_args(save_output: bool, output_data_dir: str, output_model_dir: str, output_reports_dir: str, output_figures_dir: str) -> None:
    """
    Verifies that all required output directories are specified when output saving is enabled.

    Args:
        save_output (bool): Flag indicating whether outputs should be saved.
        output_data_dir (str): Directory path to save processed data.
        output_model_dir (str): Directory path to save trained models.
        output_reports_dir (str): Directory path to save evaluation reports.
        output_figures_dir (str): Directory path to save generated figures.

    Raises:
        ValueError: If `save_output` is True and any directory path is not specified.
    """
    if save_output:
        output_dirs = [output_data_dir, output_model_dir, output_reports_dir, output_figures_dir]
        if any(d is None for d in output_dirs):
            parser.error("All output directories must be specified when --save_output is used")


def init_reload_model(args: argparse.Namespace) -> Any:
    """
    Initializes or reloads a model based on the provided command-line arguments.

    This function decides between initializing a new model with given hyperparameters or reloading an
    existing model from a specified file path. It supports model initialization for various types
    of machine learning models defined in the `MODELS` dictionary.

    Args:
        args (argparse.Namespace): The namespace object containing command-line arguments relevant to model initialization.

    Returns:
        model_instance (Any): The initialized or reloaded model instance. The specific type of this instance
                              depends on the model being loaded or initialized but is generally expected to
                              be a subclass of `sklearn.base.BaseEstimator`.

    Raises:
        ValueError: If both a new model identifier and a path to reuse a model are provided, or if the model
                    file for reuse is not found or cannot be loaded due to errors.
    """
    if args.reuse_model and args.model:
        raise ValueError("Specify either a model to train or a model to reuse, not both.")

    model = None
    if args.model:
        ModelClass = MODELS.get(args.model)
        # Extract and filter valid scalar hyperparameters
        if args.model_hparams is None:
            model = ModelClass()
        else:
            params = json.loads(args.model_hparams)
            valid_params = ModelClass().get_params().keys()
            scalar_hparams = {k: v for k, v in params.items()
                              if k in valid_params and isinstance(v, (int, float))}
            model = ModelClass(**scalar_hparams)

    else:  # reload serialized model (args.reuse_model is set)
        try:
            with open(args.reuse_model, 'rb') as file:
                model = pickle.load(file)
        except (FileNotFoundError, pickle.UnpicklingError):
            raise ValueError("Failed to load the specified model.")
    return model


def string_to_distribution(value: str) -> Union[rv_continuous, rv_discrete]:
    """
    Converts a string representation of a distribution into a SciPy distribution object.

    Args:
        value (str): String representation of the distribution, including the distribution
                     name and its parameters in parentheses.

    Returns:
        distr_obj (Union[rv_continuous, rv_discrete]): A SciPy distribution object corresponding
        to the input string. This could be either a continuous or a discrete distribution based on the
        input. Supports 'uniform', 'loguniform', and 'randint' distributions.
    """
    match = re.match(r'^(uniform|loguniform|randint)\(([^,]+), ([^)]+)\)$', value)
    func_name, arg1, arg2 = match.groups()
    arg1 = float(arg1)
    arg2 = float(arg2)
    distr_obj = RAND_DISTR_FNS[func_name](arg1, arg2)
    return distr_obj


def escape_quotes_in_curly_brackets(string: str) -> str:
    """
    Escapes double quotes inside curly brackets within a string.

    Args:
        string (str): The input string potentially containing curly brackets with unescaped double quotes.

    Returns:
        string (str): The modified string with double quotes inside curly brackets escaped.
    """
    curly_bracket_parts = re.findall(r'\{[^{}]*\}', string)
    modified_parts = [part.replace('"', r'\"') for part in curly_bracket_parts]
    for original_part, modified_part in zip(curly_bracket_parts, modified_parts):
        string = string.replace(original_part, modified_part)
    return string


def get_function_full_name(func):
    """
    Retrieves the full name of a function, including its module path.

    Args:
        func (Callable or str): The function object or a string representing a non-callable entity.

    Returns:
        func (str): The full name of the function including its module path if callable, or the
        original string if not callable.
    """
    if callable(func):
        return f"{func.__module__}.{func.__name__}"
    else:
        return func


def main(parsed_args: argparse.Namespace) -> None:
    """
    Main function to execute the machine learning workflow.

    This function dynamically loads data loading, preprocessing, feature extraction,
    and evaluation functions based on the provided arguments. It initializes or reloads
    the specified model, performs data splitting, model training (including hyperparameter
    optimization if specified), and evaluates the model.

    Parameters:
    parsed_args (argparse.Namespace): The namespace object containing command-line arguments.

    Note:
    The function handles different scenarios such as using sklearn models, TensorFlow models,
    reusing pre-trained models, and saving outputs including models, reports, and figures.
    """
    # Initialize logger
    output_reports_dir = None
    if parsed_args.save_output:
        output_reports_dir = os.path.join(parsed_args.output_reports_dir, f"report_{parsed_args.run_id}")
        ensure_dir_exists(output_reports_dir)
        log_fname = os.path.join(output_reports_dir, f"log_{parsed_args.run_id}.log")
        logging.basicConfig(
            level=parsed_args.log_level.upper(),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_fname,
            filemode='w',  # 'w' to write a new log file each time
        )
    else:
        logging.basicConfig(level=parsed_args.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Determine required functions
    logger.info("Initializing functions and model...")
    load_data_fn = DATA_LOADING_FNS.get(parsed_args.data_loading_fn)
    preprocess_fn = PREPROCESSING_FNS.get(parsed_args.preprocessing_fn)
    eda_fn = EDA_FNS.get(parsed_args.eda_fn)
    extract_features_fn = FEATURE_EXTRACTION_FNS.get(parsed_args.feature_extraction_fn)
    split_data_fn = SPLITTING_FNS.get(parsed_args.split_fn)
    evaluate_fn = EVALUATION_FNS.get(parsed_args.evaluation_fn)

    # Initialize model or reload existing one
    model = init_reload_model(parsed_args)

    if parsed_args.precomputed_features_path:
        # Load the pre-computed features
        X = pd.read_pickle(os.path.join(parsed_args.precomputed_features_path, "X.pkl"))
        Y = pd.read_pickle(os.path.join(parsed_args.precomputed_features_path, "Y.pkl"))

    else:
        # Load data
        logger.info("Loading data and extracting features...")
        sales, sell_prices, calendar = load_data_fn(parsed_args.data_path, debug=True)
        sales = preprocess_fn(sales, sell_prices, calendar)

        # # Exploratory data analysis
        # _ = eda_fn(sales)

        # preprocess, extract features
        X, Y = extract_features_fn(sales)
        if parsed_args.save_output:
            dbname = os.path.basename(parsed_args.data_path.rstrip('/'))  # Extracts filename from data_path without extension
            output_dir = os.path.join(parsed_args.output_data_dir, dbname)
            os.makedirs(output_dir, exist_ok=True)
            X.to_pickle(os.path.join(output_dir, "X.pkl"))
            Y.to_pickle(os.path.join(output_dir, "Y.pkl"))
        del sales, sell_prices, calendar

    # Split data
    if parsed_args.split_fn != "split_passthrough":
        logger.info("Computing data splits...")
        X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices = split_data_fn(X, Y)
    else:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices = src.data.split_train_test.split_data(X, Y, random_seed=0)  # Only here for completeness. Normally, it is expected that the loaded data is already split
        X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices = split_data_fn(X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices)  # This is correct as split_data_fn would be split_passthrough

    # Distinguish between sklearn and pytorch/tensorflow pipeline
    if "sklearn" in MODELS.get(parsed_args.model).__module__:
        # Placeholder for any additional data transformation
        data_transformer = FunctionTransformer()

        # Define an sklearn Pipeline
        pipeline = Pipeline([('transformer', data_transformer), ('model', model)])

        # Set up hyperparams optimization if there is any valid hyperparameter associated with a distribution string
        if parsed_args.model_hparams is None:
            optimization_needed = False
        else:
            valid_params = model.get_params().keys()
            distr_pattern = r'^(uniform|loguniform|randint)\(\d+(\.\d+)?, \d+(\.\d+)?\)$'
            param_distributions = {f"model__{k}": v for k, v in json.loads(parsed_args.model_hparams).items()
                                   if k in valid_params and isinstance(v, str) and re.match(distr_pattern, v)}
            for k, v in param_distributions.items():
                if isinstance(v, str):
                    param_distributions[k] = string_to_distribution(v)
            optimization_needed = bool(param_distributions)

        # Either perform hyperparams optimization with refit=True OR just fit the model
        if optimization_needed:
            # Determine whether to perform k-fold validation or rely on precomputed val set
            if X_val is None and Y_val is None and cv_indices is not None:
                cv = cv_indices
                n_folds = len(cv_indices)
                X_tmp = X_train
                Y_tmp = Y_train
            elif X_val is not None and Y_val is not None and cv_indices is None:
                n_folds = 1
                val_fold = [0] * len(X_train) + [1] * len(X_val)
                X_tmp = np.concatenate([X_train, X_val])
                Y_tmp = np.concatenate([Y_train, Y_val])
                cv = PredefinedSplit(test_fold=val_fold)
            else:
                raise ValueError("Only one is expected to be not None between cv_indices and (X_val, Y_val).")

            n_samplings = 5
            logger.info(f"Optimizing model hyperparameters ({n_samplings} samplings * {n_folds} folds and fitting the model...")
            search = RandomizedSearchCV(pipeline,
                                        param_distributions,
                                        n_iter=n_samplings,
                                        refit=True, cv=cv,
                                        random_state=parsed_args.random_seed,
                                        return_train_score=True,
                                        verbose=3)  # return_train_score may slow down the execution
            search = search.fit(np.squeeze(np.asarray(X_tmp)), np.squeeze(np.asarray(Y_tmp)))
            if hasattr(search, "cv_results_"):
                cv_results_df = pd.DataFrame(search.cv_results_)
                float_columns = cv_results_df.select_dtypes(include=['float']).columns
                cv_results_df[float_columns] = cv_results_df[float_columns].round(4)
            else:
                cv_results_df = None
            if hasattr(search, 'best_estimator_'):
                model = search.best_estimator_
            else:
                raise AttributeError("Failed to find the best estimator. The model fitting process did not complete successfully.")

        else:
            logger.info("Fitting the model...")
            model = model.fit(X_train.to_numpy(), np.squeeze(Y_train.to_numpy()))
            cv_results_df = None

        # Compute model predictions
        logger.info("Computing model predictions...")
        Y_pred = model.predict(X_test.to_numpy())
        Y_train_pred = model.predict(X_train.to_numpy())

    elif "tensorflow" in MODELS[parsed_args.model]:
        optimization_needed = True
        Y_pred = None
        Y_train_pred = None
        cv_results_df = None
    elif "torch" in MODELS[parsed_args.model]:
        optimization_needed = True
        Y_pred = None
        Y_train_pred = None
        cv_results_df = None
    else:
        optimization_needed = True
        Y_pred = None
        Y_train_pred = None
        cv_results_df = None

    # Evaluate model predictions
    logger.info("Evaluating model predictions...")
    scores, figs = evaluate_fn(
        np.squeeze(Y_test.to_numpy()),
        np.squeeze(Y_pred),
        model,
        Y_test.columns.tolist(),
        np.squeeze(Y_train.to_numpy()),
        np.squeeze(Y_train_pred),
    )

    # Save results
    if parsed_args.save_output:
        logger.info("Storing results...")
        ensure_dir_exists(parsed_args.output_data_dir)
        model_summaries_dir = os.path.join(parsed_args.output_model_dir, "model_summaries")
        ensure_dir_exists(model_summaries_dir)
        trained_models_dir = os.path.join(parsed_args.output_model_dir, "trained_models")
        ensure_dir_exists(trained_models_dir)
        # output_reports_dir = os.path.join(parsed_args.output_reports_dir, f"report_{parsed_args.run_id}")  # Done before
        # ensure_dir_exists(output_reports_dir)  # Done before
        output_figures_dir = os.path.join(parsed_args.output_reports_dir, f"report_{parsed_args.run_id}")
        ensure_dir_exists(parsed_args.output_figures_dir)

        model_path = os.path.join(trained_models_dir, f"trained_model_{parsed_args.run_id}.pkl")
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

        if cv_results_df is not None:
            cv_results_path = os.path.join(model_summaries_dir, f"cv_results_{parsed_args.run_id}.csv")
            cv_results_df.to_csv(cv_results_path, index=False)

        report_path = os.path.join(output_reports_dir, f"scores_{parsed_args.run_id}.csv")
        scores.to_csv(report_path, index=False)

        for fig_name, fig in figs.items():
            fig_path = os.path.join(output_figures_dir, f"{fig_name}_{parsed_args.run_id}.png")
            fig.savefig(fig_path)
            plt.close(fig)  # Close the figure after saving to free up memory

        # Collect analysis information in readable form
        script_call = "python " + " ".join(sys.argv)
        script_call = script_call.replace("\\", "/")
        script_call = script_call.replace(f'--split_ratio {parsed_args.split_ratio}', f'--split_ratio "{parsed_args.split_ratio}"')
        script_call = script_call.replace(f'--model_hparams {parsed_args.model_hparams}', f'--model_hparams "{parsed_args.model_hparams}"')
        script_call = escape_quotes_in_curly_brackets(script_call)

        # Convert relative paths to absolute paths
        matches = re.findall(re.compile(r'\.\.?/[^ ]+'), script_call)
        for match in matches:
            abs_path = os.path.abspath(match)
            script_call = script_call.replace(match, abs_path)

        # Store information as a text file. Do not use JSON because it messes up with necessary escapes.
        experiment_info = {
            "run_id": parsed_args.run_id,
            "data_path": parsed_args.data_path,
            "dataset_shape": str(X.shape),
            "preprocessing_function": get_function_full_name(PREPROCESSING_FNS.get(parsed_args.preprocessing_fn, "Not Applicable")),
            "feature_extraction_function": get_function_full_name(FEATURE_EXTRACTION_FNS.get(parsed_args.feature_extraction_fn, "Not Applicable")),
            "model_reused": parsed_args.reuse_model if parsed_args.reuse_model else "False",
            "model_type": str(type(model)),
            "initial_hyperparameters": json.loads(parsed_args.model_hparams) if parsed_args.model_hparams else "None",
            "final_hyperparameters": model.get_params() if hasattr(model, 'get_params') else "Not Applicable",
            "splitting_function": get_function_full_name(SPLITTING_FNS.get(parsed_args.split_fn, "Not Applicable")),
            "split_ratio": parsed_args.split_ratio,
            "optimization_performed": optimization_needed,
            "n_folds": parsed_args.n_folds if parsed_args.n_folds is not None else "Not Applicable",
            "evaluation_function": get_function_full_name(EVALUATION_FNS.get(parsed_args.evaluation_fn, "Not Applicable")),
            "performance_metrics": scores.index.tolist(),
            "random_seed": parsed_args.random_seed,
            "script_call": script_call,
            "additional_information": "False",
        }
        with open(os.path.join(output_reports_dir, f"experiment_details_{parsed_args.run_id}.txt"), "w") as file:
            for key, value in experiment_info.items():
                file.write(f"{key}: {value}\n\n")

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Script")
    parser.add_argument('--data_path', required=True, help='Path to the data file')
    parser.add_argument('--precomputed_features_path', type=str, help='Path to pre-computed features to skip loading, preprocessing, and feature extraction')
    parser.add_argument('--data_loading_fn', required=True, help='Function identifier for loading data')
    parser.add_argument('--model', choices=MODELS.keys(), help='Model identifier')
    parser.add_argument('--model_hparams', default=None, help='JSON string of model hyperparameters')
    parser.add_argument('--reuse_model', help='Path to a pre-trained model to reuse')
    parser.add_argument('--preprocessing_fn', default='preprocess_passthrough', choices=PREPROCESSING_FNS.keys(), help='Identifier for preprocessing function')
    parser.add_argument('--eda_fn', default='eda_passthrough', choices=EDA_FNS.keys(), help='Identifier for exploratory data analysis function')
    parser.add_argument('--feature_extraction_fn', default='features_passthrough', choices=FEATURE_EXTRACTION_FNS.keys(), help='Identifier for feature extraction function')
    parser.add_argument('--split_fn', default='split_passthrough', help='Identifier for data split function')
    parser.add_argument('--split_ratio', type=str, help='Ratio for splitting data')
    parser.add_argument('--n_folds', type=int, help='Number of folds for k-fold cross-validation')
    parser.add_argument('--stratified_kfold', action='store_true', help='Whether to perform stratified (for clf) or standard (for reg or clf) k-fold cross-validation')
    parser.add_argument('--evaluation_fn', default='evaluate_passthrough', choices=EVALUATION_FNS.keys(), help='Identifier for evaluation function')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (e.g., "INFO", "DEBUG")')
    parser.add_argument('--random_seed', type=int, default=None, help='Seed for random number generators for reproducibility')
    parser.add_argument('--run_id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help='Unique identifier for the run')
    parser.add_argument('--save_output', action='store_true', help='Save outputs (data, models, reports, figures)')
    parser.add_argument('--output_data_dir', type=str, help='Directory to save processed data')
    parser.add_argument('--output_model_dir', type=str, help='Directory to save trained models')
    parser.add_argument('--output_reports_dir', type=str, help='Directory to save evaluation reports')
    parser.add_argument('--output_figures_dir', type=str, help='Directory to save generated figures')
    parsed_args = parser.parse_args()

    # Additional consistency check
    check_split_args(parsed_args.split_fn, parsed_args.split_ratio, parsed_args.model)
    check_hparams_opt_args(parsed_args.model_hparams, parsed_args.split_fn, parsed_args.n_folds)
    check_output_args(parsed_args.save_output, parsed_args.output_data_dir, parsed_args.output_model_dir, parsed_args.output_reports_dir, parsed_args.output_figures_dir)

    main(parsed_args)
