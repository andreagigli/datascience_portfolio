"""
This script is designed for flexible machine learning workflows. It allows for dynamic loading of data, preprocessing, feature extraction, model training, and evaluation based on specified function identifiers. Users can specify model parameters, choose to train a new model or use a pre-trained one, and control data splitting for training and testing.

Example shell calls:

1. Basic usage with required arguments:
python analysis_exampledb.py
--data_path ../../data/external/exampledb/california_housing.csv
--data_loading_fn load_exampledb
--model sklearn_RandomForestRegressor
--model_hparams "{\"n_estimators\": 100, \"max_depth\": 10}"
--preprocessing_fn preprocess_exampledb
--feature_extraction_fn feature_exampledb
--split_fn split_train_test
--split_ratio "80 20"
--evaluation_fn evaluate_exampledb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ../../data/processed
--output_model_dir ../../models
--output_reports_dir ../../outputs/reports
--output_figures_dir ../../outputs/figures

2. Usage with hyperparameters optimization:
python analysis_exampledb.py
--data_path ../../data/external/exampledb/california_housing.csv
--data_loading_fn load_exampledb
--model sklearn_RandomForestRegressor
--model_hparams "{\"n_estimators\": \"randint(10, 100)\", \"max_depth\": 5}"
--preprocessing_fn preprocess_exampledb
--feature_extraction_fn feature_exampledb
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

3. Usage with tensorflow model
python analysis_exampledb.py
--data_path ../../data/external/exampledb/california_housing.csv
--data_loading_fn load_exampledb
--model tensorflow_mynet
--preprocessing_fn preprocess_exampledb
--feature_extraction_fn feature_exampledb
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

3. Using a pre-trained model: ...

4. Custom data splitting function: ...

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
import src.data.preprocess_exampledb
import src.features.features_exampledb
import src.data.split_train_val_test
import src.data.split_train_test
import src.data.split_exampledb
import src.evaluation.evaluate_exampledb

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
}
PREPROCESSING_FNS: Dict[str, Callable] = {
    "preprocess_exampledb": src.data.preprocess_exampledb.preprocess_data,
}
FEATURE_EXTRACTION_FNS: Dict[str, Callable] = {
    "feature_exampledb": src.features.features_exampledb.extract_features,
}
SPLITTING_FNS: Dict[str, Callable] = {
    "split_train_val_test": src.data.split_train_val_test.split_data,
    "split_train_test": src.data.split_train_test.split_data,
    "split_exampledb": src.data.split_exampledb.split_data,
}
RAND_DISTR_FNS: Dict[str, Type[Union[rv_continuous, rv_discrete]]] = {
    'loguniform': loguniform,
    'randint': randint,
    'uniform': uniform,
}
EVALUATION_FNS: Dict[str, Callable] = {
    "evaluate_exampledb": src.evaluation.evaluate_exampledb.evaluate,
}


def check_split_args(split_fn: str, split_ratio: str, model: str) -> None:
    """
    Check the consistency and validity of split function arguments.

    Parameters:
    split_fn (str): The identifier for the data splitting function.
    split_ratio (str): A string representing the ratio for splitting data.
    n_folds (int): Number of folds for k-fold cross-validation.

    Raises:
    ValueError: If any inconsistency is found in the arguments.
    """
    # Check that split_ratio is a string of numbers separated by spaces
    try:
        ratios = [float(num) for num in split_ratio.split()]
    except ValueError:
        raise ValueError("split_ratio must be a string of numbers separated by spaces.")

    # Validation based on the chosen model's library
    model_module = MODELS.get(model).__module__
    if "sklearn" in model_module:
        if split_fn == "split_train_val_test":
            raise ValueError("Splitting function 'split_train_val_test' incompatible with sklearn models.")
        if len(ratios) != 2:
            raise ValueError("Split ratio for sklearn models must contain exactly two numbers.")
        # if not (n_folds and isinstance(n_folds, int)):
        #     raise ValueError("Number of folds for sklearn models must be an integer.")

    elif "tensorflow" in model_module or "torch" in model_module:
        if split_fn == "split_train_test":
            raise ValueError("Splitting function 'split_train_test' incompatible with sklearn models.")
        if len(ratios) != 3:
            raise ValueError("Split ratio for TensorFlow/PyTorch models must contain exactly three numbers.")
        # if n_folds is not None:
        #     raise ValueError("Number of folds should not be specified for TensorFlow/PyTorch models.")
    else:
        raise ValueError("Unknown model type.")


def check_hparams_opt_args(model_hparams: str, split_fn: str, n_folds: Optional[int]) -> None:
    """
    Validates the format of model hyperparameters and checks their consistency with the data split function.
    This function first checks if the model hyperparameters are in the correct JSON format and then validates
    whether the values are appropriate (scalar or specified distribution strings for hyperparameter optimization).
    It also checks if the 'n_folds' argument is consistent with the chosen 'split_fn' based on whether hyperparameter
    optimization is necessary.

    Parameters:
    model_hparams (str): JSON string of model hyperparameters.
    split_fn (str): Identifier for the data split function.
    n_folds (Optional[int]): Number of folds for k-fold cross-validation, if applicable.

    Raises:
    ValueError: If there is inconsistency between hyperparameters and split function requirements.
    argparse.ArgumentTypeError: If the hyperparameters are not in a valid JSON format or if the values are not
                                valid scalars or specified distribution strings for hyperparameter optimization.
    """
    # First, validate the format of model hyperparameters
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
            raise ValueError("For 'split_train_val_test', 'n_folds' must not be specified when doing hyperparameter optimization.")
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
    Checks whether all output directories are specified when saving output is enabled.

    Parameters:
    save_output (bool): Flag indicating whether to save outputs.
    output_data_dir (str): Directory to save processed data.
    output_model_dir (str): Directory to save trained models.
    output_reports_dir (str): Directory to save evaluation reports.
    output_figures_dir (str): Directory to save generated figures.

    Raises:
    ValueError: If `save_output` is True and any of the directory arguments is None.
    """
    if save_output:
        output_dirs = [output_data_dir, output_model_dir, output_reports_dir, output_figures_dir]
        if any(d is None for d in output_dirs):
            parser.error("All output directories must be specified when --save_output is used")


def init_reload_model(args: argparse.Namespace) -> Any:
    """
    Initializes a new model or reloads a pre-trained model based on the provided arguments.

    This function either creates a new model instance using the specified model identifier
    and hyperparameters or loads a model from a given file path.

    Parameters:
    args (argparse.Namespace): The namespace object containing command-line arguments.

    Returns:
    Any: The initialized or reloaded model object.

    Raises:
    ValueError: If both a new model and a model to reuse are specified.
    FileNotFoundError, pickle.UnpicklingError: If the model file for reuse is not found or cannot be loaded.
    """
    if args.reuse_model and args.model:
        raise ValueError("Specify either a model to train or a model to reuse, not both.")

    model = None
    if args.model:
        ModelClass = MODELS.get(args.model)
        # Extract and filter valid scalar hyperparameters
        scalar_hparams = None
        if args.model_hparams:
            params = json.loads(args.model_hparams)
            valid_params = ModelClass().get_params().keys()
            scalar_hparams = {k: v for k, v in params.items()
                              if k in valid_params and isinstance(v, (int, float))}
        model = ModelClass(**scalar_hparams)
    elif args.reuse_model:  # reload serialized model
        try:
            with open(args.reuse_model, 'rb') as file:
                model = pickle.load(file)
        except (FileNotFoundError, pickle.UnpicklingError):
            raise ValueError("Failed to load the specified model.")
    return model


def string_to_distribution(value: str) -> Union[rv_continuous, rv_discrete]:
    """
    Converts a validated string representation of a distribution into a SciPy distribution object.

    This function assumes that the input string is already validated to match the pattern
    'distribution_name(arg1, arg2)', where 'distribution_name' is one of 'uniform',
    'loguniform', or 'randint', and 'arg1' and 'arg2' are numerical arguments for the
    distribution constructor. This validation is performed elsewhere in the code prior
    to calling this function.

    Parameters:
    value (str): A validated string representing the distribution and its parameters.

    Returns:
    Union[rv_continuous, rv_discrete]: A SciPy distribution object corresponding to the
    specified input string. This can be either a continuous or a discrete distribution.
    """
    match = re.match(r'^(uniform|loguniform|randint)\(([^,]+), ([^)]+)\)$', value)
    func_name, arg1, arg2 = match.groups()
    arg1 = float(arg1)
    arg2 = float(arg2)
    distr_fn = RAND_DISTR_FNS[func_name](arg1, arg2)
    return distr_fn


def escape_quotes_in_curly_brackets(string: str) -> str:
    """
    Escapes double quotes inside curly brackets in a given string.

    Note: In Python's output, escaped characters (e.g., \") are shown with double backslashes (\\") for clarity.
    However, when written to a file or used in another context, only the intended escape sequence (\")
    appears, with a single backslash.

    Args:
    string (str): The string in which the escaping should be performed.

    Returns:
    str: The modified string with escaped quotes inside curly brackets.
    """
    curly_bracket_parts = re.findall(r'\{[^{}]*\}', string)
    modified_parts = [part.replace('"', r'\"') for part in curly_bracket_parts]
    for original_part, modified_part in zip(curly_bracket_parts, modified_parts):
        string = string.replace(original_part, modified_part)
    return string


def get_function_full_name(func):
    """
    Retrieve the full name of a function, including its module path.

    This function takes a function object as an argument and returns its complete name in the format 'module_name.function_name'.
    If the argument is not a callable function (e.g., a string like "Not Applicable"), it returns the argument as it is.

    Parameters:
    func (Callable or str): The function object for which the full name is required, or a string representing a non-callable entity.

    Returns:
    str: The full name of the function including its module path if it's callable, or the original string if it's not callable.
    """
    if callable(func):
        return f"{func.__module__}.{func.__name__}"
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
    extract_features_fn = FEATURE_EXTRACTION_FNS.get(parsed_args.feature_extraction_fn)
    split_data_fn = SPLITTING_FNS.get(parsed_args.split_fn)
    evaluate_fn = EVALUATION_FNS.get(parsed_args.evaluation_fn)

    # Initialize model or reload existing one
    model = init_reload_model(parsed_args)

    # Load, preprocess, extract features
    logger.info("Loading data and extracting features...")
    X, Y = load_data_fn(parsed_args.data_path)
    X, Y = preprocess_fn(X, Y)
    X, Y = extract_features_fn(X, Y)

    # Split data
    logger.info("Computing data splits...")
    split_ratios = [float(el) for el in parsed_args.split_ratio.split()]
    additional_args = [parsed_args.n_folds,
                       parsed_args.stratified_kfold] if parsed_args.split_fn == 'split_train_test' else []
    split_data_fn_args = split_ratios + additional_args
    X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices = split_data_fn(X, Y, parsed_args.random_seed, *split_data_fn_args)

    # Distinguish between sklearn and pytorch/tensorflow pipeline
    if "sklearn" in MODELS.get(parsed_args.model).__module__:
        # Placeholder for any additional data transformation
        data_transformer = FunctionTransformer()

        # Define an sklearn Pipeline
        pipeline = Pipeline([('transformer', data_transformer), ('model', model)])

        # Set up hyperparams optimization if there is any valid hyperparameter associated with a distribution string
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
            n_samplings = 5
            logger.info(f"Optimizing model hyperparameters ({n_samplings} samplings * {len(cv_indices)} folds and fitting the model...")
            search = RandomizedSearchCV(pipeline, param_distributions, n_iter=n_samplings, refit=True, cv=cv_indices,
                                        random_state=parsed_args.random_seed, return_train_score=True, verbose=3)  # return_train_score may slow down the execution
            search = search.fit(np.squeeze(X_train.to_numpy()), np.squeeze(Y_train.to_numpy()))
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
    parser.add_argument('--data_loading_fn', required=True, help='Function identifier for loading data')
    parser.add_argument('--model', choices=MODELS.keys(), help='Model identifier')
    parser.add_argument('--model_hparams', help='JSON string of model hyperparameters')
    parser.add_argument('--reuse_model', help='Path to a pre-trained model to reuse')
    parser.add_argument('--preprocessing_fn', required=True, choices=PREPROCESSING_FNS.keys(), help='Identifier for preprocessing function')
    parser.add_argument('--feature_extraction_fn', required=True, choices=FEATURE_EXTRACTION_FNS.keys(), help='Identifier for feature extraction function')
    parser.add_argument('--split_fn', required=True, choices=SPLITTING_FNS.keys(), help='Identifier for data split function')
    parser.add_argument('--split_ratio', required=True, type=str, help='Ratio for splitting data')
    parser.add_argument('--n_folds', type=int, help='Number of folds for k-fold cross-validation')
    parser.add_argument('--stratified_kfold', type=bool, default=False, help='Whether to perform stratified (for clf) or standard (for reg or clf) k-fold cross-validation')
    parser.add_argument('--evaluation_fn', required=True, choices=EVALUATION_FNS.keys(), help='Identifier for evaluation function')
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
