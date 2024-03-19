"""
This script is designed for flexible machine learning workflows.
It allows for dynamic loading of data, preprocessing, eda, feature extraction, model hparam optimization, model training, inference and evaluation based on specified function identifiers.
Users can specify model parameters, choose to train a new model or use a pre-trained one, control data splitting for training and testing, and whether to save the outputs or not.

Example shell calls:

PREFERRED: using LGBMRegressor (handles large datasets, without need to one-hot encode cathegorical variables)
python analysis_exampledb.py
--data_path ../../data/external/m5salesdb/
--data_loading_fn load_m5salesdb
--model sklearn_compatible_LGBMRegressor
--hparams "{\"sklearn_compatible_LGBMRegressor__num_leaves\": \"randint(20, 200)\", \"sklearn_compatible_LGBMRegressor__learning_rate\": \"loguniform(0.001, 1)\", \"sklearn_compatible_LGBMRegressor__n_estimators\": 1000}"
--hopt_n_rndcv_samplings 5
--hopt_subsampling_fn subsample_train_m5salesdb
--hopt_subsampling_rate 1.0
--preprocessing_fn preprocess_m5salesdb
--eda_fn eda_m5salesdb
--feature_extraction_fn features_m5salesdb
--split_fn split_m5salesdb
--prediction_fn predict_m5salesdb
--look_back_days_sequential_prediction 380
--evaluation_fn evaluate_m5salesdb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ../../data/processed/
--output_model_dir ../../models/
--output_reports_dir ../../outputs/reports/
--output_figures_dir ../../outputs/figures/

IF FEATURES WERE PRECOMPUTED:
python analysis_exampledb.py
--precomputed_features_path ../../data/processed/m5salesdb_debug_100/
--model sklearn_compatible_LGBMRegressor
--hparams "{\"sklearn_compatible_LGBMRegressor__num_leaves\": \"randint(20, 200)\", \"sklearn_compatible_LGBMRegressor__learning_rate\": \"loguniform(0.001, 1)\", \"sklearn_compatible_LGBMRegressor__n_estimators\": 1000}"
--hopt_n_rndcv_samplings 5
--hopt_subsampling_fn subsample_train_m5salesdb
--hopt_subsampling_rate 1.0
--preprocessing_fn preprocess_m5salesdb
--eda_fn eda_m5salesdb
--feature_extraction_fn features_m5salesdb
--split_fn split_m5salesdb
--prediction_fn predict_m5salesdb
--look_back_days_sequential_prediction 380
--evaluation_fn evaluate_m5salesdb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ../../data/processed/
--output_model_dir ../../models/
--output_reports_dir ../../outputs/reports/
--output_figures_dir ../../outputs/figures/

python analysis_exampledb.py
--precomputed_features_path ../../data/processed/m5salesdb_debug_100/
--model sklearn_Ridge
--data_transformers sklearn_RBFSampler
--hparams "{\"sklearn_RBFSampler__n_components\": 1000, \"sklearn_RBFSampler__gamma\": \"loguniform(0.001, 10)\", \"sklearn_Ridge__alpha\": \"loguniform(0.00001, 1)\", }"
--hopt_n_rndcv_samplings 5
--hopt_subsampling_fn subsample_train_m5salesdb
--hopt_subsampling_rate 1.0
--preprocessing_fn preprocess_m5salesdb
--eda_fn eda_m5salesdb
--feature_extraction_fn features_m5salesdb
--split_fn split_m5salesdb
--prediction_fn predict_m5salesdb
--look_back_days_sequential_prediction 380
--evaluation_fn evaluate_m5salesdb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ../../data/processed/
--output_model_dir ../../models/
--output_reports_dir ../../outputs/reports/
--output_figures_dir ../../outputs/figures/

"""
import argparse
import inspect
import json
import logging
import os
import pickle
import re
import sys
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.sparse import csr_matrix
from scipy.stats import loguniform, randint, uniform, rv_continuous, rv_discrete
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

import src.data.data_m5salesdb
import src.data.load_exampledb
import src.data.split_train_test
import src.data.split_train_val_test
import src.eda.eda_m5salesdb
import src.evaluation.evaluate_exampledb
import src.evaluation.evaluate_m5salesdb
import src.features.features_exampledb
import src.features.features_m5salesdb
import src.models.custom_linear_regressor
import src.prediction.predict_m5salesdb
from src.eda.eda_misc import plot_correlation_heatmap, plot_pairwise_scatterplots
from src.optimization.custom_sk_validators import PredefinedSplit
from src.utils.my_dataframe import convert_df_to_sparse_matrix
from src.utils.my_os import ensure_dir_exists


# Dictionaries for mapping identifiers to strings representing sklearn or custom functions
MODELS: Dict[str, Type[BaseEstimator]] = {
    "sklearn_LinearRegression": LinearRegression,
    "sklearn_Ridge": Ridge,
    "sklearn_SVC": SVC,
    "sklearn_RandomForestRegressor": RandomForestRegressor,
    "sklearn_HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "sklearn_compatible_LGBMRegressor": LGBMRegressor,
    "mymodel": src.models.custom_linear_regressor.CustomModel,
}
DATA_TRANSFORMERS: Dict[str, Type[Union[TransformerMixin, BaseEstimator]]] = {
    "sklearn_RBFSampler": RBFSampler,
    "sklearn_StandardScaler": StandardScaler,
    "sklearn_MinMaxScaler": MinMaxScaler,
}
DATA_LOADING_FNS: Dict[str, Callable] = {
    "load_exampledb": src.data.load_exampledb.load_data,
    "load_m5salesdb": src.data.data_m5salesdb.load_data,
}
PREPROCESSING_FNS: Dict[str, Callable] = {
    "preprocess_passthrough": lambda *args, **kwargs: (args, kwargs) if kwargs else args,
    "preprocess_m5salesdb": src.data.data_m5salesdb.preprocess_data,
}
EDA_FNS: Dict[str, Callable] = {
    "eda_passthrough": lambda *args, **kwargs:  None,
    "eda_m5salesdb": src.eda.eda_m5salesdb.eda,
}
FEATURE_EXTRACTION_FNS: Dict[str, Callable] = {
    "features_exampledb": src.features.features_exampledb.extract_features,
    "features_m5salesdb": src.features.features_m5salesdb.extract_features,
}
SPLITTING_FNS: Dict[str, Callable] = {
    "split_passthrough": lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], {}),
    "split_train_val_test": src.data.split_train_val_test.split_data,
    "split_train_test": src.data.split_train_test.split_data,
    "split_m5salesdb": src.data.data_m5salesdb.split_data,
}
RAND_DISTR_FNS: Dict[str, Type[Union[rv_continuous, rv_discrete]]] = {
    'loguniform': loguniform,
    'randint': randint,
    'uniform': uniform,
}
HOPT_SUBSAMPLING_FNS: Dict[str, Callable] = {
    "subsampling_passthrough": lambda X, Y, **kwargs: (X, Y, kwargs.get('cv_indices', None)),
    "subsample_train_m5salesdb": src.data.data_m5salesdb.subsample_items,
}
PREDICTION_FNS: Dict[str, Callable] = {
    "predict_zeros": lambda model, X_test, Y_test, X_train, Y_train, *args, **kwargs: (np.zeros_like(Y_test), np.zeros_like(Y_train), None),  # Note: complex signature for consistency across prediction_fns
    "predict_sklearn": lambda model, X_test, Y_test, X_train, Y_train, *args, **kwargs: (model.predict(X_test), model.predict(X_train), None),
    "predict_m5salesdb": src.prediction.predict_m5salesdb.predict,
}
EVALUATION_FNS: Dict[str, Callable] = {
    "evaluate_passthrough": lambda *args, **kwargs: (pd.DataFrame(), {}),
    "evaluate_exampledb": src.evaluation.evaluate_exampledb.evaluate,
    "evaluate_m5salesdb": src.evaluation.evaluate_m5salesdb.evaluate,
}


def check_load_args(data_path: str, data_loading_fn: str, precomputed_features_path: str) -> None:
    """
    Ensures that either data_path with data_loading_fn or precomputed_features_path is provided, but not both or neither.

    Args:
        data_path: The path to the raw data.
        data_loading_fn: The function identifier used for loading raw data.
        precomputed_features_path: The path to precomputed features.

    Raises:
        ValueError: If neither data_path with data_loading_fn nor precomputed_features_path is provided, or if both are provided.
    """
    if bool(data_path and data_loading_fn) == bool(precomputed_features_path):
        raise ValueError(
            "You must specify either --data_path with --data_loading_fn OR --precomputed_features_path, but not both or neither.")


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


def check_hparams_opt_args(hparams: Optional[str], split_fn: str, n_folds: Optional[int]) -> None:
    """
    Validates the format of model hyperparameters and checks their consistency with the data split function.
    This function first checks if the model hyperparameters are in the correct JSON format and then validates
    whether the values are appropriate (scalar or specified distribution strings for hyperparameter optimization).
    It also checks if the 'n_folds' argument is consistent with the chosen 'split_fn' based on whether hyperparameter
    optimization is necessary.

    Args:
        hparams (Optional[str]): JSON string of hyperparameters.
        split_fn (str): Identifier for the data split function.
        n_folds (Optional[int]): Number of folds for k-fold cross-validation, if applicable.

    Raises:
        ValueError: If there's an inconsistency in hyperparameter optimization requirements.
        argparse.ArgumentTypeError: If the hyperparameters format is incorrect.
    """
    # First, validate the format of model hyperparameters

    if hparams is None:
        hparams = {}
    else:
        try:
            hparams = json.loads(hparams)
        except json.JSONDecodeError:
            raise argparse.ArgumentTypeError("Invalid JSON string for hyperparameters.")
        for key, value in hparams.items():
            if isinstance(value, str):
                if not re.match(r'^(uniform|loguniform|randint)\(\d+(\.\d+)?(e[+\-]?\d+)?, \d+(\.\d+)?(e[+\-]?\d+)?\)$', value):
                    raise argparse.ArgumentTypeError(
                        f"Invalid value for hyperparameter {key}: must be a specific distribution string.")
            elif not isinstance(value, (int, float, bool)):
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


def init_reload_model(parsed_args: argparse.Namespace) -> Any:
    """
    Initializes or reloads a model based on the provided command-line arguments.

    Args:
        parsed_args (argparse.Namespace): The namespace object containing command-line arguments relevant to model initialization.

    Returns:
        model_instance (Any): The initialized or reloaded model instance.

    Raises:
        ValueError: If both a new model identifier and a path to reuse a model are provided, or if the model
                    file for reuse is not found or cannot be loaded due to errors.
    """
    if parsed_args.reuse_model and parsed_args.model:
        raise ValueError("Specify either a model to train or a model to reuse, not both.")

    if parsed_args.model:
        ModelClass = MODELS.get(parsed_args.model)
        if parsed_args.hparams is None:
            # Initialize the default model if no hyperparamenters are given as argument
            model = ModelClass()
        else:
            # Parse the given hyperparameters to see if there is any valid one for the model
            all_params = json.loads(parsed_args.hparams)
            model_prefix = parsed_args.model + "__"  # e.g., "sklearn_Ridge__"

            # Filter parameters specific to and valid for the chosen model, stripping the model name prefix
            valid_params = inspect.signature(ModelClass.__init__).parameters

            fixed_hparams = {param_name.split("__")[1]: value for param_name, value in all_params.items()
                             if param_name.startswith(model_prefix)
                             and param_name.split("__")[1] in valid_params
                             and isinstance(value, (int, float, bool))  # TODO: Currently all string values are discarded to avoid distribution string, but in the future there may be the need to pass argument with a string value.
                             }

            # If model accepts a random_state and parsed_args.random_seed is provided, add it
            if 'random_state' in valid_params and parsed_args.random_seed is not None:
                fixed_hparams['random_state'] = parsed_args.random_seed

            model = ModelClass(**fixed_hparams)

    else:  # Reload serialized model (args.reuse_model is set)
        try:
            with open(parsed_args.reuse_model, 'rb') as file:
                model = pickle.load(file)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            raise ValueError(f"Failed to load the specified model due to: {str(e)}")
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
    aux_split_params = {}
    hopt_subsampling_fn = HOPT_SUBSAMPLING_FNS.get(parsed_args.hopt_subsampling_fn)
    predict_fn = PREDICTION_FNS.get(parsed_args.prediction_fn)
    aux_predict_params = {}
    evaluate_fn = EVALUATION_FNS.get(parsed_args.evaluation_fn)
    aux_eval_params = {}

    # Initialize model or reload existing one
    model = init_reload_model(parsed_args)

    # Save the extract_feature_fn into the predict parameters as it may be necessary for some custom prediction function, such as sequential prediction
    aux_predict_params["extract_features_fn"] = extract_features_fn

    # Load data and extract features, or reload previously computed features
    if parsed_args.precomputed_features_path is None:
        # Load data
        logger.info("Loading data and extracting features...")
        dataset = load_data_fn(parsed_args.data_path, debug=False)  # Set debug=True to select a subset of the items (faster computation)
        if not isinstance(dataset, tuple):
            dataset = (dataset,)

        # Preprocess data
        dataset = preprocess_fn(*dataset)
        if not isinstance(dataset, tuple):
            dataset = (dataset,)

        # # Exploratory data analysis
        # eda_fn(*dataset)

        # Extract features
        X, Y = extract_features_fn(*dataset)  # X and Y are expected to be pd.DataFrame

        if parsed_args.save_output:
            logger.info("Storing computed features...")
            dbname = os.path.basename(parsed_args.data_path.rstrip('/'))  # Extracts filename from data_path without extension
            output_dir = os.path.join(parsed_args.output_data_dir, dbname)
            os.makedirs(output_dir, exist_ok=True)
            X.to_pickle(os.path.join(output_dir, "X.pkl"))
            Y.to_pickle(os.path.join(output_dir, "Y.pkl"))
        del dataset

    else:
        # Reload the pre-computed features
        X = pd.read_pickle(os.path.join(parsed_args.precomputed_features_path, "X.pkl"))
        Y = pd.read_pickle(os.path.join(parsed_args.precomputed_features_path, "Y.pkl"))

    # Explore relationships within features and between features and targets
    plot_correlation_heatmap(X, Y, sample_size=1000, method='pearson')
    plot_correlation_heatmap(X, Y, sample_size=1000, method='spearman')
    # columns_to_plot = ["sold", "sell_price", "wday", "sold_robustlag_7", "sold_next_day"]
    # plot_pairwise_scatterplots(X, Y, columns_to_plot=columns_to_plot, sample_size=100)

    # Parse split arguments
    if parsed_args.split_ratio is not None:  # Parse the split_ratio if provided
        split_ratios = [int(item) for item in parsed_args.split_ratio.split()]
        if len(split_ratios) != 2 and len(split_ratios) != 3:
            raise ValueError("split_ratio must include two or three integers for train-test or train-val-test percentages.")
        aux_split_params["train_prc"] = split_ratios[0]
        # Assign val_prc only if there are 3 values, otherwise set it to 0 or None based on your logic preference
        aux_split_params["val_prc"] = split_ratios[1] if len(split_ratios) == 3 else 0
        # Assign test_prc based on the number of ratios provided
        aux_split_params["test_prc"] = split_ratios[2] if len(split_ratios) == 3 else split_ratios[1]
    else:
        aux_split_params["train_prc"] = None
        aux_split_params["val_prc"] = None
        aux_split_params["test_prc"] = None
    aux_split_params["n_folds"] = parsed_args.n_folds
    aux_split_params['stratified'] = parsed_args.stratified_kfold
    aux_split_params["random_seed"] = parsed_args.random_seed
    aux_split_params["look_back_days_sequential_prediction"] = parsed_args.look_back_days_sequential_prediction

    # Split data
    (X_train, Y_train,
     X_val, Y_val,
     X_test, Y_test,
     cv_indices,
     optional_split_info) = split_data_fn(X, Y, **aux_split_params)
    # Accumulate potential split parameters to the predict parameters as they may be useful for the prediction function
    if optional_split_info:
        aux_predict_params.update(optional_split_info)

    # Optimize hyperparameters and train model, distinguishing between sklearn and pytorch/tensorflow pipeline
    if "sklearn" in MODELS.get(parsed_args.model).__module__:
        # Ignore specific warnings relating to sparse columns warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Allowing arbitrary scalar fill_value in SparseDtype is deprecated")
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but RBFSampler was fitted with feature names")
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but Ridge was fitted with feature names")
        warnings.filterwarnings("ignore", message="pandas.DataFrame with sparse columns found.It will be converted to a dense numpy array.")

        # Construct the Transformers List Dynamically
        transformer_instances = []
        for transformer_name in parsed_args.data_transformers:
            TransformerClass = DATA_TRANSFORMERS.get(transformer_name)
            if TransformerClass is not None:
                valid_params = inspect.signature(TransformerClass.__init__).parameters
                # If hparams are passed in the command line arguments, filter them to find hparams valid and fixed for the current transformer
                if parsed_args.hparams is not None:
                    transformer_params = {pname.split('__')[1]: pvalue for pname, pvalue in json.loads(parsed_args.hparams).items()
                                          if pname.startswith(transformer_name)
                                          and pname.split('__')[1] in valid_params
                                          and not isinstance(pvalue, str)}
                else:
                    transformer_params = {}
                # If transformer accepts a random_state and parsed_args.random_seed is provided, add it
                if 'random_state' in valid_params and parsed_args.random_seed is not None:
                    transformer_params['random_state'] = parsed_args.random_seed
                # Initialize the transformer with fixed parameters
                transformer_instances.append((transformer_name, TransformerClass(**transformer_params)))

        # Define an sklearn Pipeline
        pipeline_steps = []
        pipeline_steps.extend(transformer_instances)
        pipeline_steps.append((parsed_args.model, model))

        pipeline = Pipeline(pipeline_steps)

        # Set up hyperparams optimization if valid hyperparameters are provided
        optimization_needed = False
        param_distributions = {}
        if parsed_args.hparams is not None:
            raw_param_distributions = json.loads(parsed_args.hparams)

            distr_pattern = r'^(uniform|loguniform|randint)\(\d+(\.\d+)?, \d+(\.\d+)?\)$'
            for estimname_paramname, distr_str in raw_param_distributions.items():
                # Check if the parameter value is a proper "distribution" string
                # (not well-formed distribution strings or numeric values are ignored by hopt)
                if isinstance(distr_str, str) and re.match(distr_pattern, distr_str):
                    # Split the parameter name to get the component and parameter
                    estimator, paramname = estimname_paramname.split("__")
                    # Check if the component is the model or one of the transformers
                    if estimator == parsed_args.model and paramname in model.get_params().keys():
                        param_distributions[estimname_paramname] = string_to_distribution(distr_str)
                    elif estimator in dict(transformer_instances) and paramname in dict(transformer_instances)[estimator].get_params().keys():
                        param_distributions[estimname_paramname] = string_to_distribution(distr_str)

                    optimization_needed = True

                # Note: Parameters not matching the pattern are ignored

        # Either perform hyperparams optimization with refit=True OR just fit the model
        cv_results_df = None
        if optimization_needed:
            # Case 1: Use predefined k-fold validation splits if no explicit validation set is provided
            if X_val is None and Y_val is None and cv_indices is not None:
                cv = cv_indices
                # Optionally subsample the training set and shallow copy it into X_tmp and Y_tmp for consistency with case 2
                X_tmp, Y_tmp, cv_indices = hopt_subsampling_fn(
                    X_train,
                    Y_train,
                    cv_indices=cv_indices,
                    subsampling_rate=parsed_args.hopt_subsampling_rate,
                    random_seed=parsed_args.random_seed,
                )
                n_folds = len(cv_indices)

            # Case 2: Use a predefined validation set when provided, ignoring cv_indices
            elif X_val is not None and Y_val is not None and cv_indices is None:
                # Subsample training and validation data to reduce model training time or manage computational resources
                X_train_subsampled, Y_train_subsampled, _ = hopt_subsampling_fn(
                    X_train,
                    Y_train,
                    cv_indices=cv_indices,
                    subsampling_rate=parsed_args.hopt_subsampling_rate,
                    random_seed=parsed_args.random_seed,
                )
                X_val_subsampled, Y_val_subsampled, _ = hopt_subsampling_fn(
                    X_val,
                    Y_val,
                    cv_indices=cv_indices,
                    subsampling_rate=parsed_args.hopt_subsampling_rate,
                    random_seed=parsed_args.random_seed,
                )

                # Combine training and validation datasets for use in scikit-learn's model selection tools
                X_tmp = pd.concat([X_train_subsampled, X_val_subsampled], ignore_index=True)
                Y_tmp = pd.concat([Y_train_subsampled, Y_val_subsampled], ignore_index=True)

                # Creating one explicit fold is not possible out of the box with sklearn (cv_indices must contain at
                # least two different train/val splits). For this reason, a custom validator PredefinedSplit was defined
                # to allow for a single train/val split.
                n_folds = 1
                val_fold = [0] * len(X_train_subsampled) + [1] * len(X_val_subsampled)  # Indicates which samples are from the training set (0) and which are from the validation set (1)
                del X_train_subsampled, Y_train_subsampled, X_val_subsampled, Y_val_subsampled
                cv = PredefinedSplit(test_fold=np.array(val_fold))

            else:
                raise ValueError("Only one is expected to be not None between cv_indices and (X_val, Y_val).")

            logger.info(f"Optimizing model hyperparameters ({parsed_args.hopt_n_rndcv_samplings} samplings * {n_folds} folds and fitting the model...")
            search = RandomizedSearchCV(pipeline,
                                        param_distributions,
                                        n_iter=parsed_args.hopt_n_rndcv_samplings,
                                        refit=False,  # Avoid automatic refitting because hopt may be performed on subsampled data
                                        cv=cv,
                                        random_state=parsed_args.random_seed,
                                        return_train_score=True,  # May slow down the execution
                                        verbose=3)
            search.fit(X_tmp.squeeze(), Y_tmp.squeeze())  # search.fit(convert_df_to_sparse_matrix(X_tmp), convert_df_to_sparse_matrix(Y_tmp)) for sparse fitting (debug it)

            # Collect the cv_results_
            if hasattr(search, "cv_results_"):
                cv_results_df = pd.DataFrame(search.cv_results_)
                float_columns = cv_results_df.select_dtypes(include=['float']).columns
                cv_results_df[float_columns] = cv_results_df[float_columns].round(4)
            else:
                cv_results_df = None

            # Collect the best_params_ and instantiate a new pipeline with optimized values
            if hasattr(search, 'best_params_'):
                best_params = search.best_params_

                best_pipeline = pipeline
                for pname, pvalue in best_params.items():
                    step, param = pname.split('__')
                    setattr(best_pipeline.named_steps[step], param, pvalue)
                pipeline = best_pipeline  # Shade optimized_pipeline behind the name model for unified use in case of no hopt
            else:
                raise AttributeError("Failed to find the best parameters. The model fitting process did not complete successfully.")

        # Fit the model (either optimized or not)
        logger.info("Fitting the model...")
        pipeline.fit(X_train.squeeze(), Y_train.squeeze())

        model = pipeline  # For consistency with other libraries, simply call the Sklearn pipeline "model"

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

    # Compute model predictionsce    logger.info("Computing model predictions...")
    Y_pred, Y_train_pred, optional_predictions = predict_fn(
        model,
        X_test.squeeze(),
        Y_test.squeeze(),  # Only relevant for prediction_fn = predict_zeros
        X_train.squeeze(),
        Y_train.squeeze(),  # Only relevant for prediction_fn = predict_zeros
        **aux_predict_params,
    )

    # Accumulate potential extra predictions into aux_eval_params, which may be useful for complete model evaluation
    if optional_predictions is not None:
        aux_eval_params.update(optional_predictions)

    # Evaluate model predictions
    logger.info("Evaluating model predictions...")
    scores, figs = evaluate_fn(
        Y_test.squeeze(),
        Y_pred,
        model,
        Y_test.columns.tolist() if isinstance(Y_test, (pd.DataFrame, pd.Series)) else None,
        Y_train.squeeze(),
        Y_train_pred,
        **aux_eval_params,
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
            cv_results_df.to_csv(cv_results_path, index=True)

        report_path = os.path.join(output_reports_dir, f"scores_{parsed_args.run_id}.csv")
        scores.to_csv(report_path, index=True)

        for fig_name, fig in figs.items():
            fig_path = os.path.join(output_figures_dir, f"{fig_name}_{parsed_args.run_id}.png")
            fig.savefig(fig_path)
            plt.close(fig)  # Close the figure after saving to free up memory

        # Collect analysis information in readable form
        script_call = "python " + " ".join(sys.argv)
        script_call = script_call.replace("\\", "/")
        script_call = script_call.replace(f'--split_ratio {parsed_args.split_ratio}', f'--split_ratio "{parsed_args.split_ratio}"')
        script_call = script_call.replace(f'--hparams {parsed_args.hparams}', f'--hparams "{parsed_args.hparams}"')
        script_call = escape_quotes_in_curly_brackets(script_call)

        # Convert relative paths to absolute paths
        matches = re.findall(re.compile(r'\.\.?/[^ ]+'), script_call)
        for match in matches:
            abs_path = os.path.abspath(match)
            script_call = script_call.replace(match, abs_path)

        # Format model steps' information and hyperparameters
        if "sklearn" in MODELS.get(parsed_args.model).__module__:
            model_steps_info = {step[0]: step[1].__class__.__name__ for step in model.steps}
            final_hyperparams = {step[0]: step[1].get_params() for step in model.steps}
        elif "tensorflow" in MODELS[parsed_args.model]:
            model_steps_info = None  # TODO
            final_hyperparams = None  # TODO
        elif "torch" in MODELS[parsed_args.model]:
            model_steps_info = None  # TODO
            final_hyperparams = None  # TODO
        else:
            model_steps_info = None  # TODO
            final_hyperparams = None  # TODO

        # Store information as a text file. Do not use JSON because it messes up with necessary escapes.
        experiment_info = {
            "run_id": parsed_args.run_id,
            "data_path": parsed_args.data_path,
            "dataset_shape": str(X.shape),
            "precomputed_features_path": parsed_args.precomputed_features_path,
            "data_loading_fn": parsed_args.data_loading_fn,
            "model": parsed_args.model,
            "data_transformers": parsed_args.data_transformers,
            "hparams": parsed_args.hparams,
            "hopt_n_rndcv_samplings": parsed_args.hopt_n_rndcv_samplings,
            "hopt_subsampling_fn": parsed_args.hopt_subsampling_fn,
            "hopt_subsampling_rate": parsed_args.hopt_subsampling_rate,
            "reuse_model": "True" if parsed_args.reuse_model else "False",
            "preprocessing_fn": parsed_args.preprocessing_fn,
            "eda_fn": parsed_args.eda_fn,
            "feature_extraction_fn": parsed_args.feature_extraction_fn,
            "split_fn": parsed_args.split_fn,
            "split_ratio": parsed_args.split_ratio,
            "n_folds": parsed_args.n_folds,
            "stratified_kfold": "True" if parsed_args.stratified_kfold else "False",
            "evaluation_fn": parsed_args.evaluation_fn,
            "log_level": parsed_args.log_level,
            "random_seed": parsed_args.random_seed,
            "save_output": "True" if parsed_args.save_output else "False",
            "output_data_dir": parsed_args.output_data_dir,
            "output_model_dir": parsed_args.output_model_dir,
            "output_reports_dir": parsed_args.output_reports_dir,
            "output_figures_dir": parsed_args.output_figures_dir,
            "model_steps_info": model_steps_info,
            "initial_hyperparameters": parsed_args.hparams,
            "final_hyperparameters": final_hyperparams,
            "optimization_performed": "True" if optimization_needed else "False",
            "performance_metrics": scores.index.tolist(),
            "script_call": " ".join(sys.argv),
            "additional_information": "None",
        }
        # The file is saved with the run_id as part of the filename for easy identification.
        with open(os.path.join(output_reports_dir, f"experiment_details_{parsed_args.run_id}.txt"), "w") as file:
            for key, value in experiment_info.items():
                file.write(f"{key}: {value}\n\n")

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Script")
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--data_loading_fn', choices=DATA_LOADING_FNS.keys(), help='Function identifier for loading data')
    parser.add_argument('--precomputed_features_path', type=str, help='Path to pre-computed features to skip loading, preprocessing, and feature extraction')
    parser.add_argument('--model', choices=MODELS.keys(), help='Model identifier')
    parser.add_argument('--data_transformers', nargs='*', default=[], help='List of transformer identifiers, e.g., sklearn_RBFSampler sklearn_StandardScaler')
    parser.add_argument('--hparams', default=None, help='JSON string of hyperparameters for the data transformers or the model')
    parser.add_argument('--hopt_n_rndcv_samplings', type=int, default=5, help='Number of samplings for RandomSearchCV hyperparameter optimization')
    parser.add_argument('--hopt_subsampling_fn', default='subsampling_passthrough', choices=HOPT_SUBSAMPLING_FNS.keys(), help='Identifier for training set subsampling function')
    parser.add_argument('--hopt_subsampling_rate', default=1, type=float, help='Proportion of the original training set retained for hyperparameter optimization')
    parser.add_argument('--reuse_model', help='Path to a pre-trained model to reuse')
    parser.add_argument('--preprocessing_fn', default='preprocess_passthrough', choices=PREPROCESSING_FNS.keys(), help='Identifier for preprocessing function')
    parser.add_argument('--eda_fn', default='eda_passthrough', choices=EDA_FNS.keys(), help='Identifier for exploratory data analysis function')
    parser.add_argument('--feature_extraction_fn', default='features_exampledb', choices=FEATURE_EXTRACTION_FNS.keys(), help='Identifier for feature extraction function')
    parser.add_argument('--split_fn', default='split_train_val_test', choices=SPLITTING_FNS.keys(), help='Identifier for data split function')
    parser.add_argument('--split_ratio', type=str, help='Ratio for splitting data')
    parser.add_argument('--n_folds', type=int, help='Number of folds for k-fold cross-validation')
    parser.add_argument('--stratified_kfold', action='store_true', help='Whether to perform stratified (for clf) or standard (for reg or clf) k-fold cross-validation')
    parser.add_argument('--look_back_days_sequential_prediction', type=int, default=0, help='Number of look-back days used in sequential multi-day time series forecasting for computing features at prediction time.')
    parser.add_argument('--prediction_fn', default='predict_zeros', help='Identifier for prediction function')
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
    check_load_args(parsed_args.data_path, parsed_args.data_loading_fn, parsed_args.precomputed_features_path)
    check_split_args(parsed_args.split_fn, parsed_args.split_ratio, parsed_args.model)
    check_hparams_opt_args(parsed_args.hparams, parsed_args.split_fn, parsed_args.n_folds)
    check_output_args(parsed_args.save_output, parsed_args.output_data_dir, parsed_args.output_model_dir, parsed_args.output_reports_dir, parsed_args.output_figures_dir)

    main(parsed_args)
