"""
This script is designed for flexible machine learning workflows. It allows for dynamic loading of data, preprocessing, feature extraction, model training, and evaluation based on specified function identifiers. Users can specify model parameters, choose to train a new model or use a pre-trained one, and control data splitting for training and testing.

Example shell calls:

1. Basic usage with required arguments:
python analysis_template.py
--data_path ../../data/external/exampledb.csv
--data_loading_fn load_exampledb
--model sklearn_RandomForestRegressor
--model_hparams "{\"n_estimators\": 100, \"max_depth\": 10}"
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
--output_report_dir ../../outputs/reports
--output_figures_dir ../../outputs/figures

2. Specifying a model and hyperparameters: ...

3. Using a pre-trained model: ...

4. Custom data splitting without custom splitting function: ...
"""

import argparse
import json
import os
import re
import importlib
import matplotlib.pyplot as plt
import pickle

from datetime import datetime

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.utils.my_os import ensure_dir_exists

# Dictionaries for mapping identifiers to strings representing sklearn or custom functions
MODELS = {
    "sklearn_LinearRegression": "sklearn.linear_model.LinearRegression",
    "sklearn_SVC": "sklearn.svm.SVC",
    "sklearn_RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
    "mymodel": "models.mymodel",
}
DATA_LOADING_FNS = {
    "load_exampledb": "src.data.load_exampledb.load_data",
}
PREPROCESSING_FNS = {
    "preprocess_exampledb": "src.data.preprocess_exampledb.preprocess_data",
}
FEATURE_EXTRACTION_FNS = {
    "feature_exampledb": "src.features.features_exampledb.extract_features",
}
SPLITTING_FNS = {
    "split_train_val_test": "src.data.split_train_val_test.split_data",
    "split_kfold": "src.data.split_kfold.split_data",
    "split_exampledb": "src.data.split_exampledb.split_data",
}
EVALUATION_FNS = {
    "evaluate_exampledb": "src.evaluation.evaluate_exampledb.evaluate",
}


def check_split_args(split_fn, split_ratio, n_folds):
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

    # Validation based on split_fn
    if split_fn == 'split_kfold':
        if not n_folds or n_folds < 2:
            raise ValueError("Number of folds must be specified and >= 2 for k-fold cross-validation.")
        if len(split_ratio.split()) != 2:
            raise ValueError("Split ratio must contain two numbers for k-fold cross-validation.")

    elif split_fn == 'split_train_val_test':
        if n_folds is not None:
            raise ValueError("Number of folds must not be specified for split function split_train_val_test.")
        if len(split_ratio.split()) != 3:
            raise ValueError("Split ratio must contain three numbers for one-fold validation.")

    else:  # custom split function
        # Custom checks for custom split function
        pass


def check_output_args(save_output, output_data_dir, output_model_dir, output_report_dir, output_figures_dir):
    if save_output:
        output_dirs = [output_data_dir, output_model_dir, output_report_dir, output_figures_dir]
        if any(dir is None for dir in output_dirs):
            parser.error("All output directories must be specified when --save_output is used")


def validate_hparams_format(param_string):
    """Validate that the JSON string is well-formed and contains proper hyperparameters."""
    try:
        params = json.loads(param_string)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid JSON string for hyperparameters.")

    for key, value in params.items():
        if not (isinstance(value, (int, float, str)) and (
                isinstance(value, str) and re.match(r'^(uniform|loguniform|randint)\(\d+(\.\d+)?, \d+(\.\d+)?\)$',
                                                    value) or isinstance(value, (int, float)))):
            raise argparse.ArgumentTypeError(
                f"Invalid value for hyperparameter {key}: must be a scalar or a specific distribution string.")

    return param_string  # Return the original string; parsing into dict will happen later


def load_fn(full_function_path):
    """Dynamically loads a function using the full function path."""
    module_name, function_name = full_function_path.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ImportError, AttributeError):
        raise ImportError(f"Could not dynamically load the specified function: {full_function_path}")


def init_reload_model(args):
    """Initialize new model or try reloading a preliminary trained one."""
    if args.reuse_model and args.model:
        raise ValueError("Specify either a model to train or a model to reuse, not both.")

    model = None
    if args.model:
        ModelClass = load_fn(MODELS.get(args.model))
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


def main(parsed_args):
    # Dynamic function selection
    load_data_fn = load_fn(DATA_LOADING_FNS.get(parsed_args.data_loading_fn))
    preprocess_fn = load_fn(PREPROCESSING_FNS.get(parsed_args.preprocessing_fn))
    extract_features_fn = load_fn(FEATURE_EXTRACTION_FNS.get(parsed_args.feature_extraction_fn))
    split_data_fn = load_fn(SPLITTING_FNS.get(parsed_args.split_fn))
    evaluate_fn = load_fn(EVALUATION_FNS.get(parsed_args.evaluation_fn))

    # Initialize model or reload existing one
    model = init_reload_model(parsed_args)

    # Load, preprocess, extract features
    X, Y = load_data_fn(parsed_args.data_path)
    X, Y = preprocess_fn(X, Y)
    X, Y = extract_features_fn(X, Y)

    # Split data
    split_ratios = [float(el) for el in parsed_args.split_ratio.split()]
    additional_args = [parsed_args.n_folds, parsed_args.stratified_kfold] if parsed_args.split_fn == 'split_kfold' else []
    split_data_fn_args = split_ratios + additional_args
    X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices = split_data_fn(X, Y, parsed_args.random_seed, *split_data_fn_args)

    # Placeholder for any additional data transformation
    data_transformer = FunctionTransformer()

    # Define an sklearn Pipeline 
    pipeline = Pipeline([('transformer', data_transformer), ('model', model)])

    # Set up hyperparams optimization if there is any valid hyperparameter associated with a distribution string
    valid_params = model.get_params().keys()
    distr_pattern = r'^(uniform|loguniform|randint)\(\d+(\.\d+)?, \d+(\.\d+)?\)$'
    param_distributions = {k: v for k, v in json.loads(parsed_args.model_hparams).items()
                           if k in valid_params and isinstance(v, str) and re.match(distr_pattern, v)}
    optimization_needed = bool(param_distributions)

    # Either perform hyperparams optimization with refit=True OR just fit the model
    best_params = None
    if optimization_needed:
        search = RandomizedSearchCV(pipeline, param_distributions, refit=True, cv=cv_indices, random_state=parsed_args.random_seed)
        search = search.fit(X_train.to_numpy(), Y_train.to_numpy())
        model = search.best_estimator_
        model_hparams = search.best_params_
    else:
        model = model.fit(X_train.to_numpy(), np.squeeze(Y_train.to_numpy()))
        model_hparams = model.get_params()

    # Evaluate the model
    Y_pred = model.predict(X_test.to_numpy())
    scores, figs = evaluate_fn(np.squeeze(Y_test.to_numpy()), np.squeeze(Y_pred), model, Y_test.columns.tolist())

    # Save results
    if parsed_args.save_output:
        ensure_dir_exists(parsed_args.output_data_dir)
        model_summaries_dir = os.path.join(parsed_args.output_model_dir, "model_summaries")
        ensure_dir_exists(model_summaries_dir)
        trained_models_dir = os.path.join(parsed_args.output_model_dir, "trained_models")
        ensure_dir_exists(trained_models_dir)
        ensure_dir_exists(parsed_args.output_report_dir)
        ensure_dir_exists(parsed_args.output_figures_dir)

        model_path = os.path.join(trained_models_dir, f"trained_model_{parsed_args.run_id}.pkl")
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

        report_path = os.path.join(parsed_args.output_report_dir, f"scores_{parsed_args.run_id}.csv")
        scores.to_csv(report_path, index=False)

        for fig_name, fig in figs.items():
            fig_path = os.path.join(parsed_args.output_figures_dir, f"{fig_name}_{parsed_args.run_id}.png")
            fig.savefig(fig_path)
            plt.close(fig)  # Close the figure after saving to free up memory

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Script")
    parser.add_argument('--data_path', required=True, help='Path to the data file')
    parser.add_argument('--data_loading_fn', required=True, help='Function identifier for loading data')
    parser.add_argument('--model', help='Model identifier')
    parser.add_argument('--model_hparams', help='JSON string of model hyperparameters', type=validate_hparams_format)
    parser.add_argument('--reuse_model', help='Path to a pre-trained model to reuse')
    parser.add_argument('--preprocessing_fn', required=True, help='Identifier for preprocessing function')
    parser.add_argument('--feature_extraction_fn', required=True, help='Identifier for feature extraction function')
    parser.add_argument('--split_fn', required=True, help='Identifier for data split function')
    parser.add_argument('--split_ratio', required=True, type=str, help='Ratio for splitting data')
    parser.add_argument('--n_folds', type=int, help='Number of folds for k-fold cross-validation')
    parser.add_argument('--stratified_kfold', type=bool, default=False, help='Whether to perform stratified (for clf) or standard (for reg or clf) k-fold cross-validation')
    parser.add_argument('--evaluation_fn', required=True, help='Identifier for evaluation function')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (e.g., "INFO", "DEBUG")')
    parser.add_argument('--random_seed', type=int, default=None, help='Seed for random number generators for reproducibility')
    parser.add_argument('--run_id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help='Unique identifier for the run')
    parser.add_argument('--save_output', action='store_true', help='Save outputs (data, models, reports, figures)')
    parser.add_argument('--output_data_dir', type=str, help='Directory to save processed data')
    parser.add_argument('--output_model_dir', type=str, help='Directory to save trained models')
    parser.add_argument('--output_report_dir', type=str, help='Directory to save evaluation reports')
    parser.add_argument('--output_figures_dir', type=str, help='Directory to save generated figures')
    parsed_args = parser.parse_args()

    # Additional consistency check
    check_split_args(parsed_args.split_fn, parsed_args.split_ratio, parsed_args.n_folds)
    check_output_args(parsed_args.save_output, parsed_args.output_data_dir, parsed_args.output_model_dir, parsed_args.output_report_dir, parsed_args.output_figures_dir)

    main(parsed_args)




