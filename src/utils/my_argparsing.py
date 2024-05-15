import argparse
import json
import re
from datetime import datetime
from typing import Optional, Union

from scipy.stats import rv_continuous, rv_discrete


class FunctionRegistry:
    """
        A registry to dynamically load and retrieve functions, classes, or objects using identifiers.

        This registry allows for dynamic loading and retrieval of various callable or non-callable objects
        using string identifiers, making it useful for scenarios such as command-line interface (CLI)
        arguments that specify different processing functions or models.

        Attributes:
            _functions (dict): A dictionary storing registered items with their corresponding names.
        """
    def __init__(self):
        self._functions = {}

    def register(self, name, func):
        """Register a function with a given name."""
        if name in self._functions:
            raise ValueError(f"Function {name} is already registered.")
        self._functions[name] = func

    def get_function(self, name):
        """Retrieve a function by its name. Returns None if the function is not registered."""
        return self._functions.get(name)

    def get_function_names_starting_by(self, prefix):
        """Retrieve the list of function names starting with the given prefix."""
        return [name for name in self._functions.keys() if name.startswith(prefix)]


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


def check_split_args(split_fn: str, split_ratio: str, model: str, function_registry: FunctionRegistry) -> None:
    """"
    Check the consistency and validity of split function arguments.

    Args:
        split_fn (str): Identifier for the data splitting function.
        split_ratio (str): A string representing the ratio for splitting data, e.g., "80 20".
        model (str): Model identifier to check compatibility with the splitting strategy.
        function_registry (FunctionRegistry): The function registry instance to retrieve available functions.

    Raises:
        ValueError: If any inconsistency or incompatibility is found among the arguments.
    """
    if split_fn not in function_registry.get_function_names_starting_by("split"):
        raise ValueError(f"split_fn must be one among {function_registry.get_function_names_starting_by('split')}.")

    if (split_fn == "split_train_test" or split_fn == "split_train_val_test") and split_ratio is None:
        raise ValueError("split_ratio must be provided for splitting functions 'split_train_test' and 'split_train_val_test'.")

    ratios = []
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
    model_module = function_registry.get_function(model).__module__
    # if "sklearn" in model_module:
    #     if split_fn == "split_train_val_test":
    #         raise ValueError("Splitting function 'split_train_val_test' incompatible with sklearn models.")
    #     if split_ratio is None or len(ratios) != 2:
    #         raise ValueError("Split ratio for sklearn models must contain exactly two numbers.")

    if "tensorflow" in model_module or "torch" in model_module:
        if split_fn == "split_train_test":
            raise ValueError("Splitting function 'split_train_test' incompatible with sklearn models.")
        if split_ratio is None or len(ratios) != 3:
            raise ValueError("Split ratio for TensorFlow/PyTorch models must contain exactly three numbers.")


def check_hopt_args(hparams: Optional[str], split_fn: str, n_folds: Optional[int]) -> None:
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
            raise ValueError("All output directories must be specified when --save_output is used")


def string_to_distribution(value: str, function_registry: FunctionRegistry) -> Union[rv_continuous, rv_discrete]:
    """
    Converts a string representation of a distribution into a SciPy distribution object.

    Args:
        value (str): String representation of the distribution, including the distribution
                     name and its parameters in parentheses.
        function_registry (FunctionRegistry): The function registry instance to retrieve available functions.

    Returns:
        distr_obj (Union[rv_continuous, rv_discrete]): A SciPy distribution object corresponding
        to the input string. This could be either a continuous or a discrete distribution based on the
        input. Supports 'uniform', 'loguniform', and 'randint' distributions.
    """
    match = re.match(r'^(uniform|loguniform|randint)\(([^,]+), ([^)]+)\)$', value)
    func_name, arg1, arg2 = match.groups()
    arg1 = float(arg1)
    arg2 = float(arg2)
    distr_obj = function_registry.get_function(func_name)(arg1, arg2)
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


def parse_data_science_arguments(function_registry):
    # Retrieve dynamic choices from function_registry by category
    data_loading_fns = function_registry.get_function_names_starting_by("load")
    preprocessing_fns = function_registry.get_function_names_starting_by("preprocess")
    eda_fns = function_registry.get_function_names_starting_by("eda")
    feature_extraction_fns = function_registry.get_function_names_starting_by("features")
    split_fns = function_registry.get_function_names_starting_by("split")
    prediction_fns = function_registry.get_function_names_starting_by("predict")
    evaluation_fns = function_registry.get_function_names_starting_by("evaluate")
    models = function_registry.get_function_names_starting_by("sklearn") + function_registry.get_function_names_starting_by("mymodel")
    data_transformers = function_registry.get_function_names_starting_by("sklearn")
    hopt_subsampling_fns = function_registry.get_function_names_starting_by("hopt_subsampling")

    parser = argparse.ArgumentParser(description="Machine Learning Script")

    parser.add_argument('--data_path', required=False, help='Path to the data file')
    parser.add_argument('--data_loading_fn', required=False, choices=data_loading_fns, help='Function identifier for loading data')
    parser.add_argument('--precomputed_features_path', required=False, type=str, help='Path to pre-computed features to skip loading, preprocessing, and feature extraction')
    parser.add_argument('--model', choices=models, help='Model identifier')
    parser.add_argument('--data_transformers', nargs='*', default=[], help='List of transformer identifiers, e.g., sklearn_RBFSampler sklearn_StandardScaler')
    parser.add_argument('--hparams', default=None, help='JSON string of hyperparameters for the data transformers or the model')
    parser.add_argument('--hopt_n_rndcv_samplings', type=int, default=5, help='Number of samplings for RandomSearchCV hyperparameter optimization')
    parser.add_argument('--hopt_subsampling_fn', default='hopt_subsampling_passthrough', choices=hopt_subsampling_fns, help='Identifier for training set subsampling function')
    parser.add_argument('--hopt_subsampling_rate', default=1, type=float, help='Proportion of the original training set retained for hyperparameter optimization')
    parser.add_argument('--reuse_model', help='Path to a pre-trained model to reuse')
    parser.add_argument('--preprocessing_fn', default='preprocess_passthrough', choices=preprocessing_fns, help='Identifier for preprocessing function')
    parser.add_argument('--eda_fn', default='eda_passthrough', choices=eda_fns, help='Identifier for exploratory data analysis function')
    parser.add_argument('--feature_extraction_fn', required=True, choices=feature_extraction_fns, help='Identifier for feature extraction function')
    parser.add_argument('--split_fn', required=True, choices=split_fns, help='Identifier for data split function')
    parser.add_argument('--split_ratio', type=str, help='Ratio for splitting data')
    parser.add_argument('--n_folds', type=int, help='Number of folds for k-fold cross-validation')
    parser.add_argument('--stratified_kfold', action='store_true', help='Whether to perform stratified (for clf) or standard (for reg or clf) k-fold cross-validation')
    parser.add_argument('--look_back_days_sequential_prediction', type=int, default=0, help='Number of look-back days used in sequential multi-day time series forecasting for computing features at prediction time.')
    parser.add_argument('--prediction_fn', default='predict_zeros', help='Identifier for prediction function')
    parser.add_argument('--evaluation_fn', required=True, choices=evaluation_fns, help='Identifier for evaluation function')
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
    check_split_args(parsed_args.split_fn, parsed_args.split_ratio, parsed_args.model, function_registry)
    check_hopt_args(parsed_args.hparams, parsed_args.split_fn, parsed_args.n_folds)
    check_output_args(parsed_args.save_output, parsed_args.output_data_dir, parsed_args.output_model_dir, parsed_args.output_reports_dir, parsed_args.output_figures_dir)

    return parsed_args
