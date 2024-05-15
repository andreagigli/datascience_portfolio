import argparse
import inspect
import json
import pickle
from typing import Any

from src.utils.my_argparsing import FunctionRegistry


def init_reload_model(parsed_args: argparse.Namespace, function_registry: FunctionRegistry) -> Any:
    """
    Initializes or reloads a model based on the provided command-line arguments.

    Args:
        parsed_args (argparse.Namespace): The namespace object containing command-line arguments relevant to model initialization.
        function_registry (FunctionRegistry): The function registry instance to retrieve available functions.

    Returns:
        model_instance (Any): The initialized or reloaded model instance.

    Raises:
        ValueError: If both a new model identifier and a path to reuse a model are provided, or if the model
                    file for reuse is not found or cannot be loaded due to errors.
    """
    if parsed_args.reuse_model and parsed_args.model:
        raise ValueError("Specify either a model to train or a model to reuse, not both.")

    if parsed_args.model:
        ModelClass = function_registry.get_function(parsed_args.model)
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