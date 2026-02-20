"""Functions for notebook utilities."""

import inspect
import sys
from typing import Any

from loguru import logger


# Function to delete a module and then reload it
def reload_module(module_name: str) -> None:
    """Reload a module by deleting it from sys.modules and re-importing it.

    Parameters
    ----------
    module_name : str
        Name of the module to reload.
    """
    if module_name in sys.modules:
        del sys.modules[module_name]
    logger.success(f"Module {module_name} unloaded.")
    logger.info(f"Please reload {module_name} to apply changes.")


def log_configs(*configs: dict[str, Any]) -> None:
    """
    Log multiple configuration parameters with their names and values.

    Args:
        *configs: Variable number of configuration dictionaries to log
    """
    # Get the caller's frame to retrieve variable names
    frame = inspect.currentframe()
    caller_frame = None
    var_names = []

    try:
        caller_frame = frame.f_back if frame is not None else None

        # Try to get the variable names from the caller
        if caller_frame is not None:
            for var_name, var_value in caller_frame.f_locals.items():
                for config in configs:
                    if var_value is config:
                        var_names.append(var_name)
                        break
    finally:
        # Break reference cycles involving frame objects
        if frame is not None:
            del frame
        if caller_frame is not None:
            del caller_frame

    # If we couldn't get names, use generic names
    if len(var_names) != len(configs):
        var_names = [f"Config_{i + 1}" for i in range(len(configs))]

    # Log each configuration
    for config, name in zip(configs, var_names, strict=False):
        logger.info(f"=== {name} ===")
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict):
                    logger.info(f"{key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"  {sub_key}: {sub_value}")
                else:
                    logger.info(f"{key}: {value}")
        else:
            logger.info(f"Value: {config}")
