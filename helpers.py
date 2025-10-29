from typing import Any, List, Union
from loguru import logger

def log_variable_names(param_names, scope_dict) -> None:
    """Log the variable name and its value from provided scope."""
    for param in param_names:
        if param in scope_dict:
            logger.info(f"Variable '{param}': {scope_dict[param]}")
        else:
            logger.warning(f"Variable '{param}' not found in the provided scope.")