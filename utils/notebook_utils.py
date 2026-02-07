"""Functions for notebook utilities."""

import sys

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
