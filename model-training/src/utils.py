"""Utility functions for module management."""

import sys
from importlib import import_module

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
    import_module(module_name)
    logger.info(f"Module {module_name} reloaded.")
