"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

_VALID_MODEL_EXTENSIONS = {".pt"}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Parameters
    ----------
    path : Path
        Path to the YAML file

    Returns
    -------
    dict[str, Any]
        Parsed YAML content
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: {path}. "
            "Please ensure the file exists and the path is correct."
        ) from None
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file {path}: {e}") from e


def validate_model_path(base_dir: Path, config: dict[str, Any]) -> str:
    """Validate the model path from configuration.

    Parameters
    ----------
    base_dir : Path
        Base directory for the project
    config : dict
        Model configuration dict containing 'model_config.model' key

    Returns
    -------
    Path
        Validated path to the model file
    """
    model_name = config.get("model_config", {}).get("model", "yolov8n.pt")
    model_path = base_dir / "models" / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Please ensure the model exists in the models directory."
        )

    if model_path.suffix not in _VALID_MODEL_EXTENSIONS:
        raise ValueError(
            f"Invalid model file extension: {model_path.suffix}. "
            f"Expected one of {_VALID_MODEL_EXTENSIONS}."
        )

    logger.success(f"Model file found: {model_path}")

    return str(model_path)
