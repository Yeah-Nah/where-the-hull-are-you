"""Model settings configuration."""

from pathlib import Path

from utils.config_utils import load_yaml, validate_model_path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DEFAULT_MLFLOW_URI = f"file:{BASE_DIR / 'output' / 'mlruns'}"

# Load configurations
MODEL_CONFIG = load_yaml(BASE_DIR / "config" / "model_config.yaml")
HYPERPARAM_SEARCH_SPACE = load_yaml(
    BASE_DIR / "config" / "hyperparam_search_config.yaml"
).get("search_space", {})
TRAINING_CONFIG = load_yaml(BASE_DIR / "config" / "training_config.yaml")

# Checking model path validity
MODEL_CONFIG["model_path"] = validate_model_path(BASE_DIR, MODEL_CONFIG)
