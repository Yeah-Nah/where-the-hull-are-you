"""Model settings configuration."""

from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_CONFIG_FILE = BASE_DIR / "config" / "model_config.yaml"
HYPERPARAM_SEARCH_CONFIG_FILE = BASE_DIR / "config" / "hyperparam_search_config.yaml"
DEFAULT_MLFLOW_URI = f"file:{BASE_DIR / 'output' / 'mlruns'}"

# Model configuration settings
try:
    with open(MODEL_CONFIG_FILE) as f:
        model_config = yaml.safe_load(f)
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Model configuration file not found: {MODEL_CONFIG_FILE}. "
        "Please ensure the file exists and the path is correct."
    ) from e
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing model configuration file: {e}") from e

MODEL_CONFIG = model_config.get("model_config", {})
MODEL = MODEL_CONFIG.get("model", "yolov8n.pt")
MODEL_PATH = BASE_DIR / Path("models") / MODEL

# Search space configuration settings
try:
    with open(HYPERPARAM_SEARCH_CONFIG_FILE) as f:
        hyperparam_search_config = yaml.safe_load(f)
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Hyperparameter search configuration file not found: {HYPERPARAM_SEARCH_CONFIG_FILE}. "
        "Please ensure the file exists and the path is correct."
    ) from e
except yaml.YAMLError as e:
    raise ValueError(
        f"Error parsing hyperparameter search configuration file: {e}"
    ) from e

HYPERPARAM_SEARCH_SPACE = hyperparam_search_config.get("search_space", {})
