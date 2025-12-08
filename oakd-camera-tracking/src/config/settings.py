"""Configuration settings for the boat tracking system."""
from pathlib import Path

import yaml

# Get project root (2 levels up from this file)
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"


with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

INPUT_DIR = Path(config.get("input_directory", ""))
INPUT_PREPROCESSED_DIR = INPUT_DIR / "preprocessed"
OUTPUT_DIR = Path(config.get("output_directory", ""))

# MLflow settings
MLFLOW_TRACKING_URI = config.get("mlflow_tracking_uri", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = config.get("mlflow_experiment_name", None)
