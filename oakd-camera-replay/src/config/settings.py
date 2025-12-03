# Configuration settings for the boat tracking system
import yaml
import os
from pathlib import Path

# Get project root (2 levels up from this file)
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"
MODEL_CONFIG_FILE = BASE_DIR / "config" / "model_config.yaml"

with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

# MLflow settings
MLFLOW_TRACKING_URI = config.get("mlflow_tracking_uri", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = config.get(
    "mlflow_experiment_name", None
)

# Model configuration settngs
with open(MODEL_CONFIG_FILE) as f:
    model_config = yaml.safe_load(f)

CLASS_IDS = model_config.get("class_ids", [0])
CONFIDENCE_THRESHOLD = model_config.get("confidence_threshold", 0.5)
