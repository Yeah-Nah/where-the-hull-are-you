# Configuration settings for the boat tracking system
import yaml
import os
from pathlib import Path

# Get project root (2 levels up from this file)
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"
MODEL_CONFIG_FILE = BASE_DIR / "config" / "model_config.yaml"

# Project configurations
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

# Input and output directories
INPUT_DIR = Path(config.get("input_directory", ""))
INPUT_PREPROCESSED_DIR = INPUT_DIR / "preprocessed"
OUTPUT_DIR = Path(config.get("output_directory", ""))

# Model configuration settngs
with open(MODEL_CONFIG_FILE) as f:
    model_config = yaml.safe_load(f)

MODEL_PATH = model_config.get("model_path", "models/yolov8n.pt")
CONFIDENCE_THRESHOLD = model_config.get("confidence_threshold", 0.5)

# Boat classes to detect
BOAT_CLASSES = model_config.get("boat_classes", ["boat", "ship"])

# Model tracking parameters
BOTSORT_CONFIG = model_config.get("botsort_config", {})

# MLflow settings
MLFLOW_TRACKING_URI = config.get("mlflow_tracking_uri", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = config.get(
    "mlflow_experiment_name", "boat-tracking-experiments"
)
MLFLOW_ARTIFACT_LOCATION = config.get("mlflow_artifact_location", None)

# Experiment settings
ENABLE_METRICS_COLLECTION = config.get("enable_metrics_collection", True)
