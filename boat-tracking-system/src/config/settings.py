# Configuration settings for the boat tracking system
import yaml
import os
from pathlib import Path

# Get project root (2 levels up from this file)
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_FILE = BASE_DIR / 'config' / 'config.yaml'

# Load from YAML file
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

MODEL_PATH = config.get('model_path', 'models/yolov8n.pt')
CONFIDENCE_THRESHOLD = config.get('confidence_threshold', 0.5)

# Input and output directories
INPUT_DIR = config.get("input_directory", "")
OUTPUT_DIR = config.get("output_directory", "")

# Boat classes to detect
BOAT_CLASSES = config.get("boat_classes", ["boat", "ship"])