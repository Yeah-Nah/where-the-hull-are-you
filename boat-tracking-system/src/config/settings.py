# Configuration settings for the boat tracking system

import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output directories
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'output')

# Model settings
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')

# Confidence threshold for object detection
CONFIDENCE_THRESHOLD = 0.5

# Logging settings
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'application.log')  # Ensure logs directory exists before running the application

# Other settings can be added as needed