"""Configuration settings for data preparation module."""

from pathlib import Path

import yaml
from loguru import logger

# Base directory for the data-preparation module
BASE_DIR = Path(__file__).parent.parent
CONFIG_FILE = BASE_DIR / "config" / "data_prep_config.yaml"

# Load configuration from YAML file
try:
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.error(f"Configuration file not found: {CONFIG_FILE}")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML configuration: {e}")
    raise

# Parse configuration values
INPUT_VIDEO_DIR = Path(config.get("input_video_directory", "path/to/video/folder"))
OUTPUT_FRAMES_DIR = BASE_DIR / Path(
    config.get("output_frames_directory", "output/frames")
)
FRAME_PROPORTION = float(config.get("frame_proportion", 0.5))
MAX_VIDEO_SIZE_MB = int(config.get("max_video_size_mb", 500))

# Validate frame proportion
if not 0.0 <= FRAME_PROPORTION <= 1.0:
    error_msg = f"frame_proportion must be between 0.0 and 1.0, got {FRAME_PROPORTION}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Ensure output directory exists
OUTPUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Output directory set to: {OUTPUT_FRAMES_DIR}")
