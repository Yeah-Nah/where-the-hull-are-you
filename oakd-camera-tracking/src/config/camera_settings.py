from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent.parent.parent
CONFIG = BASE_DIR / "config" / "camera_config.yaml"

# Model configuration settngs
with open(CONFIG) as f:
    config = yaml.safe_load(f)

CAM_WIDTH = config.get("cam_width_pixels", 512)
CAM_HEIGHT = config.get("cam_height_pixels", 384)
