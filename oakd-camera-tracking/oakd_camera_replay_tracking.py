#!/usr/bin/env python3 # noqa

from argparse import ArgumentParser
from pathlib import Path

from src.config.model_settings import MODEL_BLOB
from src.detection.detection_pipeline import DetectionPipeline

script_dir = Path(__file__).resolve().parent
examples_root = (
    script_dir / Path("../oakd-camera-tracking")
).resolve()  # This resolves the parent directory correctly
models = examples_root / "src" / "models"

parser = ArgumentParser()
parser.add_argument(
    "-c", "--camera", type=bool, help="Use camera as input", default=False
)
args = parser.parse_args()

# Create and run detection pipeline
detection_pipeline = DetectionPipeline(
    input_video=args.inputVideo,
    use_camera=args.camera,
    model_path=models / MODEL_BLOB,
)

detection_pipeline.run_detection()
