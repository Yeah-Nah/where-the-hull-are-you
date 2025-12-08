# Just test file for tracking an object in a replayed oak-d camera feed

#!/usr/bin/env python3


from argparse import ArgumentParser
from pathlib import Path

from src.config.model_settings import MODEL_BLOB
from src.config.settings import INPUT_DIR
from src.detection.detection_pipeline import detectionPipeline

scriptDir = Path(__file__).resolve().parent
examplesRoot = (
    scriptDir / Path("../oakd-camera-replay")
).resolve()  # This resolves the parent directory correctly
models = examplesRoot / "src" / "models"

parser = ArgumentParser()
parser.add_argument("-i", "--inputVideo", default=INPUT_DIR, help="Input video name")
parser.add_argument(
    "-c", "--camera", type=bool, help="Use camera as input", default=False
)
args = parser.parse_args()

# Create and run detection pipeline
detection_pipeline = detectionPipeline(
    input_video=args.inputVideo,
    use_camera=args.camera,
    model_path=models / MODEL_BLOB,
)

detection_pipeline.run_detection()
