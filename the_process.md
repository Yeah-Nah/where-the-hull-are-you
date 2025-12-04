# The Process: Building Where The Hull Are You

A development journey blog documenting the creation of a boat tracking system.

---

# Phase 0: Set Up Local Dev Env

## Entry 4: YOLOv8 Integration & Real-Time Detection
*Date: December 3, 2025*

### Converting Models and Building Detection Pipeline

Covered the following:

- **Model Conversion**: Created `convert_pt_to_blob.py` script to convert YOLOv8 `.pt` models to `.blob` format for OAK-D deployment (512×384 input size)
- **YOLOv8 Output Understanding**: Deep dive into anchor-free architecture - output tensor shape `(1, 84, 4032)` where:
  - 4,032 grid cells across 3 scales (fine/medium/coarse)
  - 84 values per cell (4 bbox coords + 80 COCO class scores)
  - Each grid cell makes one bounding box prediction with multi-class probabilities
- **Confidence Scores**: Learned how confidence scores work during inference without ground truth - network reports similarity to learned patterns
- **Real-Time Detection Pipeline**: Built complete pipeline in `oakd_camera_replay_tracking.py`:
  - Neural network inference on OAK-D camera feed
  - Filtering detections by confidence threshold and target classes
  - Drawing bounding boxes with class labels overlaid on video
  - Understanding absolute pixel coordinates vs normalized coordinates

## Screenshots & Visuals

### Early Development

Camera feed from OAK-D camera overlaid with bounding boxes from converted model and filtered classes.

<img src="other/images_md/Screenshot 2025-12-03 190546.png" alt="Screenshot" width="600">

---

## Entry 3: Familiarisation With NN Outputs
*Date: December 1, 2025*

### Semi-Custom Model

Covered the following:

- **blob conversion**: Tested converting non-included model (yolov8n) to blob format to use as "custom model"
- **model output**: Familiarising with NN output from yolov8n model on camera output. Yolov8n theory. Matrices... omg I'm back in uni.
- **OAK-D pipeline**: Exploring depthai pipeline capabilities

Considerations:

- **feature maps**: Creating custom anchor boxes on the feature maps for on water boat tracking for boats > 500m away. Yolov8n is **anchor free**.

---

## Entry 2: Modern Python Tooling
*Date: November 29, 2025*

### Setting Up the Project Infrastructure

Covered the following:

- **pyproject.toml**: Migrated from legacy `setup.py` to the modern standard
- **Ruff**: Adopted as the all-in-one linter and formatter (replacing flake8, isort, black)
- **Project Structure**: Organized code into proper packages with clear separation of concerns
- **OAK-D Connection**: Used depthai library to connect to OAK-d camera

---

## Screenshots & Visuals

### Early Development

Booted up the OAK-D camera and connected it to the Viewer app. Holy crap what have I gotten myself into with this project.

<img src="other/images_md/Screenshot%202025-11-29%20134603.png" alt="Screenshot" width="600">

Connected to OAK-d cameras using depthai library.

<img src="other/images_md/Screenshot 2025-11-29 163821.png" alt="Screenshot" width="600">

---

## Entry 1: Getting Started
*Date: November 23, 2025*

### Familiarisation With Open Source Tools

Initial set up learning modern Python tooling and open source computer vision techniques while waiting for OAK-D camera to be delivered.

### What I'm Building

- **Boat Detection**: Using YOLOv8 for real-time boat detection
- **Tracking System**: Implementing BoT-SORT for multi-object tracking
- **MLflow Integration**: Experiment tracking and model management
- **Video Processing Pipeline**: Efficient batch processing of video footage