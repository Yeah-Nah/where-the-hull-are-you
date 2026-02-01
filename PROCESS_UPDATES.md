# The Process: Building Where The Hull Are You

A development journey blog documenting the creation of a boat tracking system.

---

# Phase 1: Setting Up Model Training And Performance Tracking Pipeline

## Entry 9: Training Data Pipeline & Label Studio Setup
*Date: January 30-31, 2026*

### Frame Extraction & Docker-Based Annotation Workflow

Built custom training data pipeline and set up automated annotation infrastructure:

- **Frame Extraction Pipeline**:
  - Created pipeline to extract frames from video for training data generation
  - Implemented configurable frame extraction in `frame_extractor.py` with video processing utilities
  - Built notebook interface (`01_extract_frames.ipynb`) for managing extraction workflow
- **Label Studio Setup with Docker**:
  - Gained a little experience with Docker containers through Label Studio deployment
  - Set up Label Studio and Label Studio ML backend as containerized services 👏👏
  - Configured auto-annotation workflow using YOLO model to pre-label extracted frames - did ok at prelabelling

Result: Started the creation of custom training datasets for boat detection using open source software and gained experience with Docker containers.

Connected YOLO model to Label Studio UI:

<img src="other/images_md/Screenshot 2026-02-01 194859.png" alt="Screenshot" width="600">

Used it to help prelabel training data:

<img src="other/images_md/Screenshot 2026-02-01 202423.png" alt="Screenshot" width="600">

---

## Entry 8: Code Quality, CI/CD & Configuration Improvements
*Date: January 28-29, 2026*

### Pre-commit Hooks, Type Safety & Project Restructuring

Major codebase cleanup focused on code quality automation, type safety, and improved project structure:

- **Pre-commit Hooks & Github Actions**:
  - Configured comprehensive pre-commit hooks: Ruff linting/formatting, YAML validation, trailing whitespace removal, merge conflict detection
  - Added Jupyter notebook linting with nbqa-ruff for maintaining notebook code quality
  - Implemented per-package MyPy type checking (shared-tracking-metrics, model-training, oakd-camera-tracking)
  - Created Github Actions workflow with separate linting and validation jobs running on PRs and main branch pushes
- **Type Safety & Code Quality**:
  - Added strict type hints across evaluation, tracking metrics, and model conversion modules
  - Enhanced type annotations in `unlabeled_evaluator.py`, calculators, collectors, and inference classes
  - Improved docstrings and function signatures for better code documentation
  - Fixed MyPy to run separately per package to avoid namespace conflicts
- **Project Structure Improvements**:
  - Deleted 200+ lines of moot code: removed `labeled_evaluator.py`, `video_loader.py`, outdated READMEs, and legacy `setup.py` files
  - Restructured config file locations for better organization across all packages
  - Cleaned up unused imports and consolidated evaluation module exports
  - Moved settings files to package roots for cleaner import paths
- **Configuration Validation & Mlflow**:
  - Added import-time model path validation in `settings.py` with extension checking (`.pt`)
  - Changed Mlflow logging from full path to model name (e.g., "yolo11n.pt") for portability
  - Added UTC datetime suffix to experiment run names: `exp_0001_20260129143022`
  - Prevents duplicate run names and provides better experiment organization

Result: Automated code quality enforcement, comprehensive type checking, streamlined project structure, and fail-fast configuration validation.

Basic Github actions in action:

<img src="other/images_md/Screenshot 2026-01-24 212430.png" alt="Screenshot" width="600">

Mlflow experiments logged with unlabelled data performance metrics and accessed through the Mlflow UI:

<img src="other/images_md/Screenshot 2026-01-25 121554.png" alt="Screenshot" width="600">

---

## Entry 7: Hyperparameter Search & Code Quality
*Date: January 24, 2026*

### Mlflow Experiment Tracking & Pythonic Refactoring

Implemented comprehensive hyperparameter search infrastructure with experiment tracking alongside major code quality improvements:

- **Hyperparameter Search Pipeline**:
  - Built grid search functionality for BoTSORT tracker and model parameters
  - Implemented nested config handling (`_flatten_search_space()`, `_reconstruct_nested_config()`) to support hierarchical parameter spaces
  - Created `search()` method to orchestrate experiments across all parameter combinations
  - Added configurable Mlflow integration for tracking experiment results
- **Mlflow Experiment Logging**:
  - Integrated Mlflow tracking URI and experiment naming
  - Automated parameter and metric logging for each experiment run
  - Added error handling and failure logging for robust experiment tracking
  - Organized experiments with sequential naming (exp_0001, exp_0002, etc.)
- **Code Quality & Refactoring**:
  - Fixed 2 critical bugs (undefined `model_config` variable, incorrect return type)
  - Reorganized imports following PEP 8 standards
  - Added class-level constants for magic values and weighted metrics
  - Extracted helper methods to eliminate code duplication (`_process_batch()`, `_calculate_weighted_metric()`)
  - Applied Pythonic patterns (dict comprehensions, enumerate, concise conditionals)
  - Removed debug print statements and improved error handling

Result: Hyperparameter optimization system with clean, maintainable code that tracks all experiments systematically.

Mlflow logging weighted metrics for hyperparameter search on multiple videos:

<img src="other/images_md/Screenshot 2026-01-24 193431.png" alt="Screenshot" width="600">
---

## Entry 6: Batch Video Tracking Metrics
*Date: December 16, 2025*

Building on the single-video tracking metrics from Entry 5, implemented batch processing capabilities:

- **Multi Video Evaluation**: Extended the tracking metrics pipeline to batch process videos
  - Implemented shared metric calculation logic to avoid code duplication
  - Added progress tracking and error handling for robust batch operations
- **Consolidated Metrics Output**: Aggregated tracking metrics across multiple videos
  - Standardized metric collection format for easy comparison between videos
  - Generated summary statistics across entire video batches
- **Performance Optimization**: Improved processing speed for large video datasets
  - Created batch processing utilities for batching frames and detections to increase efficiency
  - Leveraged parallel processing where applicable
  - Optimized memory usage for handling multiple concurrent video streams

Aggregated and weighted metrics across multiple videos:

<img src="other/images_md/Screenshot 2025-12-16 164325.png" alt="Screenshot" width="600">

---

## Entry 5: Tracking Metrics & Inference Pipeline Progress
*Date: December 12, 2025*

### Calculating tracking metrics on video files

- **Collectors, calculators**: Built out a reusable tracking metrics library
  - Added track history collection to support metrics like MOTA/IDF1 and stability
  - Implemented initial metrics computation from tracking outputs
- **Inference Tooling**: Reusable inference to to apply YOLO to any frame
  - Created a `ModelInference` class for applying tracking to single videos
  - Added a video reader utility and streamlined config files across projects
- **Evaluators**: Classes for evaluating labelled and unlabelled video data
- **Codebase hygiene and consistency**
  - Applied Ruff linting/formatting across modules for consistent style
  - Consolidated configs to simplify experimentation and deployment

Tracking metrics from an unlabelled test video containing an 18ft skiff:

<img src="other/images_md/Screenshot 2025-12-12 221810.png" alt="Screenshot" width="600">

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
- **Mlflow Integration**: Experiment tracking and model management
- **Video Processing Pipeline**: Efficient batch processing of video footage
