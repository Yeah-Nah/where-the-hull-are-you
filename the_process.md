# The Process: Building Where The Hull Are You

A development journey blog documenting the creation of a boat tracking system.

---

# Phase 0: Set Up Local Dev Env

## Entry 1: Getting Started
*Date: November 23, 2025*

### Familiarisation With Open Source Tools

Initial set up learning modern Python tooling and open source computer vision techniques while waiting for OAK-D camera to be delivered.

### What I'm Building

- **Boat Detection**: Using YOLOv8 for real-time boat detection
- **Tracking System**: Implementing BoT-SORT for multi-object tracking
- **MLflow Integration**: Experiment tracking and model management
- **Video Processing Pipeline**: Efficient batch processing of video footage

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

## Entry 3: Familiarisation With NN Outputs
*Date: December 1, 2025*

### Semi-Custom Model

Covered the following:

- **blob conversion**: Converted yolov8n model (not already pre-loaded from depthai API) to blob format to use in pipeline
- **model output**: Familiarising with NN output from yolov8n model on camera output. Yolov8n theory.
- **OAK-D pipeline**: Exploring depthai pipeline capabilities

Considerations:

- **feature maps**: Customising smaller feature maps for on water boat tracking for boats > 500m away