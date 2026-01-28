# Where The Hull Are You

A real-time maritime object detection and tracking system built for embedded vision hardware. This project explores the intersection of computer vision, robotics, and edge AI by developing a complete pipeline from model training to deployment on OAK-D cameras.

### **Check out my progress updates here:** ([`👉 Progress Updates 👈`](PROCESS_UPDATES.md))

## Project Vision

Develop an intelligent system capable of detecting and tracking boats in real-time using custom-trained models deployed on edge devices. The project emphasizes practical robotics skills, sensor integration, and modern ML workflows.

### Learning Objectives

Inspired by advanced robotics and ML engineering roles, this project focuses on:
- **Edge AI Deployment**: Running custom models on OAK-D camera hardware
- **Model Training & Evaluation**: Training YOLO models on maritime datasets with comprehensive evaluation
- **Experiment Tracking**: MLflow integration for reproducible ML workflows
- **Modular Architecture**: Shared metrics libraries and clean separation of concerns
- **Future Direction**: ROS2, sensor fusion, and multi-modal tracking 😨😅👏

## Architecture Overview

The project is organized into three main components:

### 1. **Model Training Pipeline** ([`model-training/`](model-training))
- Train custom YOLOv8 models for maritime object detection
- Hyperparameter search with automated experiment tracking
- Evaluate models on unlabeled boat footage using confidence-based metrics
- Compare performance across configurations and select optimal models

**Key Features:**
- Grid search for BoTSORT tracker and model parameters
- Unlabeled evaluator with confidence, track stability, and bbox quality metrics
- MLflow experiment logging and comparison
- Batch video processing with weighted metric aggregation
- Model export to OAK-D compatible formats (.blob)

### 2. **Shared Tracking Metrics** ([`shared-tracking-metrics/`](shared-tracking-metrics))
- Reusable metrics library for tracking quality assessment
- Unified interface across training, evaluation, and deployment
- Installable package shared across all project components

**Key Features:**
- Collectors for tracking detections and track histories
- Calculators for confidence, bbox stability, track coverage metrics
- Generic YOLO inference wrapper for any frame source
- Visualizers for metric overlays on video
- MLflow loggers for experiment tracking

### 3. **OAK-D Camera Tracking** ([`oakd-camera-tracking/`](oakd-camera-tracking))
- Real-time detection and tracking on OAK-D hardware
- Pipeline for live camera and recorded video processing
- Integration with custom-trained models (.blob format)

**Key Features:**
- DepthAI integration for OAK-D cameras
- Detection pipeline with configurable models
- Support for camera and video replay modes

## Technology Stack

- **Computer Vision**: YOLOv8, OpenCV, Ultralytics
- **Edge AI**: OAK-D cameras, DepthAI, OpenVINO model format
- **ML Workflow**: MLflow for experiment tracking and model registry
- **Development**: Python 3.13+, Ruff (linting/formatting), pytest
- **Packaging**: Modern `pyproject.toml` configuration

## Project Structure

```
where-the-hull-are-you/
├── model-training/              # Custom model training and evaluation
│   ├── src/                     # Training, evaluation, data utilities
│   │   ├── config/              # Configuration loaders
│   │   ├── data/                # Video loading utilities
│   │   ├── evaluation/          # Unlabeled evaluator with hyperparam search
│   │   └── training/            # Training orchestration
│   ├── notebooks/               # Experimentation notebooks
│   │   ├── 01_train_custom_model.ipynb
│   │   ├── 02_evaluate_models.ipynb
│   │   └── 03_hyperparameter_search.ipynb
│   ├── config/                  # Model and search space configs (YAML)
│   ├── models/                  # Trained model weights (.pt, .blob)
│   └── output/                  # MLflow runs and metrics
├── shared-tracking-metrics/     # Reusable metrics library
│   ├── src/tracking_metrics/    # Collectors, calculators, inference, loggers
│   └── tests/                   # Unit tests
├── oakd-camera-tracking/        # OAK-D deployment pipeline
│   ├── src/                     # Detection pipeline and config
│   └── models/                  # Converted .blob models
└── pyproject.toml               # Project dependencies and tooling
```

## Development Status

### Completed
- ✅ Model training pipeline with custom datasets
- ✅ Hyperparameter search with grid search across BoTSORT parameters
- ✅ Unlabeled evaluation using confidence and tracking stability metrics
- ✅ Shared metrics library for tracking quality assessment
- ✅ OAK-D camera integration and detection pipeline
- ✅ MLflow experiment tracking and comparison
- ✅ Batch video processing with weighted metric aggregation
- ✅ Modern Python tooling (pyproject.toml, Ruff, GitHub Actions)

### In Progress
- 🔄 Model optimization and selection using hyperparam search results
- 🔄 Real-time tracking performance optimization
- 🔄 Comprehensive test coverage

### Future Roadmap
- 📋 ROS2 integration for robotics workflows
- 📋 Multi-sensor fusion (camera + IMU + depth)
- 📋 Containerization and deployment strategies
- 📋 Web-based visualization dashboard

## Getting Started

### Installation

1. **Install shared metrics library** (required by other components):
```bash
cd shared-tracking-metrics
pip install -e .
```

2. **Install model training pipeline**:
```bash
cd ../model-training
pip install -e .
```

3. **Run hyperparameter search**:
```bash
# Configure search space in config/hyperparam_search_config.yaml
# Run notebook: notebooks/03_hyperparameter_search.ipynb

# View results in MLflow UI
cd model-training
mlflow ui --backend-store-uri file:output/mlruns
# Open http://localhost:5000
```

**Note**: Python 3.13+ required. See individual component READMEs for detailed usage.

## Contributing

Contributions welcome! This is a learning project focused on practical robotics and ML engineering skills.

## License

MIT License - See LICENSE file for details.

---

*A hands-on exploration of edge AI, computer vision, and robotics engineering.*
