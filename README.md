# Where The Hull Are You

A real-time maritime object detection and tracking system built for embedded vision hardware. This project explores the intersection of computer vision, robotics, and edge AI by developing a complete pipeline from model training to deployment on OAK-D cameras.

## Project Vision

Develop an intelligent system capable of detecting and tracking boats in real-time using custom-trained models deployed on edge devices. The project emphasizes practical robotics skills, sensor integration, and modern ML workflows.

### Learning Objectives

Inspired by advanced robotics and ML engineering roles, this project focuses on:
- **Edge AI Deployment**: Running custom models on OAK-D camera hardware
- **Model Training & Evaluation**: Training YOLOv8 models on maritime datasets with comprehensive evaluation
- **Experiment Tracking**: MLflow integration for reproducible ML workflows
- **Modular Architecture**: Shared metrics libraries and clean separation of concerns
- **Future Direction**: ROS2, sensor fusion, and multi-modal tracking

## Architecture Overview

The project is organized into three main components:

### 1. **Model Training Pipeline** ([`model-training/`](model-training))
- Train custom YOLOv8 models for maritime object detection
- Evaluate models on labeled and unlabeled boat footage
- Compare performance metrics (mAP, precision, recall, confidence)
- Select optimal models for deployment

**Key Features:**
- Custom training workflows with configurable hyperparameters
- Dual evaluation strategy (labeled ground truth + unlabeled confidence metrics)
- MLflow integration for experiment tracking
- Model export to OAK-D compatible formats

### 2. **Shared Tracking Metrics** ([`shared-tracking-metrics/`](shared-tracking-metrics))
- Reusable metrics library for tracking quality assessment
- Unified interface across training, evaluation, and deployment
- Collectors, calculators, visualizers, and MLflow loggers

**Key Features:**
- MOTA, IDF1, and custom tracking stability metrics
- Video overlay visualizations
- MLflow experiment logging
- Installable package for cross-project use

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
│   ├── notebooks/               # Experimentation notebooks
│   ├── config/                  # Training/eval configuration
│   └── models/                  # Trained model weights
├── shared-tracking-metrics/     # Reusable metrics library
│   ├── src/tracking_metrics/    # Collectors, calculators, loggers
│   └── tests/                   # Unit tests
├── oakd-camera-tracking/        # OAK-D deployment pipeline
│   ├── src/                     # Detection pipeline and config
│   └── models/                  # Converted .blob models
└── pyproject.toml               # Project dependencies and tooling
```

## Development Status

### Completed
- ✅ Model training pipeline with custom datasets
- ✅ Dual evaluation strategy (labeled + unlabeled)
- ✅ Shared metrics library for tracking quality
- ✅ OAK-D camera integration and detection pipeline
- ✅ MLflow experiment tracking
- ✅ Modern Python tooling (pyproject.toml, Ruff)

### In Progress
- 🔄 Model export to OAK-D .blob format
- 🔄 Real-time tracking performance optimization
- 🔄 Comprehensive test coverage

### Future Roadmap
- 📋 ROS2 integration for robotics workflows
- 📋 Multi-sensor fusion (camera + IMU + depth)
- 📋 Containerization and deployment strategies
- 📋 Web-based visualization dashboard

## Getting Started

Each component can be installed independently:

```bash
# Install shared metrics library
cd shared-tracking-metrics
pip install -e .

# Install model training tools
cd ../model-training
pip install -e ".[dev]"

# Install OAK-D tracking pipeline
cd ../oakd-camera-tracking
pip install -e .
```

**Note**: This project is under active development. Full deployment instructions will be provided as components stabilize.

## Contributing

Contributions welcome! This is a learning project focused on practical robotics and ML engineering skills.

## License

MIT License - See LICENSE file for details.

---

*A hands-on exploration of edge AI, computer vision, and robotics engineering.*
