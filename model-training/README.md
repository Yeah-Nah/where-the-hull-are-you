# Model Training Pipeline

Train and evaluate custom maritime object detection models.

## Overview

This project provides tools to:
- Train custom YOLO models for boat detection and tracking
- Evaluate models on labeled datasets with ground truth
- Evaluate models on unlabeled boat footage using confidence-based metrics
- Compare model performance and select the best for deployment

## Project Structure

```
model-training/
├── src/
│   ├── training/          # Training orchestration
│   ├── evaluation/        # Model evaluation (labeled & unlabeled)
│   └── data/             # Data loading utilities
├── notebooks/
│   ├── 01_train_custom_model.ipynb
│   └── 02_evaluate_models.ipynb
├── config/               # Configuration files
├── models/               # Saved model weights
└── output/              # Evaluation results
```

## Dependencies

- `tracking-metrics` (shared package for metrics calculation)
- `ultralytics` (YOLO models)
- `mlflow` (experiment tracking)
- `opencv-python` (video processing)

## Installation

```bash
# Install shared tracking metrics
cd ../shared-tracking-metrics
pip install -e .

# Install model training dependencies
cd ../model-training
pip install -e .
```

## Usage

### 1. Configure Training

Edit `config/training_config.yaml` with your dataset paths and hyperparameters.

### 2. Train Model

Run `notebooks/01_train_custom_model.ipynb` to train a custom model.

### 3. Evaluate Models

Edit `config/evaluation_config.yaml` with your test data paths, then run `notebooks/02_evaluate_models.ipynb` to compare models.

## Workflow

```
Train custom model → Evaluate on labeled data → Evaluate on unlabeled videos
    ↓                        ↓                           ↓
Save weights         Get mAP, precision, recall    Get confidence metrics
    ↓                        ↓                           ↓
                    Compare with baseline (YOLOv8n)
                              ↓
                    Select best model for OAK-D deployment
```
