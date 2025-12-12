# Shared Tracking Metrics

Reusable tracking metrics library for boat tracking projects.

## Overview

This package provides a unified interface for collecting, calculating, visualizing, and logging tracking metrics across different projects:
- `oakd-camera-tracking` - Live camera inference
- `video-evaluation` - Batch video evaluation
- `model-training` - Custom model training validation

## Features

- **Collectors**: Gather detections and tracks over time
- **Calculators**: Compute metrics (MOTA, IDF1, confidence, stability)
- **Visualizers**: Overlay metrics on video frames
- **Loggers**: MLflow integration for experiment tracking

## Installation

```bash
# Install in editable mode for development
pip install -e .
```

## Usage

```python
from tracking_metrics import TrackingMetricsCollector, MetricsCalculator

# Collect tracking data
collector = TrackingMetricsCollector()
collector.add_detection_with_track(frame_id=0, detections=[...])

# Calculate metrics
calculator = MetricsCalculator(collector)
metrics = calculator.compute_all_metrics()
print(metrics)
```

## Development

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=tracking_metrics tests/
```
