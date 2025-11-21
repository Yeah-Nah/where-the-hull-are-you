# MLflow Experiment Tracking - Usage Guide

This guide demonstrates how to use the MLflow integration for hyperparameter tuning in the boat tracking system.

## Installation

First, install the updated requirements:

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Run with Metrics Collection (No MLflow)

```python
from src.config.settings import *
from src.models.detector import Detector
from src.models.tracker import Tracker
from src.processors.batch_processor import BatchProcessor

# Initialize detector and tracker
detector = Detector(
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    target_classes=BOAT_CLASSES
)

tracker = Tracker(tracker_type='botsort', config=BOTSORT_CONFIG)

# Run batch processing with metrics collection enabled
batch_processor = BatchProcessor(
    input_dir=INPUT_PREPROCESSED_DIR,
    output_dir=OUTPUT_DIR,
    detector=detector,
    tracker=tracker,
    collect_metrics=True  # Enable metrics collection
)

metrics = batch_processor.run()

# Print aggregated metrics
print(metrics['aggregated'])
```

### 2. Run Single Experiment with MLflow

```python
from src.experiments.mlflow_runner import MLflowRunner
from src.config.settings import *

# Initialize MLflow runner
mlflow_runner = MLflowRunner(
    experiment_name="boat-tracking-experiments",
    tracking_uri="file:./mlruns"
)

# Define hyperparameters
hyperparameters = {
    'confidence_threshold': 0.5,
    'target_height': 720,
    'tracker_type': 'botsort',
    'botsort_config': {
        'track_thresh': 0.5,
        'track_buffer': 30,
        'match_thresh': 0.8
    }
}

# Run experiment
metrics = mlflow_runner.run_experiment(
    hyperparameters=hyperparameters,
    input_dir=str(INPUT_PREPROCESSED_DIR),
    output_dir=str(OUTPUT_DIR),
    model_path=str(MODEL_PATH),
    boat_classes=BOAT_CLASSES,
    run_name="my_experiment"
)
```

### 3. Run Experiment from YAML Config

```python
from src.experiments.mlflow_runner import MLflowRunner
from src.config.settings import *

mlflow_runner = MLflowRunner(experiment_name="boat-tracking-experiments")

# Run baseline experiment
metrics = mlflow_runner.run_experiment_from_config(
    config_path="config/experiments/baseline.yaml",
    input_dir=str(INPUT_PREPROCESSED_DIR),
    output_dir=str(OUTPUT_DIR),
    model_path=str(MODEL_PATH),
    boat_classes=BOAT_CLASSES
)
```

### 4. Run Multiple Named Experiments

```python
from pathlib import Path
from src.experiments.mlflow_runner import MLflowRunner
from src.config.settings import *

mlflow_runner = MLflowRunner(experiment_name="boat-tracking-experiments")

# Run all experiment configurations
for config_file in Path("config/experiments").glob("*.yaml"):
    print(f"\nRunning experiment: {config_file.stem}")
    mlflow_runner.run_experiment_from_config(
        config_path=str(config_file),
        input_dir=str(INPUT_PREPROCESSED_DIR),
        output_dir=str(OUTPUT_DIR),
        model_path=str(MODEL_PATH),
        boat_classes=BOAT_CLASSES
    )
```

### 5. Run Grid Search

```python
from src.experiments.mlflow_runner import MLflowRunner
from src.config.settings import *

mlflow_runner = MLflowRunner(experiment_name="boat-tracking-grid-search")

# Run grid search (generates all combinations)
results = mlflow_runner.run_grid_search(
    config_path="config/hyperparameter_search.yaml",
    input_dir=str(INPUT_PREPROCESSED_DIR),
    output_dir=str(OUTPUT_DIR),
    model_path=str(MODEL_PATH),
    boat_classes=BOAT_CLASSES
)

print(f"Completed {len(results)} experiments")
```

## Viewing Results in MLflow UI

Start the MLflow UI to view experiment results:

```bash
cd boat-tracking-system
mlflow ui
```

Then open your browser to `http://localhost:5000`

## Metrics Logged

### Aggregated Metrics (across all videos)

- `mean_detection_confidence_avg` - Average detection confidence
- `bbox_area_mean_avg` - Average bounding box size (normalized)
- `avg_track_length_avg` - Average track length in frames
- `track_fragmentation_rate_avg` - Track fragmentation rate (tracks/frames)
- `frame_to_frame_iou_mean_avg` - Frame-to-frame IoU for tracks
- `processing_fps_avg` - Processing speed (frames/second)
- `short_track_ratio_avg` - Ratio of tracks < 5 frames
- `total_processing_time_avg` - Average processing time per video

Each metric also includes `_std`, `_min`, `_max`, and `_median` statistics.

### Per-Video Metrics

Detailed metrics for each video are saved as artifacts in `per_video_metrics.json`.

## Comparing Experiments

In the MLflow UI:

1. Select multiple runs using checkboxes
2. Click "Compare" button
3. View parallel coordinates plot and metric comparisons
4. Sort by any metric to find best configurations

## Creating Custom Experiment Configs

Create a new YAML file in `config/experiments/`:

```yaml
# config/experiments/my_experiment.yaml
experiment_name: my_custom_experiment

confidence_threshold: 0.6
target_height: 720
tracker_type: botsort

botsort_config:
  track_thresh: 0.55
  track_buffer: 45
  match_thresh: 0.82
  det_thresh: 0.55
```

Then run it:

```python
mlflow_runner.run_experiment_from_config(
    config_path="config/experiments/my_experiment.yaml",
    input_dir=str(INPUT_PREPROCESSED_DIR),
    output_dir=str(OUTPUT_DIR)
)
```

## Backward Compatibility

The existing notebook code continues to work without any changes:

```python
# This still works exactly as before
batch_processor = BatchProcessor(INPUT_DIR, OUTPUT_DIR, detector, tracker)
batch_processor.run()
```

Metrics collection and MLflow are completely opt-in.
