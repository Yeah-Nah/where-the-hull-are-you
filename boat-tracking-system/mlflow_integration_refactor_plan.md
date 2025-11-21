# MLflow Integration Refactor Plan

## Executive Summary

This document outlines a refactor plan to integrate MLflow experiment tracking into the existing boat tracking system while preserving all current functionality. The refactor will enable hyperparameter tuning with comprehensive metrics collection and aggregation.

---

## 1. Architecture Overview

### Current Architecture
```
BatchProcessor
  └── VideoProcessor (per video)
       └── Detector + Tracker (per frame)
```

### Proposed Architecture
```
MLflowExperimentRunner
  └── BatchProcessor (with MetricsCollector)
       └── VideoProcessor (with VideoMetricsCollector)
            └── Detector + Tracker (per frame)
```

**Key Change**: Wrap existing pipeline with metrics collection layers without modifying core detection/tracking logic.

---

## 2. New Modules to Create

### 2.1 `src/metrics/video_metrics_collector.py`
**Purpose**: Collect metrics for a single video during processing.

**Responsibilities**:
- Track per-frame detection data (confidence scores, bounding boxes)
- Track per-frame tracking data (track IDs, IoU calculations)
- **Record total video processing time** (measured once per video, not per frame)
- Calculate video-level aggregated metrics

**Key Methods**:
```
- add_frame_detections(detections)
- add_frame_tracks(tracks, previous_tracks)
- set_total_processing_time(total_time, total_frames)
- compute_video_metrics() -> dict
```

**Metrics Computed**:
- Mean detection confidence
- Bbox size statistics (mean, std, median)
- Average track length
- Track fragmentation rate
- Frame-to-frame IoU
- Processing FPS **(calculated from total time and frame count)**
- Short track ratio (< 5 frames)
- Total processing time

**Performance Optimization**: Processing time is measured once per video rather than per frame to minimize overhead. FPS is calculated as `total_frames / total_processing_time`.

### 2.2 `src/metrics/metrics_aggregator.py`
**Purpose**: Aggregate metrics across multiple videos.

**Responsibilities**:
- Collect metrics from multiple VideoMetricsCollectors
- Compute cross-video statistics (mean, median, std)
- Prepare final metrics dictionary for MLflow logging

**Key Methods**:
```
- add_video_metrics(video_name, metrics_dict)
- compute_aggregated_metrics() -> dict
- get_per_video_metrics() -> dict
```

**Output Format**:
```python
{
    'aggregated': {
        'mean_detection_confidence_avg': float,
        'mean_detection_confidence_std': float,
        'avg_track_length_avg': float,
        ...
    },
    'per_video': {
        'video1.mp4': {...},
        'video2.mp4': {...}
    }
}
```

### 2.3 `src/experiments/mlflow_runner.py`
**Purpose**: Orchestrate MLflow experiment runs with hyperparameter configurations.

**Responsibilities**:
- Initialize MLflow experiment
- Start/end MLflow runs
- **Load hyperparameter configurations from YAML files**
- Log hyperparameters
- Log metrics (both aggregated and per-video)
- Log artifacts (config files, sample outputs)
- Handle multiple experiment runs

**Key Methods**:
```
- setup_experiment(experiment_name)
- load_hyperparameter_config(config_path)
- run_experiment(hyperparameters, input_dir, output_dir)
- run_experiment_from_config(config_path, input_dir, output_dir)
- log_hyperparameters(params)
- log_metrics(metrics_dict)
- log_artifacts(artifact_paths)
```

### 2.4 `src/utils/metrics_calculator.py`
**Purpose**: Pure utility functions for metric calculations.

**Responsibilities**:
- Calculate IoU between bounding boxes
- Compute track statistics (length, fragmentation)
- Normalize bounding box sizes
- Calculate temporal consistency metrics

**Key Functions**:
```
- calculate_iou(box1, box2) -> float
- normalize_bbox_area(bbox, frame_width, frame_height) -> float
- calculate_track_fragmentation(track_ids, total_frames) -> float
- compute_track_lengths(track_history) -> dict
```

---

## 3. Modifications to Existing Classes

### 3.1 `BatchProcessor` (`src/processors/batch_processor.py`)

**Changes**:
- Add optional `metrics_aggregator` parameter to `__init__`
- Create `MetricsAggregator` instance if metrics collection is enabled
- Pass `metrics_aggregator` to each `VideoProcessor`
- Return aggregated metrics after processing all videos

**Modified Methods**:
```python
def __init__(self, input_dir, output_dir, detector, tracker, collect_metrics=False):
    # ... existing code ...
    self.collect_metrics = collect_metrics
    self.metrics_aggregator = MetricsAggregator() if collect_metrics else None

def process_videos(self):
    # ... existing video loop ...
    # Add: pass metrics collector to VideoProcessor
    video_processor = VideoProcessor(
        input_path=input_path,
        output_path=output_path,
        detector=self.detector,
        tracker=self.tracker,
        metrics_collector=VideoMetricsCollector(video_file) if self.collect_metrics else None
    )
    
    # After processing each video, collect metrics
    if self.collect_metrics and video_processor.metrics_collector:
        video_metrics = video_processor.metrics_collector.compute_video_metrics()
        self.metrics_aggregator.add_video_metrics(video_file, video_metrics)

def run(self):
    self.process_videos()
    if self.collect_metrics:
        return self.metrics_aggregator.compute_aggregated_metrics()
    return None
```

**Backward Compatibility**: Default `collect_metrics=False` preserves existing behavior.

### 3.2 `VideoProcessor` (`src/processors/video_processor.py`)

**Changes**:
- Add optional `metrics_collector` parameter to `__init__`
- Collect frame-level data during processing loop
- Record video start/end time only
- Store track history for IoU calculations

**Modified Methods**:
```python
def __init__(self, input_path, output_path, detector, tracker, metrics_collector=None):
    # ... existing code ...
    self.metrics_collector = metrics_collector
    self.previous_tracks = {}  # For IoU calculation

def process_video(self):
    # ... existing setup code ...
    
    import time
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Existing tracking code
        results = self.tracker.track(...)
        
        # NEW: Collect metrics if enabled
        if self.metrics_collector:
            # Extract detection data
            detections = self._extract_detections_from_results(results, frame.shape)
            self.metrics_collector.add_frame_detections(detections)
            
            # Extract tracking data
            tracks = self._extract_tracks_from_results(results)
            self.metrics_collector.add_frame_tracks(tracks, self.previous_tracks)
            self.previous_tracks = tracks
        
        # ... existing output writing code ...
    
    # Record total time (once per video)
    if self.metrics_collector:
        total_time = time.time() - start_time
        self.metrics_collector.set_total_processing_time(total_time, frame_count)
    
    # ... existing cleanup code ...

def _extract_detections_from_results(self, results, frame_shape):
    """Extract detection info from YOLO results."""
    # Return list of dicts with bbox, confidence, class_id
    pass

def _extract_tracks_from_results(self, results):
    """Extract tracking info from YOLO results."""
    # Return dict mapping track_id to bbox
    pass
```

**Backward Compatibility**: `metrics_collector=None` preserves existing behavior.

### 3.3 `Detector` (`src/models/detector.py`)

**Changes**: None required. All detection data is already accessible through YOLO results.

### 3.4 `Tracker` (`src/models/tracker.py`)

**Changes**: None required. All tracking data is already accessible through YOLO results.

### 3.5 `settings.py` (`src/config/settings.py`)

**Changes**:
- Add MLflow configuration parameters
- Add experiment tracking settings

**New Configuration**:
```python
# MLflow settings
MLFLOW_TRACKING_URI = config.get('mlflow_tracking_uri', 'file:./mlruns')
MLFLOW_EXPERIMENT_NAME = config.get('mlflow_experiment_name', 'boat-tracking-experiments')
MLFLOW_ARTIFACT_LOCATION = config.get('mlflow_artifact_location', None)

# Experiment settings
ENABLE_METRICS_COLLECTION = config.get('enable_metrics_collection', True)
```

**Update `config.yaml`**:
```yaml
# Existing settings...

# MLflow Configuration
mlflow_tracking_uri: "file:./mlruns"
mlflow_experiment_name: "boat-tracking-experiments"
enable_metrics_collection: true
```

---

## 4. Data Flow for Metrics Collection

### 4.1 Frame-Level Data Collection
```
Frame → Detector/Tracker → Results
                              ↓
                    VideoMetricsCollector
                    - Extracts: confidence, bbox, track_id
                    - Stores: per-frame data
                    - Calculates: frame-to-frame IoU
```

### 4.2 Video-Level Aggregation
```
VideoMetricsCollector (end of video)
  ↓
compute_video_metrics()
  - Aggregates frame data
  - Calculates statistics (mean, std, median)
  - Computes track-level metrics
  ↓
Returns: video_metrics_dict
```

### 4.3 Cross-Video Aggregation
```
MetricsAggregator (after all videos)
  ↓
compute_aggregated_metrics()
  - Collects all video metrics
  - Computes cross-video statistics
  - Separates aggregated vs per-video metrics
  ↓
Returns: {aggregated: {...}, per_video: {...}}
```

### 4.4 MLflow Logging
```
MLflowRunner
  ↓
mlflow.start_run()
  ↓
log_params(hyperparameters)
  ↓
BatchProcessor.run() → aggregated_metrics
  ↓
log_metrics(aggregated_metrics['aggregated'])
  ↓
log_artifact(per_video_metrics.json)
  ↓
mlflow.end_run()
```

---

## 5. MLflow Logging Strategy

### 5.1 What to Log

**Hyperparameters** (logged once per run):
```python
mlflow.log_param("confidence_threshold", 0.5)
mlflow.log_param("tracker_type", "botsort")
mlflow.log_param("target_height", 720)
mlflow.log_param("track_thresh", 0.5)  # from BOTSORT_CONFIG
mlflow.log_param("track_buffer", 30)   # from BOTSORT_CONFIG
mlflow.log_param("match_thresh", 0.8)  # from BOTSORT_CONFIG
# ... all BOTSORT_CONFIG params
```

**Aggregated Metrics** (logged once per run):
```python
mlflow.log_metric("mean_detection_confidence_avg", 0.85)
mlflow.log_metric("mean_detection_confidence_std", 0.05)
mlflow.log_metric("bbox_area_mean_avg", 0.02)
mlflow.log_metric("bbox_area_std_avg", 0.01)
mlflow.log_metric("avg_track_length_avg", 45.3)
mlflow.log_metric("track_fragmentation_rate_avg", 0.12)
mlflow.log_metric("frame_to_frame_iou_avg", 0.89)
mlflow.log_metric("processing_fps_avg", 28.5)
mlflow.log_metric("short_track_ratio_avg", 0.08)
mlflow.log_metric("total_processing_time", 125.4)
```

**Artifacts**:
```python
mlflow.log_artifact("per_video_metrics.json")  # Detailed per-video metrics
mlflow.log_artifact("config.yaml")              # Config snapshot
mlflow.log_artifact("model_config.yaml")        # Model config snapshot
```

### 5.2 When to Log

**Experiment Setup** (once):
```python
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
```

**Per Run**:
1. `mlflow.start_run()` - before processing videos
2. `log_params()` - immediately after start
3. `BatchProcessor.run()` - process all videos and collect metrics
4. `log_metrics()` - after processing complete
5. `log_artifacts()` - after metrics logged
6. `mlflow.end_run()` - cleanup

### 5.3 Where to Log

**Primary Location**: `MLflowRunner.run_experiment()`

**Rationale**: Centralize all MLflow operations in one place to:
- Avoid scattered logging calls
- Ensure consistent logging pattern
- Simplify error handling and rollback
- Make it easy to disable MLflow without changing core code

---

## 6. Hyperparameter Search Integration

### 6.1 Hyperparameter Configuration

**Location**: `config/hyperparameter_search.yaml`

**Rationale**: Keep hyperparameter configurations in the `config/` folder rather than source code to:
- Allow users to modify without touching Python code
- Separate configuration from implementation
- Enable version control of experiment configurations
- Support multiple experiment configuration files

**Example Format**:

**Option A**: Grid Search Configuration (`config/hyperparameter_search.yaml`)
```yaml
# Grid search configuration
search_type: grid
parameters:
  confidence_threshold: [0.3, 0.5, 0.7]
  target_height: [480, 720, 1080]
  botsort_config:
    track_thresh: [0.25, 0.5, 0.75]
    track_buffer: [30, 60, 90]
    match_thresh: [0.7, 0.8, 0.9]
```

**Option B**: Named Experiments (`config/experiments/baseline.yaml`, `config/experiments/high_confidence.yaml`)
```yaml
# config/experiments/baseline.yaml
experiment_name: baseline
confidence_threshold: 0.5
target_height: 720
botsort_config:
  track_thresh: 0.5
  track_buffer: 30
  match_thresh: 0.8
  det_thresh: 0.5
```

```yaml
# config/experiments/high_confidence.yaml
experiment_name: high_confidence
confidence_threshold: 0.7
target_height: 720
botsort_config:
  track_thresh: 0.6
  track_buffer: 30
  match_thresh: 0.85
  det_thresh: 0.6
```

### 6.2 Running Multiple Experiments

**Option A**: Grid search from config
```python
mlflow_runner = MLflowRunner(experiment_name="boat-tracking-grid-search")

# Load grid search configuration
mlflow_runner.run_grid_search(
    config_path="config/hyperparameter_search.yaml",
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR
)
```

**Option B**: Named configurations from individual files
```python
import glob
from pathlib import Path

mlflow_runner = MLflowRunner(experiment_name="boat-tracking-experiments")

# Run all experiment configurations in config/experiments/
for config_file in Path("config/experiments").glob("*.yaml"):
    mlflow_runner.run_experiment_from_config(
        config_path=str(config_file),
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )
```

**Option C**: Single experiment from config
```python
mlflow_runner = MLflowRunner(experiment_name="boat-tracking-experiments")
mlflow_runner.run_experiment_from_config(
    config_path="config/experiments/baseline.yaml",
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR
)
```

---

## 7. Minimal Code Changes Summary

### New Python Files (5)
1. `src/metrics/video_metrics_collector.py` (~200 lines)
2. `src/metrics/metrics_aggregator.py` (~150 lines)
3. `src/utils/metrics_calculator.py` (~100 lines)
4. `src/experiments/mlflow_runner.py` (~200 lines) - includes YAML loading
5. `src/metrics/__init__.py` (empty)

### New Configuration Files (3)
1. `config/hyperparameter_search.yaml` (example grid search config)
2. `config/experiments/baseline.yaml` (example named experiment)
3. `config/experiments/high_confidence.yaml` (example named experiment)

### Modified Files (4)
1. `src/processors/batch_processor.py` (~30 lines added)
2. `src/processors/video_processor.py` (~40 lines added) - optimized timing
3. `src/config/settings.py` (~10 lines added)
4. `config/config.yaml` (~5 lines added)

### Total New Code: ~750 lines
### Total Modified Code: ~85 lines
### Total New Config: ~100 lines (YAML)

---

## 8. Implementation Phases

### Phase 1: Metrics Collection Foundation
**Goal**: Implement metrics collection without MLflow.

**Tasks**:
1. Create `metrics_calculator.py` utility functions
2. Create `VideoMetricsCollector` class
3. Create `MetricsAggregator` class
4. Add unit tests for metric calculations

**Validation**: Run batch processor with metrics collection enabled, verify output format.

### Phase 2: Integrate Metrics into Pipeline
**Goal**: Wire metrics collectors into existing pipeline.

**Tasks**:
1. Modify `VideoProcessor` to accept and use `VideoMetricsCollector`
2. Modify `BatchProcessor` to accept and use `MetricsAggregator`
3. Add helper methods to extract data from YOLO results
4. Test end-to-end metrics collection

**Validation**: Process sample videos and verify all metrics are calculated correctly.

### Phase 3: MLflow Integration
**Goal**: Add MLflow logging infrastructure.

**Tasks**:
1. Add MLflow to `requirements.txt`
2. Create `MLflowRunner` class
3. Add MLflow configuration to `settings.py` and `config.yaml`
4. Implement logging methods

**Validation**: Run single experiment, verify metrics appear in MLflow UI.

### Phase 4: Hyperparameter Search
**Goal**: Enable multiple experiment runs.

**Tasks**:
1. Create example hyperparameter YAML configs (`config/hyperparameter_search.yaml`, `config/experiments/`)
2. Implement YAML config loading in `MLflowRunner`
3. Implement grid search or named configuration loops
4. Add experiment comparison utilities
5. Create notebook for hyperparameter analysis

**Validation**: Run multiple configurations from YAML files, compare results in MLflow UI.

### Phase 5: Documentation & Examples
**Goal**: Provide clear usage documentation.

**Tasks**:
1. Update `README.md` with MLflow instructions
2. Create example notebook demonstrating hyperparameter tuning
3. Document metric definitions and interpretations
4. Add troubleshooting guide

---

## 9. Backward Compatibility

**Key Principle**: All new functionality is opt-in via flags.

### Existing Usage (unchanged)
```python
detector = Detector(MODEL_PATH, CONFIDENCE_THRESHOLD, BOAT_CLASSES)
tracker = Tracker('botsort', BOTSORT_CONFIG)
batch_processor = BatchProcessor(INPUT_DIR, OUTPUT_DIR, detector, tracker)
batch_processor.run()  # Works exactly as before
```

### New Usage (with metrics)
```python
detector = Detector(MODEL_PATH, CONFIDENCE_THRESHOLD, BOAT_CLASSES)
tracker = Tracker('botsort', BOTSORT_CONFIG)
batch_processor = BatchProcessor(
    INPUT_DIR, OUTPUT_DIR, detector, tracker,
    collect_metrics=True  # NEW: Enable metrics
)
metrics = batch_processor.run()  # NEW: Returns metrics dict
```

### New Usage (with MLflow)
```python
mlflow_runner = MLflowRunner("my-experiment")
hyperparams = {
    'confidence_threshold': 0.5,
    'target_height': 720,
    'botsort_config': BOTSORT_CONFIG
}
mlflow_runner.run_experiment(hyperparams, INPUT_DIR, OUTPUT_DIR)
```

---

## 10. Testing Strategy

### Unit Tests
- `test_metrics_calculator.py`: Test IoU, bbox normalization, etc.
- `test_video_metrics_collector.py`: Test frame data collection
- `test_metrics_aggregator.py`: Test cross-video aggregation
- `test_mlflow_runner.py`: Test MLflow logging (with mock)

### Integration Tests
- Test full pipeline with metrics collection enabled
- Test multiple experiment runs
- Test with different video formats and resolutions

### Validation Tests
- Compare manual calculations with computed metrics
- Verify metric ranges are sensible (e.g., IoU ∈ [0,1])
- Check for edge cases (empty frames, single detection, etc.)

---

## 11. Dependencies

### New Dependencies
Add to `requirements.txt`:
```
mlflow>=2.10.0
pyyaml>=6.0  # Already in use
numpy>=1.24.0  # Already in use (via ultralytics)
```

### Version Compatibility
- MLflow 2.10.0+ for latest features
- Compatible with existing ultralytics/YOLO versions
- No breaking changes to existing dependencies

---

## 12. Performance Considerations

### Metrics Collection Overhead
- **Estimated**: 3-5% processing time increase
- **Mitigation**: Only compute metrics when explicitly enabled
- **Optimization**: 
  - Use vectorized numpy operations where possible
  - Measure processing time once per video (not per frame) to minimize timing overhead
  - Process detections and tracks in batches where feasible

### Memory Overhead
- **Per-frame**: Minimal (~100 bytes per detection)
- **Per-video**: ~1-5 MB for typical videos
- **Total**: Scales linearly with number of videos

### MLflow Logging Overhead
- **Impact**: Negligible (async writes)
- **Storage**: ~10 KB per experiment run (metrics only)
- **Artifacts**: Depends on artifact size (configs are small)

---

## 13. Future Enhancements (Out of Scope)

### Possible Extensions
1. **Real-time metrics visualization**: Stream metrics to dashboard during processing
2. **Automated hyperparameter optimization**: Use Optuna or similar
3. **Model comparison**: Compare different YOLO models (v8n vs v8s vs v8m)
4. **Distributed processing**: Parallelize video processing
5. **Advanced metrics**: Add optical flow analysis, trajectory smoothness
6. **Ground truth comparison**: Add support for labeled data evaluation

---

## 14. Risk Mitigation

### Risk: Breaking Existing Functionality
**Mitigation**: 
- All new features are opt-in
- Extensive backward compatibility testing
- Default behavior unchanged

### Risk: Performance Degradation
**Mitigation**:
- Metrics collection is optional
- Benchmark before/after with large videos
- Profile and optimize bottlenecks

### Risk: MLflow Configuration Issues
**Mitigation**:
- Provide clear setup documentation
- Use sensible defaults
- Add configuration validation

### Risk: Metric Calculation Errors
**Mitigation**:
- Comprehensive unit tests
- Validation against known values
- Edge case handling (empty frames, etc.)

---

## 15. Success Criteria

✅ **Functional Requirements**
- All existing batch processing functionality preserved
- Metrics computed accurately for all specified metrics
- MLflow successfully logs experiments with hyperparameters and metrics
- Multiple experiment runs can be compared in MLflow UI

✅ **Non-Functional Requirements**
- Performance overhead < 10%
- No breaking changes to existing code
- Clear documentation and examples provided
- All tests passing

✅ **User Experience**
- Simple opt-in for metrics collection
- Clear metric definitions and interpretations
- Easy hyperparameter configuration
- Intuitive MLflow experiment organization

---

## 16. Conclusion

This refactor plan provides a comprehensive roadmap for integrating MLflow experiment tracking with minimal disruption to the existing codebase. By implementing metrics collection as a separate, optional layer, we maintain backward compatibility while enabling powerful hyperparameter tuning capabilities.

The modular design ensures that each component can be developed, tested, and validated independently, reducing risk and enabling incremental rollout.

**Recommended Next Steps**:
1. Review and approve this plan
2. Set up development environment with MLflow
3. Begin Phase 1 implementation (metrics collection foundation)
4. Iterate with testing and validation at each phase
