# Boat Tracking Pipeline - Code Plan

## Project Overview
Create a Jupyter notebook pipeline to detect and track boats in video footage using YOLO object detection. The pipeline will read video files, identify boats, draw bounding boxes, and save annotated output video.

## Learning Objectives
- Understand how YOLO object detection works
- Learn video processing with OpenCV
- Practice object tracking in video streams
- Build a foundation for more advanced marine vessel detection

## Pipeline Architecture

### 1. Setup and Dependencies
**Cell 1: Import Libraries**
```python
# Core libraries
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

# For visualization and progress tracking
import matplotlib.pyplot as plt
from IPython.display import Video, display
import time
```

**Cell 2: Configuration**
```python
# File paths
INPUT_VIDEO_PATH = "path/to/your/video.mp4"
OUTPUT_VIDEO_PATH = "output/tracked_boats.mp4" 
MODEL_PATH = "yolov8n.pt"  # Will auto-download

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5
BOAT_CLASSES = ['boat', 'ship']  # YOLO class names for boats
```

### 2. Model Loading and Setup
**Cell 3: Load YOLO Model**
```python
# Load pre-trained YOLO model
model = YOLO(MODEL_PATH)

# Display model information
print("Model classes:", model.names)
print("Boat-related classes:", [k for k, v in model.names.items() if 'boat' in v.lower() or 'ship' in v.lower()])
```

### 3. Video Input Processing
**Cell 4: Video Analysis**
```python
# Open video file and get properties
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video Properties:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Total Frames: {frame_count}")
print(f"  Duration: {frame_count/fps:.2f} seconds")
```

**Cell 5: Sample Frame Analysis**
```python
# Read and display first frame
ret, sample_frame = cap.read()
if ret:
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
    plt.title("Sample Frame from Input Video")
    plt.axis('off')
    plt.show()
    
    # Test detection on sample frame
    results = model(sample_frame, conf=CONFIDENCE_THRESHOLD)
    print(f"Objects detected in sample frame: {len(results[0].boxes) if results[0].boxes is not None else 0}")
```

### 4. Core Detection Function
**Cell 6: Boat Detection Function**
```python
def detect_boats_in_frame(frame, model, confidence=0.5):
    """
    Detect boats in a single frame
    
    Args:
        frame: Input image frame
        model: YOLO model
        confidence: Detection confidence threshold
    
    Returns:
        annotated_frame: Frame with bounding boxes
        detections: List of detection data
    """
    # Run detection
    results = model(frame, conf=confidence)
    
    # Extract detections
    detections = []
    annotated_frame = frame.copy()
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence_score = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            
            # Filter for boat/ship classes
            if any(boat_class in class_name.lower() for boat_class in ['boat', 'ship']):
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence_score),
                    'class': class_name
                })
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name}: {confidence_score:.2f}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return annotated_frame, detections
```

### 5. Video Processing Pipeline
**Cell 7: Main Processing Loop**
```python
def process_video(input_path, output_path, model, confidence=0.5):
    """
    Process entire video for boat detection and tracking
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing statistics
    stats = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'boats_detected': 0,
        'detection_summary': []
    }
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect boats in current frame
        annotated_frame, detections = detect_boats_in_frame(frame, model, confidence)
        
        # Update statistics
        stats['processed_frames'] += 1
        if detections:
            stats['boats_detected'] += len(detections)
            stats['detection_summary'].append({
                'frame': frame_number,
                'boats': len(detections),
                'detections': detections
            })
        
        # Write annotated frame to output
        out.write(annotated_frame)
        
        # Progress update
        if frame_number % 30 == 0:  # Every 30 frames
            progress = (frame_number / total_frames) * 100
            print(f"Progress: {progress:.1f}% (Frame {frame_number}/{total_frames})")
        
        frame_number += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    return stats
```

### 6. Execution and Results
**Cell 8: Run the Pipeline**
```python
# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

# Process the video
print("Starting boat detection and tracking...")
start_time = time.time()

processing_stats = process_video(
    INPUT_VIDEO_PATH, 
    OUTPUT_VIDEO_PATH, 
    model, 
    CONFIDENCE_THRESHOLD
)

end_time = time.time()
processing_time = end_time - start_time

print(f"\nProcessing completed in {processing_time:.2f} seconds")
print(f"Total frames processed: {processing_stats['processed_frames']}")
print(f"Total boat detections: {processing_stats['boats_detected']}")
print(f"Average detections per frame: {processing_stats['boats_detected']/processing_stats['processed_frames']:.2f}")
```

**Cell 9: Display Results**
```python
# Display sample results
if processing_stats['detection_summary']:
    print("Sample detections:")
    for i, detection in enumerate(processing_stats['detection_summary'][:5]):  # First 5 detections
        print(f"Frame {detection['frame']}: {detection['boats']} boat(s) detected")
        for boat in detection['detections']:
            print(f"  - {boat['class']} (confidence: {boat['confidence']:.2f})")

# Display output video (if small enough)
if os.path.getsize(OUTPUT_VIDEO_PATH) < 50*1024*1024:  # Less than 50MB
    display(Video(OUTPUT_VIDEO_PATH, width=800))
else:
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
    print("Video too large to display inline - please open file directly")
```

### 7. Analysis and Visualization
**Cell 10: Detection Analysis**
```python
# Analyze detection patterns
if processing_stats['detection_summary']:
    frame_numbers = [d['frame'] for d in processing_stats['detection_summary']]
    boat_counts = [d['boats'] for d in processing_stats['detection_summary']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, boat_counts, 'b-', alpha=0.7)
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Boats Detected')
    plt.title('Boat Detection Over Time')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Summary statistics
    print(f"Peak boats in single frame: {max(boat_counts)}")
    print(f"Frames with boats detected: {len(frame_numbers)}")
    print(f"Detection rate: {len(frame_numbers)/processing_stats['processed_frames']*100:.1f}%")
```

## Next Steps for Enhancement

### Immediate Improvements
1. **Add tracking IDs** - Use YOLO's built-in tracking to maintain boat identities across frames
2. **Improve filtering** - Add size/shape filters to reduce false positives
3. **Better visualization** - Add boat trails, speed indicators, or count overlays

### Advanced Features
1. **Multi-class detection** - Distinguish between sailboats, motorboats, kayaks
2. **Speed calculation** - Track boat movement and calculate speeds
3. **Area of interest** - Define regions for focused detection
4. **Export data** - Save detection coordinates to CSV for analysis

### Performance Optimizations
1. **Batch processing** - Process multiple frames at once
2. **GPU acceleration** - Utilize CUDA if available
3. **Resolution optimization** - Process at lower resolution for speed

## File Structure
```
project/
├── boat_tracking_pipeline.ipynb    # Main notebook
├── input/                          # Input videos
├── output/                         # Processed videos
├── models/                         # YOLO model files
└── data/                          # Analysis exports
```

## Key Learning Points
- **YOLO Integration**: How to load and use pre-trained models
- **Video Processing**: Frame-by-frame analysis with OpenCV
- **Object Detection**: Understanding confidence thresholds and bounding boxes
- **Data Pipeline**: Input → Processing → Output workflow
- **Performance Monitoring**: Tracking processing statistics

This plan provides a solid foundation for boat tracking that can be extended with more sophisticated features as your understanding grows.