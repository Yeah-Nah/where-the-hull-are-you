# where-the-hull-are-you
Basic boat detection and tracking using open-source models.

## Overview
This project provides simple Python scripts to detect and track boats in images and videos using YOLOv8, a state-of-the-art object detection model.

Perfect for learning computer vision and object tracking!

## Features
- **Boat Detection**: Identify boats in still images
- **Boat Tracking**: Track boats across video frames with unique IDs
- **Easy to Use**: Simple API with clear examples
- **Pre-trained Models**: Uses YOLOv8 trained on COCO dataset

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Yeah-Nah/where-the-hull-are-you.git
cd where-the-hull-are-you
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Note: On first run, YOLOv8 will automatically download the pre-trained model (~6MB).

## Quick Start

### Using the Demo Script

The easiest way to get started is using the demo script:

```bash
# For images
python demo.py path/to/your/image.jpg

# For videos
python demo.py path/to/your/video.mp4
```

Results will be saved in the `output/` directory.

### Detecting Boats in Images

```python
from detect_boat import detect_boats

# Detect boats and save annotated image
detections = detect_boats('boat_image.jpg', 'output.jpg', confidence_threshold=0.5)

# Print results
for detection in detections:
    print(f"Boat found at {detection['bbox']} with confidence {detection['confidence']:.2f}")
```

### Tracking Boats in Videos

```python
from track_boat import track_boats_in_video

# Track boats and save output video
stats = track_boats_in_video('boat_video.mp4', 'tracked_output.mp4', confidence_threshold=0.5)

# Print statistics
print(f"Processed {stats['total_frames']} frames")
print(f"Found {stats['unique_boat_ids']} unique boats")
```

## Project Structure

```
where-the-hull-are-you/
├── detect_boat.py      # Boat detection in images
├── track_boat.py       # Boat tracking in videos
├── demo.py            # Demo script with examples
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## How It Works

1. **YOLOv8 Model**: Uses the YOLOv8n (nano) model - the smallest and fastest variant
2. **COCO Dataset**: Model is pre-trained on COCO dataset which includes 'boat' as class 8
3. **Tracking**: Uses built-in tracking algorithm to maintain boat IDs across frames
4. **Confidence Threshold**: Filters detections based on confidence score (default: 0.5)

## Parameters

### detect_boats()
- `image_path`: Path to input image
- `output_path`: Path to save annotated image (optional)
- `confidence_threshold`: Minimum confidence score (0-1, default: 0.5)

### track_boats_in_video()
- `video_path`: Path to input video
- `output_path`: Path to save tracked video (optional)
- `confidence_threshold`: Minimum confidence score (0-1, default: 0.5)
- `show_video`: Display video during processing (default: False)

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy

See `requirements.txt` for specific versions.

## Tips for Beginners

1. **Start Simple**: Run the demo script first to understand the basics
2. **Adjust Confidence**: Lower the threshold (e.g., 0.3) to detect more boats, raise it (e.g., 0.7) for more confident detections
3. **Model Size**: YOLOv8n is fast but less accurate. Try `yolov8s.pt`, `yolov8m.pt`, or `yolov8l.pt` for better accuracy
4. **Video Quality**: Better quality videos produce better tracking results
5. **Learn More**: Check out the [Ultralytics documentation](https://docs.ultralytics.com/) for advanced features

## Next Steps

Once you're comfortable with the basics, you can:
- Fine-tune the model on your own boat dataset
- Experiment with different YOLO model sizes
- Add custom post-processing logic
- Integrate with other systems (e.g., save to database, send alerts)
- Try other object classes (ships, yachts, kayaks)

## License

See LICENSE file for details.

## Contributing

Contributions welcome! This is a learning project - feel free to submit improvements.
