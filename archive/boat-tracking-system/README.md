# Boat Tracking System

## Overview
The Boat Tracking System is a Python application designed to detect and track boats in video footage using the YOLO object detection model. This project provides a structured approach to processing video files, applying detection and tracking algorithms, and saving the results.

## Project Structure
```
boat-tracking-system/
├── src/                     # Source code for the application
│   ├── __init__.py
│   ├── main.py              # Main entry point of the application
│   ├── config/              # Configuration settings
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── models/              # Models for detection and tracking
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── tracker.py
│   ├── processors/          # Video processing logic
│   │   ├── __init__.py
│   │   ├── video_processor.py
│   │   └── batch_processor.py
│   └── utils/              # Utility functions and classes
│       ├── __init__.py
│       ├── file_handler.py
│       └── logger.py
├── config/                  # Configuration files
│   ├── config.yaml
│   └── config.example.yaml
├── data/                    # Data directories for input and output
│   ├── input
│   └── output
├── models/                  # Pre-trained models
│   └── yolov8n.pt
├── tests/                   # Unit tests for the application
│   ├── __init__.py
│   ├── test_detector.py
│   └── test_processor.py
├── notebooks/               # Jupyter notebooks for experimentation
│   └── boat_tracking_pipeline_experimentation.ipynb
├── requirements.txt         # Project dependencies
├── setup.py                 # Packaging and installation
└── README.md                # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd boat-tracking-system
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration
Edit the `config/config.yaml` file to set the input and output folder paths for your video files. An example configuration is provided in `config/config.example.yaml`.

## Usage
To run the application, execute the following command:
```
python src/main.py
```
This will process all video files in the specified input directory and save the annotated output videos to the output directory.

## Testing
To run the unit tests, use:
```
pytest tests/
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.