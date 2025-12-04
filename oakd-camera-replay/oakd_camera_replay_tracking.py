# Just test file for tracking an object in a replayed oak-d camera feed

#!/usr/bin/env python3

from socket import socket
import cv2
import depthai as dai
import time
import numpy as np

from pathlib import Path
from argparse import ArgumentParser

from src.config.settings import INPUT_DIR
from src.config.model_settings import CLASS_IDS, CONFIDENCE_THRESHOLD, MODEL_BLOB, COCO_CLASSES

scriptDir = Path(__file__).resolve().parent
examplesRoot = (
    scriptDir / Path("../oakd-camera-replay")
).resolve()  # This resolves the parent directory correctly
models = examplesRoot / "src" / "models"

parser = ArgumentParser()
parser.add_argument("-i", "--inputVideo", default=INPUT_DIR, help="Input video name")
parser.add_argument(
    "-c", "--camera", type=bool, help="Use camera as input", default=False
)
args = parser.parse_args()

# Create pipeline
with dai.Pipeline() as pipeline:
    # Define sources and outputs
    inputSource = None
    if args.camera:
        camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        inputSource = camRgb
        
        # Get the video size from camera configuration
        video_size = camRgb.getVideoSize()
        output_width, output_height = video_size
    else:
        # For replay, you can get it from the first frame
        output_width, output_height = 512, 384  # Default
    # If your nn model requires 512x384 input size (BGR):
    cam_out = inputSource.requestOutput((output_width, output_height), dai.ImgFrame.Type.BGR888p)
    
    # Get video output for display (full resolution)
    video_out = inputSource.requestOutput((output_width, output_height), dai.ImgFrame.Type.BGR888p)
    video_queue = video_out.createOutputQueue()

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(models / MODEL_BLOB))
    
    # Link input to neural network
    cam_out.link(nn.input)

    qNNData = nn.out.createOutputQueue()
    
    pipeline.start()
    
    while pipeline.isRunning():
        # Get video frame
        videoFrame = video_queue.tryGet()
        inNNData = qNNData.tryGet()
        
        if videoFrame is not None and inNNData is not None:
            assert isinstance(videoFrame, dai.ImgFrame)
            frame = videoFrame.getCvFrame()
            frame_height, frame_width = frame.shape[:2]
            
            tensor = inNNData.getFirstTensor()
            assert(isinstance(tensor, np.ndarray))
            
            # Reshape tensor to (1, 84, 4032)
            tensor = tensor.reshape(1, 84, 4032)
            
            # Check each grid cell
            valid_detections = []
            for i in range(4032):  # For each grid cell
                # Get bounding box and class scores for this grid cell
                bbox = tensor[0, :4, i]  # x, y, w, h (normalized 0-1)
                class_scores = tensor[0, 4:, i]  # 80 class scores
                
                # Find max confidence and corresponding class
                max_confidence = np.max(class_scores)
                predicted_class = np.argmax(class_scores)
                
                # Filter by confidence threshold and class IDs
                if max_confidence > CONFIDENCE_THRESHOLD and predicted_class in CLASS_IDS:
                    valid_detections.append({
                        'grid_cell': i,
                        'class': predicted_class,
                        'confidence': max_confidence,
                        'bbox': bbox
                    })
            
            # Draw bounding boxes on frame
            for det in valid_detections:
                # YOLOv8 outputs are in absolute pixel coordinates
                x_center, y_center, width, height = det['bbox']
                
                # Convert center coordinates to top-left corner
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, frame_width))
                y1 = max(0, min(y1, frame_height))
                x2 = max(0, min(x2, frame_width))
                y2 = max(0, min(y2, frame_height))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with class name and confidence
                class_name = COCO_CLASSES.get(det['class'], f"Class {det['class']}")
                label = f"{class_name}: {det['confidence']:.2f}"
                
                # Draw label background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            # Display detection count
            info_text = f"Detections: {len(valid_detections)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Object Detection", frame)
            
            print(f"Valid detections: {len(valid_detections)}")
        
        if cv2.waitKey(1) == ord("q"):
            break
    
    cv2.destroyAllWindows()
