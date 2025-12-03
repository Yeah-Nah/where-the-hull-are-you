# Just test file for tracking an object in a replayed oak-d camera feed

#!/usr/bin/env python3

from socket import socket
import cv2
import depthai as dai
import time
import numpy as np

from pathlib import Path
from argparse import ArgumentParser

scriptDir = Path(__file__).resolve().parent
examplesRoot = (
    scriptDir / Path("../oakd-camera-replay")
).resolve()  # This resolves the parent directory correctly
models = examplesRoot / "src" / "models"
videoPath = models / "construction_vest.mp4"



parser = ArgumentParser()
parser.add_argument("-i", "--inputVideo", default=videoPath, help="Input video name")
parser.add_argument(
    "-c", "--camera", type=bool, help="Use camera as input", default=False
)
args = parser.parse_args()

# Create pipeline
with dai.Pipeline() as pipeline:
    # Define sources and outputs
    inputSource = None
    if args.camera:
        # CAM_A i think is the rgb camera
        camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        inputSource = camRgb
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.inputVideo))
        inputSource = replay
    # If your nn model requires 512x384 input size (BGR):
    cam_out = inputSource.requestOutput((512, 384), dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(models / "yolov8n_openvino_2022.1_6shave.blob"))
    
    # Link input to neural network
    cam_out.link(nn.input)

    qNNData = nn.out.createOutputQueue()
    
    pipeline.start()
    while pipeline.isRunning():
        inNNData: dai.NNData = qNNData.get()
        tensor = inNNData.getFirstTensor()
        assert(isinstance(tensor, np.ndarray))
        
        # Reshape tensor to (1, 84, 4032)
        tensor = tensor.reshape(1, 84, 4032)
        
        print(f"Received NN data: {tensor.shape}")
        
        # Check each grid cell
        detections_found = 0
        for i in range(4032):  # For each grid cell
            # Get bounding box and class scores for this grid cell
            bbox = tensor[0, :4, i]  # x, y, w, h
            class_scores = tensor[0, 4:, i]  # 80 class scores
            
            # Find max confidence and corresponding class
            max_confidence = np.max(class_scores)
            predicted_class = np.argmax(class_scores)
            
            # Print if any class has probability > 0
            if max_confidence > 0:
                detections_found += 1
                print(f"Grid cell {i}: Class {predicted_class}, Confidence: {max_confidence:.4f}, BBox: {bbox}")
        
        print(f"\nTotal grid cells with confidence > 0: {detections_found}")
        print("-" * 80)
        break
