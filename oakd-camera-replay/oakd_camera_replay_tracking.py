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
        print(f"Received NN data: {tensor.shape}")
        print("First 10 rows of the tensor:")
        print(tensor[0, :10, :])
