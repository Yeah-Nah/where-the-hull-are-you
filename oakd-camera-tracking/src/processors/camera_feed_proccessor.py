"""Processor for handling camera feed input and processing."""

from typing import Any

import depthai as dai
from src.config.camera_settings import CAM_HEIGHT, CAM_WIDTH


class CameraFeedProcessor:
    """Processor for handling camera feed input and processing."""

    def __init__(self) -> None:
        """Initialize the camera feed processor."""
        self.pipeline = dai.Pipeline()

    def camera_feed_connect(self) -> tuple[Any, Any, Any]:
        """Connect to the camera feed.

        Returns
        -------
            tuple: (pipeline, cam_out, video_queue)
        """
        # Define sources and outputs
        cam_rgb = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B
        )

        # Setup neural network input/output
        cam_out = cam_rgb.requestOutput(
            (CAM_WIDTH, CAM_HEIGHT), dai.ImgFrame.Type.BGR888p, fps=30
        )
        video_out = cam_rgb.requestOutput(
            (CAM_WIDTH, CAM_HEIGHT), dai.ImgFrame.Type.BGR888p, fps=30
        )
        video_queue = video_out.createOutputQueue()

        return self.pipeline, cam_out, video_queue
