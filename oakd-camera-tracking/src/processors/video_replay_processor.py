"""Processor for handling video replay input and processing."""

from pathlib import Path
from typing import Any

import cv2
import depthai as dai


class VideoFeedProcessor:
    """Processor for handling video replay input and processing."""

    def __init__(self, video_path: Path):
        """Initialize the video feed processor."""
        self.video_path = video_path
        self.pipeline = dai.Pipeline()

    def get_video_resolution(self) -> tuple[int | None, int | None]:
        """Get video resolution (width, height)."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return None, None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return width, height

    def video_feed_connect(self) -> tuple[Any, Any, Any]:
        """Connect to the video feed.

        Returns
        -------
            tuple: (pipeline, replay_node, video_queue)
        """
        # Define sources and outputs
        replay = self.pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(self.video_path))
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        replay.setLoop(True)

        # Create output queue directly from replay.out
        video_queue = replay.out.createOutputQueue()

        return self.pipeline, replay, video_queue
