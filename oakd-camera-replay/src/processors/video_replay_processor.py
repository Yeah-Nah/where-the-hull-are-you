import cv2
import depthai as dai
from pathlib import Path

class videoFeedProcessor:
    """Processor for handling video replay input and processing."""
    def __init__(self, video_path: Path):
        """Initialize the video feed processor."""
        self.video_path = video_path
        self.pipeline = dai.Pipeline()

    def get_video_resolution(self):
        """Get video resolution (width, height)."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return None, None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return width, height

    def video_feed_connect(self):
        """Connect to the video feed.
        
        Returns:
            tuple: (pipeline, cam_out, video_queue)
        """
        # Get video resolution
        width, height = self.get_video_resolution()

        # Define sources and outputs
        replay = self.pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(self.video_path))

        # Setup neural network input/output
        cam_out = replay.requestOutput((width, height), dai.ImgFrame.Type.BGR888p, fps=30)
        video_out = replay.requestOutput((width, height), dai.ImgFrame.Type.BGR888p, fps=30)
        video_queue = video_out.createOutputQueue()

        return self.pipeline, cam_out, video_queue