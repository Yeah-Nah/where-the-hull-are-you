import depthai as dai
from src.config.camera_settings import CAM_WIDTH, CAM_HEIGHT


class cameraFeedProcessor:
    """Processor for handling camera feed input and processing."""

    def __init__(self):
        """Initialize the camera feed processor."""
        self.pipeline = dai.Pipeline()

    def camera_feed_connect(self):
        """Connect to the camera feed.

        Returns:
            tuple: (pipeline, cam_out, video_queue)
        """
        # Define sources and outputs
        camRgb = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B
        )

        # Setup neural network input/output
        cam_out = camRgb.requestOutput(
            (CAM_WIDTH, CAM_HEIGHT), dai.ImgFrame.Type.BGR888p, fps=30
        )
        video_out = camRgb.requestOutput(
            (CAM_WIDTH, CAM_HEIGHT), dai.ImgFrame.Type.BGR888p, fps=30
        )
        video_queue = video_out.createOutputQueue()

        return self.pipeline, cam_out, video_queue
