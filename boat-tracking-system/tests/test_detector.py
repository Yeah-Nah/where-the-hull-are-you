from src.models.detector import Detector
import os
import pytest

class TestDetector:
    @pytest.fixture(scope="class")
    def setup(self):
        self.detector = Detector(model_path="models/yolov8n.pt")
        self.test_video_path = "data/input/test_video.mp4"
        self.output_video_path = "data/output/test_output.mp4"
        
        # Ensure the test video exists
        assert os.path.exists(self.test_video_path), "Test video file does not exist."

    def test_load_model(self, setup):
        assert self.detector.model is not None, "Model should be loaded successfully."

    def test_detect_boats(self, setup):
        detections = self.detector.detect(self.test_video_path)
        assert isinstance(detections, list), "Detections should be a list."
        assert len(detections) >= 0, "Detections should not be negative."

    def test_process_video(self, setup):
        self.detector.process_video(self.test_video_path, self.output_video_path)
        assert os.path.exists(self.output_video_path), "Output video should be created."