"""Evaluator for unlabeled data using confidence-based metrics."""

from pathlib import Path
from typing import Dict, List
import cv2
import time
from loguru import logger

from tracking_metrics import MetricsCalculator, TrackingMetricsCollector
from tracking_metrics.inference import ModelInference


class UnlabeledEvaluator:
    """Evaluate model performance on unlabeled data."""

    def __init__(self, model_path: Path):
        """Initialize evaluator with model.

        Parameters
        ----------
        model_path : Path
            Path to model weights
        """
        self.model_path = model_path
        self.inference = ModelInference(str(model_path))
        self.collector = TrackingMetricsCollector()
        self.calculator = MetricsCalculator(self.collector)

    def _read_video(self, video_path: Path):
        """Helper to read video frames.
        
        Yields
        ------
        Tuple[int, np.ndarray]
            (frame_id, frame)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame_id, frame
                frame_id += 1
        finally:
            cap.release()

    def evaluate_single_video(
            self, video_path: Path, tracker_config: str = "botsort.yaml"
        ) -> Dict[str, float]:
            """Evaluate model on single video.
            
            Parameters
            ----------
            video_path : Path
                Path to video file
            tracker_config : str
                Tracker configuration (not used in ModelInference currently)
                
            Returns
            -------
            Dict[str, float]
                Computed metrics
            """
            logger.info(f"Evaluating video: {video_path}")
            self.collector.reset()
            
            # Read video and run inference on each frame
            for frame_id, frame in self._read_video(video_path):
                # Use the inference instance to predict
                detections = self.inference.predict_frame(frame)
                
                # Collect detections for metrics
                self.collector.add_frame_detections(detections)
                
                if frame_id % 100 == 0:
                    logger.debug(f"Processed frame {frame_id}")
            
            # Compute and return metrics
            metrics = self.compute_metrics()
            logger.info(f"Evaluation complete: {metrics}")
            return metrics

    def evaluate_batch(
        self, folder_path: Path, tracker_config: str = "botsort.yaml"
    ) -> Dict[str, float]:
        """Evaluate model on unlabeled videos.

        Parameters
        ----------
        folder_path : Path
            Path to folder containing video files
        tracker_config : str
            Tracker configuration file

        Returns
        -------
        Dict[str, float]
            Confidence-based metrics (no ground truth required)
        """
        pass

    def compute_metrics(self) -> Dict[str, float]:
        """Compute confidence-based metrics from collected data.

        Returns
        -------
        Dict[str, float]
            Computed metrics
        """
        pass
