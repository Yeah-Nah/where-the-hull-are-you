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

    def __init__(self, model_path: Path, model_config: Dict):
        """Initialize evaluator with model.

        Parameters
        ----------
        model_path : Path
            Path to model weights
        model_config : Dict
            Model configuration parameters
        """
        self.model_path = model_path
        self.model_config = model_config
        self.inference = ModelInference(
            model_path=str(model_path),
            model_config=model_config
        )
        self.collector = TrackingMetricsCollector()
        self.calculator = MetricsCalculator(self.collector)

    def _initiate_cap(self, video_path: str):
        """Inititate the video capture cv2 object."""

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        return cap

    def _get_video_properties(self, cap: cv2.VideoCapture) -> Dict[str, int]:
        """Get video properties efficiently.
        
        Parameters
        ----------
        cap : cv2.VideoCapture
            Opened video capture object
            
        Returns
        -------
        Dict[str, int]
            Video properties (width, height, fps, frame_count)
        """

        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
    
    def _read_video(self, cap):
        """Helper to read video frames.
        
        Yields
        ------
        Tuple[int, np.ndarray]
            (frame_id, frame)
        """
        
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
            self,
            video_path: Path,
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

            cap = self._initiate_cap(video_path)
            video_props = self._get_video_properties(cap)
            
            # Read video and run inference on each frame
            for frame_id, frame in self._read_video(cap):
                # Use the inference instance to predict
                detections = self.inference.predict_frame(frame)
                
                # Process all detections
                for det in detections:
                    self.collector.add_detection_with_track(det, frame_id)
                
                self.collector.frame_count += 1
                
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
