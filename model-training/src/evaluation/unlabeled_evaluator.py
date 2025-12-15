"""Evaluator for unlabeled data using confidence-based metrics."""

from pathlib import Path

import cv2
from loguru import logger
from tracking_metrics import MetricsCalculator, TrackingMetricsCollector
from tracking_metrics.inference import ModelInference


class UnlabeledEvaluator:
    """Evaluate model performance on unlabeled data."""

    def __init__(self, model_path: Path, model_config: dict):
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
            model_path=str(model_path), model_config=model_config
        )
        self.collector = TrackingMetricsCollector()
        self.calculator = MetricsCalculator(self.collector)

    def _initiate_cap(self, video_path: str):
        """Inititate the video capture cv2 object."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        return cap

    def _get_video_properties(self, cap: cv2.VideoCapture) -> dict[str, int]:
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
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def _read_video(self, cap):
        """Read video frames helper.

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
        batch_size: int = 16,  # Adjust based on GPU memory
    ) -> dict[str, float]:
        """Evaluate model on single video.
        
        Parameters
        ----------
        video_path : Path
            Path to video file
        batch_size : int
            Number of frames to process at once
            
        Returns
        -------
        Dict[str, float]
            Computed metrics
        """
        logger.info(f"Evaluating video: {video_path}")
        self.collector.reset()

        cap = self._initiate_cap(video_path)
        
        frame_batch = []
        frame_ids = []
        
        for frame_id, frame in self._read_video(cap):
            frame_batch.append(frame)
            frame_ids.append(frame_id)
            
            # Process batch when full
            if len(frame_batch) == batch_size:
                batch_detections = self.inference.predict_batch_frames(frame_batch)
                
                for fid, detections in zip(frame_ids, batch_detections):
                    for det in detections:
                        self.collector.add_detection_with_track(det, fid)
                    self.collector.frame_count += 1
                
                if frame_id % 100 == 0:
                    logger.debug(f"Processed frame {frame_id}")
                
                frame_batch = []
                frame_ids = []
        
        # Process remaining frames
        if frame_batch:
            batch_detections = self.inference.predict_batch_frames(frame_batch)
            for fid, detections in zip(frame_ids, batch_detections):
                for det in detections:
                    self.collector.add_detection_with_track(det, fid)
                self.collector.frame_count += 1

        logger.info("Computing metrics...")
        metrics = self.calculator.compute_all_metrics()
        logger.success("Results:")
        for metric_name, value in metrics.items():
            logger.success(f"  {metric_name}: {value:.4f}")

        return metrics

    def evaluate_batch_video(
        self, folder_path: Path, tracker_config: str = "botsort.yaml"
    ) -> dict[str, float]:
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
