"""Evaluator for unlabeled data using confidence-based metrics."""

from pathlib import Path
from typing import Dict, List

from tracking_metrics import MetricsCalculator, TrackingMetricsCollector


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
        self.model = None
        self.collector = TrackingMetricsCollector()
        self.calculator = MetricsCalculator(self.collector)

    def load_model(self):
        """Load the model."""
        pass

    def evaluate(
        self, video_paths: List[Path], tracker_config: str = "botsort.yaml"
    ) -> Dict[str, float]:
        """Evaluate model on unlabeled videos.

        Parameters
        ----------
        video_paths : List[Path]
            List of video file paths
        tracker_config : str
            Tracker configuration file

        Returns
        -------
        Dict[str, float]
            Confidence-based metrics (no ground truth required)
        """
        pass

    def evaluate_single_video(
        self, video_path: Path, tracker_config: str = "botsort.yaml"
    ) -> Dict[str, float]:
        """Evaluate model on single video.

        Parameters
        ----------
        video_path : Path
            Path to video file
        tracker_config : str
            Tracker configuration file

        Returns
        -------
        Dict[str, float]
            Metrics for single video
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
