"""Evaluator for labeled data with ground truth."""

from pathlib import Path

from tracking_metrics import MetricsCalculator, TrackingMetricsCollector
from tracking_metrics.collectors import Track


class LabeledEvaluator:
    """Evaluate model performance on labeled data with ground truth."""

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

    def load_ground_truth(self, ground_truth_path: Path) -> list[Track]:
        """Load ground truth annotations.

        Parameters
        ----------
        ground_truth_path : Path
            Path to ground truth annotations

        Returns
        -------
        List[Track]
            Ground truth tracks
        """
        pass

    def evaluate(self, data_yaml: Path) -> dict[str, float]:
        """Evaluate model on labeled dataset.

        Parameters
        ----------
        data_yaml : Path
            Path to dataset YAML configuration

        Returns
        -------
        Dict[str, float]
            Evaluation metrics including mAP, MOTA, IDF1
        """
        pass

    def evaluate_with_ground_truth(
        self, video_path: Path, ground_truth: list[Track]
    ) -> dict[str, float]:
        """Evaluate model on single video with ground truth.

        Parameters
        ----------
        video_path : Path
            Path to video file
        ground_truth : List[Track]
            Ground truth tracks

        Returns
        -------
        Dict[str, float]
            Tracking metrics with ground truth comparison
        """
        pass
