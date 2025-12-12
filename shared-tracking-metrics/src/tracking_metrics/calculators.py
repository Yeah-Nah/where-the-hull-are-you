"""Calculate tracking metrics from collected data."""

from typing import Dict, List, Optional

from .collectors import Track, TrackingMetricsCollector


class MetricsCalculator:
    """Calculate tracking metrics from collected data."""

    def __init__(self, collector: TrackingMetricsCollector):
        """Initialize the metrics calculator.

        Parameters
        ----------
        collector : TrackingMetricsCollector
            Collector with tracking data
        """
        self.collector = collector

    def compute_confidence_metrics(self) -> Dict[str, float]:
        """Compute detection confidence statistics.

        Returns
        -------
        Dict[str, float]
            Confidence metrics (mean, std, min, max)
        """
        pass

    def compute_track_metrics(self) -> Dict[str, float]:
        """Compute track-based metrics.

        Returns
        -------
        Dict[str, float]
            Track metrics (avg_length, num_tracks, etc.)
        """
        pass

    def compute_bbox_area(self) -> Dict[str, float]:
        """Calculate the bounding box area for the given coordinates.

        Returns:
            Dict[str, float]: _description_
        """

    def compute_bbox_stability(self) -> Dict[str, float]:
        """Compute bounding box stability metrics.

        Returns
        -------
        Dict[str, float]
            Stability metrics (jitter, consistency)
        """
        pass

    def compute_mota(self, ground_truth: Optional[List[Track]] = None) -> float:
        """Compute Multiple Object Tracking Accuracy (MOTA).

        Parameters
        ----------
        ground_truth : Optional[List[Track]]
            Ground truth tracks (None for unlabeled data)

        Returns
        -------
        float
            MOTA score (only if ground truth provided)
        """
        pass

    def compute_short_track_ratio(self, threshold: int = 5) -> float:
        """_summary_

        Parameters
        ----------
            threshold : int, optional
                _description_, by default 5

        Returns
        -------
            float
                _description_
        """
        pass

    def compute_idf1(self, ground_truth: Optional[List[Track]] = None) -> float:
        """Compute ID F1 score.

        Parameters
        ----------
        ground_truth : Optional[List[Track]]
            Ground truth tracks (None for unlabeled data)

        Returns
        -------
        float
            IDF1 score (only if ground truth provided)
        """
        pass

    def compute_all_metrics(
        self, ground_truth: Optional[List[Track]] = None
    ) -> Dict[str, float]:
        """Compute all available metrics.

        Parameters
        ----------
        ground_truth : Optional[List[Track]]
            Ground truth tracks (None for unlabeled data)

        Returns
        -------
        Dict[str, float]
            All computed metrics
        """
        metrics = {}
        metrics.update(self.compute_confidence_metrics())
        metrics.update(self.compute_track_metrics())
        metrics.update(self.compute_bbox_stability())

        if ground_truth is not None:
            metrics["mota"] = self.compute_mota(ground_truth)
            metrics["idf1"] = self.compute_idf1(ground_truth)

        return metrics
