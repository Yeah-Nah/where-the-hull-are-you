"""Calculate tracking metrics from collected data."""

import numpy as np

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

    def compute_confidence_metrics(self) -> dict[str, float]:
        """Compute detection confidence statistics.

        Returns
        -------
        Dict[str, float]
            Confidence metrics (mean, std, min, max)
        """
        if not self.collector.confidences:
            return {
                "confidence_mean": 0.0,
                "confidence_std": 0.0,
            }

        confidences = np.array(self.collector.confidences)
        return {
            "confidence_mean": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
        }

    def compute_track_metrics(self) -> dict[str, float]:
        """Compute track-based metrics.

        Returns
        -------
        Dict[str, float]
            Track metrics (avg_length, num_tracks, etc.)
        """
        if not self.collector.track_history:
            return {
                "num_tracks": 0,
                "avg_track_length": 0.0,
                "max_track_length": 0,
                "min_track_length": 0,
            }

        track_lengths = [
            len(frames) for frames in self.collector.track_history.values()
        ]
        return {
            "num_tracks": len(self.collector.track_history),
            "avg_track_length": float(np.mean(track_lengths)),
            "max_track_length": int(np.max(track_lengths)),
            "min_track_length": int(np.min(track_lengths)),
        }

    def compute_bbox_area_metrics(self) -> dict[str, float]:
        """Calculate the bounding box area statistics.

        Returns
        -------
        Dict[str, float]
            Bounding box area metrics (mean, std, min, max)
        """
        if len(self.collector.bbox_areas) == 0:
            return {
                "bbox_area_mean": 0.0,
                "bbox_area_std": 0.0,
                "bbox_area_min": 0.0,
                "bbox_area_max": 0.0,
            }

        # bbox_areas already contains calculated areas (floats)
        areas = np.array(self.collector.bbox_areas)
        
        return {
            "bbox_area_mean": float(np.mean(areas)),
            "bbox_area_std": float(np.std(areas)),
            "bbox_area_min": float(np.min(areas)),
            "bbox_area_max": float(np.max(areas)),
        }

    def compute_bbox_stability(self) -> dict[str, float]:
        """Compute bounding box stability metrics.

        Returns
        -------
        Dict[str, float]
            Stability metrics (jitter, consistency)
        """
        if not self.collector.track_bboxes:
            return {
                "bbox_jitter_mean": 0.0,
                "bbox_jitter_std": 0.0,
            }

        all_jitters = []
        for _, frame_bboxes in self.collector.track_bboxes.items():
            sorted_frames = sorted(frame_bboxes.keys())
            if len(sorted_frames) < 2:
                continue

            # Calculate frame-to-frame displacement
            for i in range(len(sorted_frames) - 1):
                bbox1 = frame_bboxes[sorted_frames[i]]
                bbox2 = frame_bboxes[sorted_frames[i + 1]]

                # Calculate center displacement
                center1_x = (bbox1[0] + bbox1[2]) / 2
                center1_y = (bbox1[1] + bbox1[3]) / 2
                center2_x = (bbox2[0] + bbox2[2]) / 2
                center2_y = (bbox2[1] + bbox2[3]) / 2

                displacement = np.sqrt(
                    (center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2
                )
                all_jitters.append(displacement)

        if not all_jitters:
            return {
                "bbox_jitter_mean": 0.0,
                "bbox_jitter_std": 0.0,
            }

        jitters = np.array(all_jitters)
        return {
            "bbox_jitter_mean": float(np.mean(jitters)),
            "bbox_jitter_std": float(np.std(jitters)),
        }

    def compute_short_track_ratio(self, threshold: int = 5) -> float:
        """Calculate ratio of short tracks to total tracks.

        Parameters
        ----------
        threshold : int, optional
            Minimum frame count for a track to be considered "long", by default 5

        Returns
        -------
        float
            Ratio of tracks shorter than threshold to total tracks
        """
        if not self.collector.track_history:
            return 0.0

        # TODO: Repeated code with track_lengths. Refactor to call each function after creating track
        track_lengths = [
            len(frames) for frames in self.collector.track_history.values()
        ]
        short_tracks = sum(1 for length in track_lengths if length < threshold)

        return short_tracks / len(track_lengths)

    def compute_track_coverage_ratio(self, total_frames: int) -> dict[str, float]:
        """Compute track length as percentage of total video length.

        Parameters
        ----------
        total_frames : int
            Total frames in the current video being analyzed

        Returns
        -------
        Dict[str, float]
            Track coverage metrics for this video
        """
        if not self.collector.track_history or total_frames == 0:
            return {
                "avg_track_coverage": 0.0,
                "max_track_coverage": 0.0,
                "median_track_coverage": 0.0,
            }

        track_lengths = [len(frames) for frames in self.collector.track_history.values()]
        coverage_ratios = [length / total_frames for length in track_lengths]

        return {
            "avg_track_coverage": float(np.mean(coverage_ratios)),
            "max_track_coverage": float(np.max(coverage_ratios)),
            "median_track_coverage": float(np.median(coverage_ratios)),
        }

    def compute_detection_density(self, total_frames: int) -> dict[str, float]:
        """Calculate detections per frame to understand detection frequency.

        Parameters
        ----------
        total_frames : int
            Total frames in the current video being analyzed

        Returns
        -------
        Dict[str, float]
            Detection density metrics
        """
        if total_frames == 0:
            return {
                "total_detections": 0,
                "detections_per_frame": 0.0,
            }

        total_detections = len(self.collector.confidences)
        return {
            "total_detections": total_detections,
            "detections_per_frame": float(total_detections / total_frames),
        }

    def compute_mota(self, ground_truth: list[Track] | None = None) -> float:
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

    def compute_idf1(self, ground_truth: list[Track] | None = None) -> float:
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
        self, ground_truth: list[Track] | None = None, total_frames: int | None = None
    ) -> dict[str, float]:
        """Compute all available metrics.

        Parameters
        ----------
        ground_truth : Optional[List[Track]]
            Ground truth tracks (None for unlabeled data)
        total_frames : Optional[int]
            Total frames in video for normalized metrics (track coverage, detection density)

        Returns
        -------
        Dict[str, float]
            All computed metrics
        """
        metrics = {}
        metrics.update(self.compute_confidence_metrics())
        metrics.update(self.compute_track_metrics())
        metrics.update(self.compute_bbox_area_metrics())
        metrics.update(self.compute_bbox_stability())
        metrics["short_track_ratio"] = self.compute_short_track_ratio()

        # Add normalized metrics if total_frames provided
        if total_frames is not None:
            metrics.update(self.compute_track_coverage_ratio(total_frames))
            metrics.update(self.compute_detection_density(total_frames))

        if ground_truth is not None:
            metrics["mota"] = self.compute_mota(ground_truth)
            metrics["idf1"] = self.compute_idf1(ground_truth)

        return metrics
