"""Visualize tracking metrics on video frames."""

import numpy as np

from .collectors import TrackingMetricsCollector


class MetricsVisualizer:
    """Visualize tracking metrics on video frames."""

    def __init__(self, collector: TrackingMetricsCollector):
        """Initialize the metrics visualizer.

        Parameters
        ----------
        collector : TrackingMetricsCollector
            Collector with tracking data
        """
        self.collector = collector

    def draw_detections(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """Draw detections on a frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame
        frame_id : int
            Frame number

        Returns
        -------
        np.ndarray
            Frame with detections drawn
        """
        raise NotImplementedError()

    def draw_tracks(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """Draw tracks on a frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame
        frame_id : int
            Frame number

        Returns
        -------
        np.ndarray
            Frame with tracks drawn
        """
        raise NotImplementedError()

    def draw_metrics_overlay(
        self, frame: np.ndarray, metrics: dict[str, float]
    ) -> np.ndarray:
        """Draw metrics as text overlay on frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame
        metrics : Dict[str, float]
            Metrics to display

        Returns
        -------
        np.ndarray
            Frame with metrics overlay
        """
        raise NotImplementedError()

    def create_confidence_heatmap(
        self, frame_shape: tuple[int, int], window_size: int = 100
    ) -> np.ndarray:
        """Create confidence heatmap over time.

        Parameters
        ----------
        frame_shape : tuple
            Shape of output frame (height, width)
        window_size : int
            Number of frames to include in heatmap

        Returns
        -------
        np.ndarray
            Heatmap visualization
        """
        raise NotImplementedError()

    def plot_track_length_distribution(self) -> np.ndarray:
        """Plot histogram of track lengths.

        Returns
        -------
        np.ndarray
            Track length distribution plot
        """
        raise NotImplementedError()
