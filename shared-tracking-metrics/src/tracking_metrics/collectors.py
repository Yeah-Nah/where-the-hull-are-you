"""Collect detections and tracks over time for metrics calculation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Detection:
    """Single object detection."""

    frame_id: int
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int


@dataclass
class Track:
    """Track information across frames."""

    track_id: int
    detections: List[Detection] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0


class TrackingMetricsCollector:
    """Collect detections and tracks over time for metrics calculation."""

    def __init__(self):
        """Initialize the metrics collector."""
        self.detections: List[Detection] = []
        self.tracks: Dict[int, Track] = {}
        self.frame_count: int = 0

    def add_frame_detections(self, frame_id: int, detections: List[Dict[str, Any]]):
        """Add detections from a single frame.

        Parameters
        ----------
        frame_id : int
            Frame number
        detections : List[Dict[str, Any]]
            List of detections with keys: track_id, bbox, confidence, class_id
        """
        pass

    def add_track(self, track_id: int, track_data: Dict[str, Any]):
        """Add or update a track.

        Parameters
        ----------
        track_id : int
            Unique track identifier
        track_data : Dict[str, Any]
            Track information
        """
        pass

    def get_active_tracks(self, frame_id: int) -> List[Track]:
        """Get tracks active at a specific frame.

        Parameters
        ----------
        frame_id : int
            Frame number

        Returns
        -------
        List[Track]
            Active tracks at the frame
        """
        pass

    def reset(self):
        """Reset all collected data."""
        self.detections.clear()
        self.tracks.clear()
        self.frame_count = 0
