"""Collect detections and tracks over time for metrics calculation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


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
        self.confidences: List[float] = []
        self.bbox_areas: List[float] = []

    def add_frame_detections(
        self, detections: List[Dict], frame_shape: Tuple[int, int] = None
    ):
        """
        Add detection data from a single frame.

        Parameters
        ----------
        detections : List[Dict]
            List of detection dictionaries with keys:
            - 'track_id': int
            - 'bbox': [x1, y1, x2, y2]
            - 'confidence': float
            - 'class_id': int (optional)
        frame_shape : Tuple[int, int], optional
            Frame (height, width) for bbox normalization
        """
        self.frame_count += 1

        for det in detections:
            # Store confidence
            self.confidences.append(det["confidence"])

            # Normalize and store bbox area if frame shape provided
            if frame_shape is not None:
                # height, width = frame_shape
                # normalized_area = normalize_bbox_area(det["bbox"], width, height)
                self.bbox_areas.append(frame_shape)
            else:
                self.bbox_areas.append(frame_shape)

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
