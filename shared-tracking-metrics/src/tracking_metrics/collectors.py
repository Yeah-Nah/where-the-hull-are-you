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
        
        # Track data
        self.track_history: Dict[int, List[int]] = {}  # track_id -> [frame_numbers]
        self.track_bboxes: Dict[int, Dict[int, List[float]]] = {}  # track_id -> {frame_num -> bbox}
        self.all_track_ids: List[int] = []

    def add_detection_with_track(
        self, 
        detection: Dict, 
        frame_id: int, 
        frame_shape: Tuple[int, int] = None
    ):
        """Add detection and update track for a single frame.
        
        Parameters
        ----------
        detection : Dict
            Detection dictionary with keys:
            - 'bbox': [x1, y1, x2, y2]
            - 'confidence': float
            - 'track_id': int (optional)
            - 'class_id': int (optional)
        frame_num : int
            Current frame number
        frame_shape : Tuple[int, int], optional
            Frame (height, width) for bbox normalization
        """
        # Collect detection data
        self.confidences.append(detection["confidence"])
        self.bbox_areas.append(detection["bbox"])
        
        # Collect track data if track_id exists
        track_id = detection.get('track_id')
        if track_id is not None:
            if track_id not in self.track_history:
                self.track_history[track_id] = []
                self.track_bboxes[track_id] = {}
            
            self.track_history[track_id].append(frame_id)
            self.track_bboxes[track_id][frame_id] = detection["bbox"]
            self.all_track_ids.append(track_id)

    def reset(self):
        """Reset all collected data."""
        self.detections.clear()
        self.tracks.clear()
        self.frame_count = 0
        self.track_history.clear()
        self.track_bboxes.clear()
        self.all_track_ids.clear()
