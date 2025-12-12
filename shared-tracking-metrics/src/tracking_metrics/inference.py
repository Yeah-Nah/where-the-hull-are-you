# shared-tracking-metrics/src/tracking_metrics/inference.py
from typing import Any

import numpy as np
from ultralytics import YOLO

"""Generic YOLO inference - works on any frame."""


class ModelInference:
    """Generic YOLO inference - works on any frame."""

    def __init__(
        self,
        model_path: str,
        tracker_config: str | None = None,
        model_config: dict = None,
    ):
        """Initialize inference with model and default parameters.

        Parameters
        ----------
        model_path : str
            Path to YOLO model weights
        tracker_config : str, optional
            Tracker configuration file (e.g., 'botsort.yaml')
        model_config : Dict, optional
            Model configuration parameters
        """
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.model_config = model_config
        self.model_kwargs = self.create_kwargs()

    def create_kwargs(self) -> dict[str, Any]:
        """Create model keyword arguments from model_config."""
        # Remove keys with None values
        return {k: v for k, v in self.model_config.items() if v is not None}

    def predict_frame(
        self,
        frame: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Run YOLO on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Frame from ANY source (camera, video file, etc.)

        Returns
        -------
        List[Dict[str, Any]]
            Detections in standardized format
        """
        # Call model.track with filtered parameters
        results = self.model.track(frame, **self.model_kwargs)

        # Parse detections
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append(
                    {
                        "track_id": int(box.id[0]) if box.id is not None else -1,
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                    }
                )

        return detections
