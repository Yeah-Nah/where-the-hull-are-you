"""Generic YOLO inference - works on any frame."""
# shared-tracking-metrics/src/tracking_metrics/inference.py
from typing import Any
import yaml
from pathlib import Path

import numpy as np
from ultralytics import YOLO

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
        """Create model keyword arguments from model_config and tracker_config.
        
        Returns
        -------
        dict[str, Any]
            Keyword arguments for model.track()
        """
        kwargs = {}
        
        # Add model config parameters (remove None values)
        if self.model_config:
            kwargs.update({k: v for k, v in self.model_config.items() if v is not None})
        
        # Add tracker config if present
        if self.tracker_config:
            # Filter out None values
            tracker_params = {k: v for k, v in self.tracker_config.items() if v is not None}
            
            # Get tracker type (default to botsort)
            tracker_type = tracker_params.get("tracker_type", "botsort")
            
            # Create temporary tracker config file
            temp_dir = Path("temp_tracker_configs")
            temp_dir.mkdir(exist_ok=True)
            
            # Generate filename
            config_path = temp_dir / f"{tracker_type}_custom.yaml"
            
            # Write tracker config to file
            with open(config_path, "w") as f:
                yaml.dump(tracker_params, f)
            
            kwargs["tracker"] = str(config_path)
        
        return kwargs


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

    def predict_batch_frames(
        self,
        frames: list[np.ndarray],
    ) -> list[list[dict[str, Any]]]:
        """Run YOLO on multiple frames at once.
        
        Parameters
        ----------
        frames : List[np.ndarray]
            List of frames to process
            
        Returns
        -------
        List[List[Dict[str, Any]]]
            Detections for each frame
        """
        results = self.model.track(frames, **self.model_kwargs)
        
        all_detections = []
        for result in results:
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    detections.append(
                        {
                            "track_id": int(box.id[0]) if box.id is not None else -1,
                            "bbox": box.xyxy[0].tolist(),
                            "confidence": float(box.conf[0]),
                            "class_id": int(box.cls[0]),
                        }
                    )
            all_detections.append(detections)
        
        return all_detections