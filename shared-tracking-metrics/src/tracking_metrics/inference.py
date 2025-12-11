# shared-tracking-metrics/src/tracking_metrics/inference.py
from ultralytics import YOLO
from typing import List, Dict, Any
import numpy as np

class ModelInference:
    """Generic YOLO inference - works on any frame."""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
    
    def predict_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
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
        results = self.model.track(frame, persist=True)
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    "track_id": int(box.id[0]) if box.id is not None else -1,
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0])
                })
        
        return detections