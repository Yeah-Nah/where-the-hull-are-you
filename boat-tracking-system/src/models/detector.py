from ultralytics import YOLO

class Detector:
    def __init__(self, model_path, confidence_threshold=0.5, target_classes=None):
        """
        Initialize detector.
        
        Parameters
        ----------
        model_path : str
            Path to YOLO model
        confidence_threshold : float
            Confidence threshold for detections
        target_classes : list, optional
            List of class names to detect (e.g., ['boat', 'ship'])
            If None, detects all classes
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        
        # Convert class names to class IDs
        self.target_class_ids = None
        if target_classes:
            self.target_class_ids = [
                class_id for class_id, class_name in self.model.names.items()
                if class_name in target_classes
            ]
            print(f"Filtering for classes: {target_classes}")
            print(f"Class IDs: {self.target_class_ids}")

    def detect(self, frame):
        """Detect objects in a single frame (no tracking)."""
        # Use classes parameter to filter
        results = self.model(
            frame, 
            conf=self.confidence_threshold,
            classes=self.target_class_ids,  # Filter by class IDs
            verbose=False
        )
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0].cpu().numpy())
                confidence_score = box.conf[0].cpu().numpy()
                if confidence_score >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence_score),
                        'class_id': class_id,
                        'class_name': self.model.names[class_id]
                    })
        
        return detections

    def get_model(self):
        """Return the underlying YOLO model for tracking."""
        return self.model
    
    def get_class_ids(self):
        """Return the filtered class IDs."""
        return self.target_class_ids

    def get_classes(self):
        """Get class names from the model."""
        return self.model.names