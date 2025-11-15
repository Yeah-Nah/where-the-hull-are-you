import cv2

class Tracker:
    """Track objects across video frames using YOLO's built-in tracking."""
    
    def __init__(self, tracker_type='botsort'):
        """
        Initialize tracker.
        
        Parameters
        ----------
        tracker_type : str
            Tracking algorithm: 'botsort' (default), 'bytetrack', or 'deepocsort'
        """
        self.tracker_type = tracker_type
        self.tracker_config = f"{tracker_type}.yaml"
    
    def track(self, model, frame, confidence_threshold=0.5, target_class_ids=None):
        """
        Track objects in frame using YOLO's built-in tracking.
        
        Parameters
        ----------
        model : YOLO
            Pre-loaded YOLO model from Detector
        frame : numpy.ndarray
            Current video frame
        confidence_threshold : float
            Confidence threshold for detections
        target_class_ids : list, optional
            List of class IDs to track (filters detection results)
            
        Returns
        -------
        results : ultralytics.engine.results.Results
            Tracking results with track IDs
        """
        results = model.track(
            frame,
            conf=confidence_threshold,
            persist=True,
            tracker=self.tracker_config,
            classes=target_class_ids,  # Filter by class IDs
            verbose=False
        )
        return results
    
    def draw_tracks(self, frame, results):
        """
        Draw bounding boxes with track IDs on frame.
        
        Parameters
        ----------
        frame : numpy.ndarray
            Video frame
        results : ultralytics.engine.results.Results
            Tracking results from YOLO
            
        Returns
        -------
        annotated_frame : numpy.ndarray
            Frame with tracking annotations
        """
        annotated_frame = frame.copy()
        
        # Check if any tracks exist
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Get class names from results
            names = results[0].names
            
            for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw track ID, class name, and confidence
                class_name = names[cls_id]
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                
                # Add background to text for better visibility
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1
                )
                
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                )
        
        return annotated_frame
    
    def extract_tracks(self, results):
        """
        Extract tracking information as a list of dictionaries.
        
        Parameters
        ----------
        results : ultralytics.engine.results.Results
            Tracking results from YOLO
            
        Returns
        -------
        tracks : list
            List of tracked objects with IDs, bboxes, confidence, class
        """
        tracks = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            names = results[0].names
            
            for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                tracks.append({
                    'track_id': int(track_id),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': names[cls_id]
                })
        
        return tracks