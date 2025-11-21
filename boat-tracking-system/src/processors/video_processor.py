import cv2
import time

class VideoProcessor:
    def __init__(self, input_path, output_path, detector, tracker, metrics_collector=None):
        self.input_path = input_path
        self.output_path = output_path
        self.detector = detector
        self.tracker = tracker
        self.metrics_collector = metrics_collector
        self.previous_tracks = {}  # For IoU calculation

    def process_video(self):
        """Process video with detection and tracking."""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Track with class filtering
            results = self.tracker.track(
                self.detector.get_model(), 
                frame,
                confidence_threshold=self.detector.confidence_threshold,
                target_class_ids=self.detector.get_class_ids()  # Pass filtered classes
            )
            
            # Collect metrics if enabled
            if self.metrics_collector:
                # Extract detection data
                detections = self._extract_detections_from_results(results, frame.shape)
                self.metrics_collector.add_frame_detections(detections, frame.shape[:2])
                
                # Extract tracking data
                tracks = self._extract_tracks_from_results(results)
                self.metrics_collector.add_frame_tracks(tracks, frame_count, self.previous_tracks)
                self.previous_tracks = tracks
            
            tracked_frame = self.tracker.draw_tracks(frame, results)
            out.write(tracked_frame)
            
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames...")

        # Record total processing time
        if self.metrics_collector:
            total_time = time.time() - start_time
            self.metrics_collector.set_total_processing_time(total_time, frame_count)

        cap.release()
        out.release()
        print(f"✓ Completed: {frame_count} frames processed")
    
    def _extract_detections_from_results(self, results, frame_shape):
        """Extract detection info from YOLO results."""
        detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        return detections
    
    def _extract_tracks_from_results(self, results):
        """Extract tracking info from YOLO results."""
        tracks = {}
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for bbox, track_id in zip(boxes, track_ids):
                tracks[track_id] = bbox.tolist()
        
        return tracks