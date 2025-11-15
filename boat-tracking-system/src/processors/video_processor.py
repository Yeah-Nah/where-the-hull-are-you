import cv2

class VideoProcessor:
    def __init__(self, input_path, output_path, detector, tracker):
        self.input_path = input_path
        self.output_path = output_path
        self.detector = detector
        self.tracker = tracker

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
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Track with class filtering
            results = self.tracker.track(
                self.detector.get_model(), 
                frame,
                confidence_threshold=self.detector.confidence_threshold,
                target_class_ids=self.detector.get_class_ids()  # Pass filtered classes
            )
            
            tracked_frame = self.tracker.draw_tracks(frame, results)
            out.write(tracked_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames...")

        cap.release()
        out.release()
        print(f"✓ Completed: {frame_count} frames processed")