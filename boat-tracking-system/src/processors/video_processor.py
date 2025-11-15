class VideoProcessor:
    def __init__(self, input_path, output_path, detector, tracker):
        self.input_path = input_path
        self.output_path = output_path
        self.detector = detector
        self.tracker = tracker

    def process_video(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)
            tracked_frame = self.tracker.track(frame, detections)

            out.write(tracked_frame)

        cap.release()
        out.release()