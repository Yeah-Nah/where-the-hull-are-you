class BatchProcessor:
    def __init__(self, input_dir, output_dir, model, confidence_threshold):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model
        self.confidence_threshold = confidence_threshold

    def process_videos(self):
        import os
        from src.processors.video_processor import VideoProcessor

        video_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
        for video_file in video_files:
            input_path = os.path.join(self.input_dir, video_file)
            output_path = os.path.join(self.output_dir, f"processed_{video_file}")
            video_processor = VideoProcessor(input_path, output_path, self.model, self.confidence_threshold)
            video_processor.process_video()

    def run(self):
        self.process_videos()