import os
from pathlib import Path
from src.processors.video_processor import VideoProcessor

class BatchProcessor:
    """Batch process video files in the given input folder location.

    Parameters
    ----------
    input_dir : str
        Path to the input directory containing video files.
    output_dir : str
        Path to the output directory where processed videos will be saved.
    detector : object
        Pre-loaded object detector (e.g., BoatDetector instance).
    tracker : object
        Pre-loaded object tracker (e.g., BoatTracker instance).
    """
    def __init__(self, input_dir, output_dir, detector, tracker):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.detector = detector
        self.tracker = tracker
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def process_videos(self):
        """Process all video files in the input directory."""
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv')
        video_files = [f for f in os.listdir(self.input_dir) 
                      if f.lower().endswith(video_extensions)]
        
        print(f"Found {len(video_files)} video files to process")
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\nProcessing {i}/{len(video_files)}: {video_file}")
            
            input_path = os.path.join(self.input_dir, video_file)
            
            # Add _processed after file extension
            name, ext = os.path.splitext(video_file)
            output_filename = f"{name}_processed{ext}"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Create video processor and process
            video_processor = VideoProcessor(
                input_path=input_path,
                output_path=output_path,
                detector=self.detector,
                tracker=self.tracker
            )
            video_processor.process_video()
            print(f"✓ Saved to: {output_path}")

    def run(self):
        """Run the batch processing."""
        self.process_videos()
