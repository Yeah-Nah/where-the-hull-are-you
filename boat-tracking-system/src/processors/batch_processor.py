import os
from pathlib import Path
from src.processors.video_processor import VideoProcessor
from src.metrics.metrics_aggregator import MetricsAggregator
from src.metrics.video_metrics_collector import VideoMetricsCollector
from loguru import logger


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
    collect_metrics : bool, optional
        Whether to collect metrics during processing (default: False).
    """

    def __init__(self, input_dir, output_dir, detector, tracker, collect_metrics=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.detector = detector
        self.tracker = tracker
        self.collect_metrics = collect_metrics
        self.metrics_aggregator = MetricsAggregator() if collect_metrics else None

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def process_videos(self):
        """Process all video files in the input directory."""
        video_extensions = (".mp4", ".mov", ".avi", ".mkv")
        video_files = [
            f
            for f in os.listdir(self.input_dir)
            if f.lower().endswith(video_extensions)
        ]

        logger.info(f"Found {len(video_files)} video files to process")

        for i, video_file in enumerate(video_files, 1):
            logger.info(f"\nProcessing {i}/{len(video_files)}: {video_file}")

            input_path = os.path.join(self.input_dir, video_file)

            # Add _processed after file extension
            name, ext = os.path.splitext(video_file)
            output_filename = f"{name}_processed{ext}"
            output_path = os.path.join(self.output_dir, output_filename)

            # Create metrics collector if enabled
            metrics_collector = (
                VideoMetricsCollector(video_file) if self.collect_metrics else None
            )

            # Create video processor and process
            video_processor = VideoProcessor(
                input_path=input_path,
                output_path=output_path,
                detector=self.detector,
                tracker=self.tracker,
                metrics_collector=metrics_collector,
            )
            video_processor.process_video()
            logger.info(f"✓ Saved to: {output_path}")

            # Collect video metrics if enabled
            if self.collect_metrics and metrics_collector:
                video_metrics = metrics_collector.compute_video_metrics()
                self.metrics_aggregator.add_video_metrics(video_file, video_metrics)
                logger.info(f"✓ Metrics collected for {video_file}")

    def run(self):
        """Run the batch processing."""
        self.process_videos()

        if self.collect_metrics:
            aggregated_metrics = self.metrics_aggregator.compute_aggregated_metrics()
            logger.info("\n" + "=" * 60)
            logger.info(self.metrics_aggregator.get_summary_string())
            logger.info("=" * 60)
            return aggregated_metrics

        return None
