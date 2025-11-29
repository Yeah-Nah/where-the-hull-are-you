import cv2
import os
from pathlib import Path
import shutil
from loguru import logger


class VideoPreprocessor:
    """Preprocess video to lower resolution for faster processing."""

    def __init__(self, input_dir, output_dir, target_height=720):
        """
        Initialize video preprocessor.

        Parameters
        ----------
        input_dir : str
            Directory containing original video files
        output_dir : str
            Directory to save preprocessed videos
        target_height : int
            Target height in pixels (default: 720 for HD)
            Width will be calculated to maintain aspect ratio
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_height = target_height

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Supported video extensions
        self.video_extensions = (".mp4", ".mov", ".avi", ".mkv")

    def get_video_files(self, directory):
        """Get all video files in directory."""
        video_files = [
            f
            for f in os.listdir(directory)
            if f.lower().endswith(self.video_extensions)
        ]
        return set(video_files)

    def get_video_resolution(self, video_path):
        """Get video resolution (width, height)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return width, height

    def resize_video(self, input_path, output_path):
        """
        Resize video to target resolution while maintaining aspect ratio.
        Handles both landscape and portrait orientations.

        Parameters
        ----------
        input_path : Path
            Path to input video
        output_path : Path
            Path to save resized video
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        # Get original video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine orientation and calculate new dimensions
        is_portrait = original_height > original_width

        if is_portrait:
            # For vertical videos, target_height applies to width (shorter dimension)
            new_width = self.target_height
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio)
            orientation = "Portrait"
        else:
            # For horizontal videos, target_height applies to height
            new_height = self.target_height
            aspect_ratio = original_width / original_height
            new_width = int(new_height * aspect_ratio)
            orientation = "Landscape"

        # Round to even numbers (required by some codecs)
        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1

        logger.info(f"  Orientation: {orientation}")
        logger.info(
            f"  Resizing from {original_width}x{original_height} to {new_width}x{new_height}"
        )

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_width, new_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            resized_frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            out.write(resized_frame)

            frame_count += 1
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(
                    f"    Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)"
                )

        cap.release()
        out.release()
        logger.info(f"  ✓ Resized video saved")

    def preprocess_all(self):
        """
        Preprocess all videos in input directory.

        Returns
        -------
        dict
            Summary of preprocessing results
        """
        logger.info("=" * 60)
        logger.info("Video Preprocessing")
        logger.info("=" * 60)
        logger.info(f"Input directory:  {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Target resolution: {self.target_height}p (HD)")
        logger.info("=" * 60)

        # Get video files from both directories
        input_videos = self.get_video_files(self.input_dir)
        output_videos = self.get_video_files(self.output_dir)

        # Find videos that need processing
        videos_to_process = input_videos - output_videos

        if not videos_to_process:
            logger.info("\n✓ All videos have already been preprocessed!")
            logger.info(f"  {len(input_videos)} video(s) found in output directory")
            return {
                "total": len(input_videos),
                "already_processed": len(input_videos),
                "copied": 0,
                "resized": 0,
                "skipped": 0,
            }

        logger.info(f"\nFound {len(videos_to_process)} video(s) to process:")
        for video in sorted(videos_to_process):
            logger.info(f"{video}")

        stats = {
            "total": len(videos_to_process),
            "already_processed": len(output_videos),
            "copied": 0,
            "resized": 0,
            "skipped": 0,
        }

        for i, video_file in enumerate(videos_to_process, 1):
            logger.info(f"\n[{i}/{len(videos_to_process)}] Processing: {video_file}")

            input_path = self.input_dir / video_file
            output_path = self.output_dir / video_file

            # Get video resolution
            width, height = self.get_video_resolution(input_path)

            if width is None or height is None:
                logger.info(f"  ✗ Error: Could not read video file")
                stats["skipped"] += 1
                continue

            logger.info(f"  Original resolution: {width}x{height}")

            # Determine orientation and check if resizing needed
            is_portrait = height > width
            shorter_dimension = width if is_portrait else height

            # Check if video needs resizing based on shorter dimension
            if shorter_dimension <= self.target_height:
                # Video is already at or below target - just copy
                logger.info(
                    f"  Already at or below {self.target_height}p - copying as-is"
                )
                shutil.copy2(input_path, output_path)
                stats["copied"] += 1
                logger.success(f"  ✓ Copied to output directory")
            else:
                # Video needs resizing
                logger.warning(
                    f"  Needs resizing (shorter dimension > {self.target_height}p)"
                )
                try:
                    self.resize_video(input_path, output_path)
                    stats["resized"] += 1
                except Exception as e:
                    logger.info(f"  ✗ Error resizing video: {e}")
                    stats["skipped"] += 1

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.success("Preprocessing Complete!")
        logger.info("=" * 60)
        logger.info(f"Total videos processed: {stats['total']}")
        logger.info(f"Already preprocessed:   {stats['already_processed']}")
        logger.info(f"Copied (HD or lower):   {stats['copied']}")
        logger.info(f"Resized (>HD):          {stats['resized']}")
        logger.info(f"Skipped (errors):       {stats['skipped']}")
        logger.info("=" * 60)

        return stats
