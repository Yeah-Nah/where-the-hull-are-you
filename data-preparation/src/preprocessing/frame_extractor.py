"""Frame extraction from video files for data preparation."""

from pathlib import Path

import cv2
from loguru import logger


class FrameExtractor:
    """Extract single frames from video files at specified proportions."""

    # Supported video file extensions
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        proportion: float = 0.5,
        max_size_mb: int = 500,
    ):
        """Initialize the FrameExtractor.

        Parameters
        ----------
        input_dir : Path
            Directory containing video files to process.
        output_dir : Path
            Directory to save extracted frames.
        proportion : float, optional
            Proportion of video duration at which to extract the frame (0.0 to 1.0).
        max_size_mb : int, optional
            Maximum video file size in MB. Videos larger than this will be skipped.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.proportion = proportion
        self.max_size_mb = max_size_mb

        # Validate proportion
        if not 0.0 <= self.proportion <= 1.0:
            raise ValueError(
                f"Proportion must be between 0.0 and 1.0, got {self.proportion}"
            )

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames_from_directory(self) -> None:
        """Extract frames from all video files in the input directory.

        Processes all supported video files, filters by size, and extracts
        one frame per video at the specified proportion.
        """
        # Checking path exists and is a directory
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            logger.error(
                f"Input directory does not exist or is not a directory: {self.input_dir}"
            )
            return
        # Get all video files from input directory
        video_files = [
            f
            for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.VIDEO_EXTENSIONS
        ]

        if not video_files:
            logger.warning(f"No video files found in {self.input_dir}")
            return

        logger.info(f"Found {len(video_files)} video file(s) to process")

        processed_count = 0
        skipped_count = 0

        for video_path in video_files:
            logger.info(f"Processing video: {video_path.name}")

            # Check file size
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_size_mb:
                logger.warning(
                    f"Skipping {video_path.name} ({file_size_mb:.1f} MB exceeds limit of {self.max_size_mb} MB)"
                )
                skipped_count += 1
                continue

            # Extract frame from video
            success = self._extract_single_frame(video_path)
            if success:
                processed_count += 1
            else:
                skipped_count += 1

        logger.success(
            f"Frame extraction complete. Processed: {processed_count}, Skipped: {skipped_count}"
        )

    def _extract_single_frame(self, video_path: Path) -> bool:
        """Extract a single frame from a video file.

        Parameters
        ----------
        video_path : Path
            Path to the video file.

        Returns
        -------
        bool
            True if frame extraction was successful, False otherwise.
        """
        cap = None
        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path.name}")
                return False

            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                logger.error(f"Video has no frames: {video_path.name}")
                return False

            # Calculate target frame index
            target_frame_idx = int(total_frames * self.proportion)

            # Ensure we don't exceed frame count (for proportion = 1.0)
            if target_frame_idx >= total_frames:
                target_frame_idx = total_frames - 1

            # Set video position to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

            # Read frame
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.error(
                    f"Failed to read frame at index {target_frame_idx} from {video_path.name}"
                )
                return False

            # Generate output filename
            output_filename = f"{video_path.stem}_frame.png"
            output_path = self.output_dir / output_filename

            # Save frame
            cv2.imwrite(str(output_path), frame)
            logger.info(
                f"Successfully extracted frame from {video_path.name} "
                f"(frame {target_frame_idx}/{total_frames})"
            )

            return True

        except cv2.error as e:
            logger.error(f"OpenCV error while processing {video_path.name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while processing {video_path.name}: {e}")
            return False
        finally:
            # Ensure video capture is released
            if cap is not None:
                cap.release()
