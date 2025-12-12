"""Load and manage test videos."""

from pathlib import Path


class VideoLoader:
    """Load test videos for evaluation."""

    def __init__(self, video_dir: Path):
        """Initialize video loader.

        Parameters
        ----------
        video_dir : Path
            Directory containing test videos
        """
        self.video_dir = Path(video_dir)

    def get_all_videos(self, extensions: list[str] = None) -> list[Path]:
        """Get all video files in directory.

        Parameters
        ----------
        extensions : List[str], optional
            List of video file extensions to include

        Returns
        -------
        List[Path]
            List of video file paths
        """
        if extensions is None:
            extensions = [".mp4", ".avi", ".mov", ".mkv"]

        videos = []
        for ext in extensions:
            videos.extend(self.video_dir.glob(f"*{ext}"))

        return sorted(videos)

    def get_video_by_name(self, name: str) -> Path:
        """Get specific video by name.

        Parameters
        ----------
        name : str
            Video file name

        Returns
        -------
        Path
            Path to video file
        """
        video_path = self.video_dir / name
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return video_path
