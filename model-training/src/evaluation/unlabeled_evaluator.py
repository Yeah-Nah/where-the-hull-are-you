"""Evaluator for unlabeled data using confidence-based metrics."""

from pathlib import Path

import cv2
from loguru import logger
from tracking_metrics import MetricsCalculator, TrackingMetricsCollector
from tracking_metrics.inference import ModelInference


class UnlabeledEvaluator:
    """Evaluate model performance on unlabeled data."""

    def __init__(self, model_path: Path, model_config: dict):
        """Initialize evaluator with model.

        Parameters
        ----------
        model_path : Path
            Path to model weights
        model_config : Dict
            Model configuration parameters
        """
        self.model_path = model_path
        self.model_config = model_config
        self.inference = ModelInference(
            model_path=str(model_path), model_config=model_config
        )
        self.collector = TrackingMetricsCollector()
        self.calculator = MetricsCalculator(self.collector)

    def _initiate_cap(self, video_path: str):
        """Inititate the video capture cv2 object."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        return cap

    def _get_video_properties(self, cap: cv2.VideoCapture) -> dict[str, int]:
        """Get video properties efficiently.

        Parameters
        ----------
        cap : cv2.VideoCapture
            Opened video capture object

        Returns
        -------
        Dict[str, int]
            Video properties (width, height, fps, frame_count)
        """
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def _read_video(self, cap):
        """Read video frames helper.

        Yields
        ------
        Tuple[int, np.ndarray]
            (frame_id, frame)
        """
        try:
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame_id, frame
                frame_id += 1
        finally:
            cap.release()

    def _process_single_video(
        self,
        video_path: Path,
        frame_batch_size: int = 16,
    ) -> int:
        """Process single video and collect tracking data.

        Parameters
        ----------
        video_path : Path
            Path to video file
        frame_batch_size : int
            Number of frames to process at once

        Returns
        -------
        int
            Total number of frames processed
        """
        logger.info(f"Processing video: {video_path}")
        self.collector.reset()

        cap = self._initiate_cap(video_path)
        props = self._get_video_properties(cap)

        frame_batch = []
        frame_ids = []

        for frame_id, frame in self._read_video(cap):
            frame_batch.append(frame)
            frame_ids.append(frame_id)

            # Process batch when full
            if len(frame_batch) == frame_batch_size:
                batch_detections = self.inference.predict_batch_frames(frame_batch)

                for fid, detections in zip(frame_ids, batch_detections, strict=True):
                    self.collector.add_batch_detection_with_track(detections, fid)
                    self.collector.frame_count += 1

                frame_batch = []
                frame_ids = []

            if frame_id % 500 == 0:
                logger.debug(f"Processed frame {frame_id}")

        # Process remaining frames
        if frame_batch:
            batch_detections = self.inference.predict_batch_frames(frame_batch)
            for fid, detections in zip(frame_ids, batch_detections, strict=True):
                self.collector.add_batch_detection_with_track(detections, fid)
                self.collector.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        return props["frame_count"]

    def evaluate_single_video(
        self,
        video_path: Path,
        batch_size: int = 16,  # Adjust based on GPU memory
    ) -> dict[str, float]:
        """Evaluate model on single video.

        Parameters
        ----------
        video_path : Path
            Path to video file
        batch_size : int
            Number of frames to process at once

        Returns
        -------
        Dict[str, float]
            Computed metrics
        """
        self.collector.reset()

        # Convert to Path if string
        video_path = Path(video_path)

        frame_count = self._process_single_video(video_path, batch_size)

        logger.info("Computing metrics...")
        metrics = self.calculator.compute_all_metrics(total_frames=frame_count)
        logger.success("Metrics Calculated")

        return {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "frame_count": frame_count,
            "metrics": metrics,
        }

    def _compute_weighted_aggregates(
        self, video_metrics: list[dict]
    ) -> dict[str, float]:
        """Compute frame-weighted averages across multiple videos.

        Parameters
        ----------
        video_metrics : List[Dict]
            List of dicts from evaluate_single_video()

        Returns
        -------
        Dict[str, float]
            Weighted aggregated metrics
        """
        if not video_metrics:
            return {}

        # Extract frame counts as weights
        weights = [vm["frame_count"] for vm in video_metrics]
        total_frames = sum(weights)

        # TODO: Need to use a different calculation for the std metrics
        # Metrics to weight by frame count
        metrics_to_weight = [
            "confidence_mean",
            # "confidence_std",
            "bbox_area_mean",
            # "bbox_area_std",
            "bbox_jitter_mean",
            # "bbox_jitter_std",
            "avg_track_coverage",
            "detections_per_frame",
            "short_track_ratio",
        ]

        weighted = {}

        # Compute weighted averages
        for metric in metrics_to_weight:
            values = [vm["metrics"].get(metric, 0) for vm in video_metrics]
            if all(v == 0 for v in values):
                weighted[f"weighted_{metric}"] = 0.0
            else:
                weighted_value = (
                    sum(v * w for v, w in zip(values, weights, strict=True)) / total_frames
                )
                weighted[f"weighted_{metric}"] = weighted_value

        # Totals and counts
        weighted["total_videos"] = len(video_metrics)
        weighted["total_frames"] = total_frames
        weighted["total_unique_tracks"] = sum(
            vm["metrics"].get("num_tracks", 0) for vm in video_metrics
        )
        weighted["total_detections"] = sum(
            vm["metrics"].get("total_detections", 0) for vm in video_metrics
        )

        return weighted

    def _get_video_files_in_folder(self, folder_path: Path) -> list[Path]:
        """Get list of video files in a folder.

        Parameters
        ----------
        folder_path : Path
            Path to folder

        Returns
        -------
        List[Path]
            List of video file paths
        """
        # Get all video files
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        video_files = [
            f for f in folder_path.iterdir() if f.suffix.lower() in video_extensions
        ]

        if not video_files:
            logger.warning(f"No video files found in {folder_path}")
            return {}

        logger.info(f"Found {len(video_files)} videos")

        return video_files

    def evaluate_unlabeled_videos(
        self, folder_path: Path, batch_size: int = 16
    ) -> dict[str, float]:
        """Evaluate model on multiple unlabeled videos with weighted aggregation.

        Parameters
        ----------
        folder_path : Path
            Path to folder containing video files
        batch_size : int
            Number of frames to process at once

        Returns
        -------
        Dict[str, float]
            Aggregated metrics across all videos
        """
        logger.info(f"Evaluating videos in: {folder_path}")
        folder_path = Path(folder_path)
        video_files = self._get_video_files_in_folder(folder_path)

        # Process each video and collect metrics
        per_video_metrics = []
        for video_path in video_files:
            logger.info(f"Processing {video_path.name}...")
            video_metrics = self.evaluate_single_video(video_path, batch_size)
            per_video_metrics.append(video_metrics)
            logger.info(
                f"  {video_path.name}: {video_metrics['frame_count']} frames, "
                f"{video_metrics['metrics'].get('num_tracks', 0)} tracks"
            )

        # Compute aggregated metrics
        logger.info("Computing aggregated metrics...")
        weighted_metrics = self._compute_weighted_aggregates(per_video_metrics)

        # Merge all metrics
        final_metrics = {}
        final_metrics.update(weighted_metrics)
        final_metrics["per_video_details"] = per_video_metrics

        # Log results
        logger.success("Calculated aggregated metrics")

        return final_metrics

    def search(self, search_space, video_paths):
        """Hyperparameter search with logging to MLFlow.
        
        Parameters
        ----------
        search_space : config
            Config containing hyperparameter search space.
        video_paths : path or str
            Path to single video file, or folder path containing multiple video files.
        
        Returns
        -------
        TBD
        """