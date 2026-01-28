"""Evaluator for unlabeled data using confidence-based metrics."""

# Standard library
from collections.abc import Generator
from itertools import product
from pathlib import Path
from typing import Any

# Third-party
import cv2
import mlflow
import numpy as np
from loguru import logger

# Local
from settings import DEFAULT_MLFLOW_URI
from tracking_metrics import MetricsCalculator, ModelInference, TrackingMetricsCollector


class UnlabeledEvaluator:
    """Evaluate model performance on unlabeled data."""

    # Constants
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
    DEFAULT_BATCH_SIZE = 16
    PROGRESS_LOG_INTERVAL = 500
    WEIGHTED_METRICS = [
        "confidence_mean",
        "bbox_area_mean",
        "bbox_jitter_mean",
        "avg_track_coverage",
        "detections_per_frame",
        "short_track_ratio",
    ]

    def __init__(self, model_path: Path):
        """Initialize evaluator with model.

        Parameters
        ----------
        model_path : Path
            Path to model weights
        """
        self.model_path = model_path
        self.collector = TrackingMetricsCollector()
        self.calculator = MetricsCalculator(self.collector)

    def _create_inference(self, config: dict[str, Any]) -> ModelInference:
        """Create model inference for given config.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing model_config and optional botsort_config

        Returns
        -------
        ModelInference
            Configured inference instance

        Raises
        ------
        ValueError
            If config structure is invalid (missing 'model_config' key)
        """
        # Validate config structure
        if "model_config" not in config:
            raise ValueError(
                "Invalid config structure: expected nested dict with 'model_config' key. "
                "Config should be: {'model_config': {...}, 'botsort_config': {...}}"
            )

        model_config = (
            config.get("model_config") if config.get("model_config") else None
        )
        botsort_config = config.get("botsort_config")

        return ModelInference(
            model_path=str(self.model_path),
            model_config=model_config,
            tracker_config=botsort_config,
        )

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

    def _read_video(self, cap: cv2.VideoCapture) -> Generator[np.ndarray]:
        """Read video frames.

        Yields
        ------
        np.ndarray
            Video frame
        """
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    def _process_batch(
        self, inference: ModelInference, frames: list[np.ndarray], frame_ids: list[int]
    ) -> None:
        """Process a batch of frames and collect detections.

        Parameters
        ----------
        inference : ModelInference
            Model inference instance
        frames : list[np.ndarray]
            Batch of frames to process
        frame_ids : list[int]
            Corresponding frame IDs
        """
        if not frames:
            return

        batch_detections = inference.predict_batch_frames(frames)
        for fid, detections in zip(frame_ids, batch_detections, strict=True):
            self.collector.add_batch_detection_with_track(detections, fid)
            self.collector.frame_count += 1

    def _process_single_video(
        self,
        video_path: Path,
        inference: ModelInference,
        frame_batch_size: int = 16,
    ) -> int:
        """Process single video and collect tracking data.

        Parameters
        ----------
        video_path : Path
            Path to video file
        inference : ModelInference
            Instance of inference class
        frame_batch_size : int
            Number of frames to process at once

        Returns
        -------
        int
            Total number of frames processed
        """
        logger.info(f"Processing video: {video_path}")
        self.collector.reset()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        props = self._get_video_properties(cap)
        frame_batch = []
        frame_ids = []
        last_frame_id = -1

        for frame_id, frame in enumerate(self._read_video(cap)):
            frame_batch.append(frame)
            frame_ids.append(frame_id)
            last_frame_id = frame_id

            if len(frame_batch) == frame_batch_size:
                self._process_batch(inference, frame_batch, frame_ids)
                frame_batch = []
                frame_ids = []

            if frame_id % self.PROGRESS_LOG_INTERVAL == 0:
                logger.debug(f"Processed frame {frame_id}")

        # Process remaining frames
        self._process_batch(inference, frame_batch, frame_ids)
        cv2.destroyAllWindows()

        # Return actual processed count, warn if differs from metadata
        actual_frame_count = last_frame_id + 1
        metadata_frame_count = props["frame_count"]

        if actual_frame_count != metadata_frame_count:
            logger.warning(
                f"Processed {actual_frame_count} frames but metadata indicates "
                f"{metadata_frame_count} frames for {video_path}"
            )

        return actual_frame_count

    def evaluate_single_video(
        self,
        video_path: Path | str,
        model_config: dict[str, Any],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict[str, Any]:
        """Evaluate model on single video.

        Parameters
        ----------
        video_path : Path | str
            Path to video file
        model_config : dict
            Configuration for model
        batch_size : int
            Number of frames to process at once

        Returns
        -------
        dict[str, Any]
            Dictionary containing video metadata and computed metrics
        """
        video_path = Path(video_path)
        inference = self._create_inference(model_config)
        frame_count = self._process_single_video(video_path, inference, batch_size)

        logger.info("Computing metrics...")
        metrics = self.calculator.compute_all_metrics(total_frames=frame_count)
        logger.success("Metrics calculated")

        return {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "frame_count": frame_count,
            "metrics": metrics,
        }

    def _calculate_weighted_metric(
        self,
        metric_name: str,
        video_metrics: list[dict[str, Any]],
        weights: list[int],
        total_frames: int,
    ) -> float:
        """Calculate weighted average for a single metric.

        Parameters
        ----------
        metric_name : str
            Name of metric to calculate
        video_metrics : list[dict]
            Per-video metrics
        weights : list[int]
            Frame counts for weighting
        total_frames : int
            Total frames across all videos

        Returns
        -------
        float
            Weighted average
        """
        values = [vm["metrics"].get(metric_name, 0) for vm in video_metrics]
        weighted_sum = sum(v * w for v, w in zip(values, weights, strict=True))
        return weighted_sum / total_frames if total_frames > 0 else 0.0

    def _compute_weighted_aggregates(
        self, video_metrics: list[dict[str, Any]]
    ) -> dict[str, float | int]:
        """Compute frame-weighted averages across multiple videos.

        Parameters
        ----------
        video_metrics : list[dict]
            List of dicts from evaluate_single_video()

        Returns
        -------
        dict[str, float]
            Weighted aggregated metrics
        """
        if not video_metrics:
            return {}

        weights = [vm["frame_count"] for vm in video_metrics]
        total_frames = sum(weights)

        # Calculate weighted metrics
        weighted = {
            f"weighted_{metric}": self._calculate_weighted_metric(
                metric, video_metrics, weights, total_frames
            )
            for metric in self.WEIGHTED_METRICS
        }

        # Add totals and counts
        weighted.update(
            {
                "total_videos": len(video_metrics),
                "total_frames": total_frames,
                "total_unique_tracks": sum(
                    vm["metrics"].get("num_tracks", 0) for vm in video_metrics
                ),
                "total_detections": sum(
                    vm["metrics"].get("total_detections", 0) for vm in video_metrics
                ),
            }
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
        list[Path]
            List of video file paths
        """
        video_files = [
            f
            for f in folder_path.iterdir()
            if f.suffix.lower() in self.VIDEO_EXTENSIONS
        ]

        if not video_files:
            logger.warning(f"No video files found in {folder_path}")
            return []

        logger.info(f"Found {len(video_files)} videos")
        return video_files

    def evaluate_unlabeled_videos(
        self,
        folder_path: Path | str,
        model_config: dict[str, Any],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict[str, Any]:
        """Evaluate model on multiple unlabeled videos with weighted aggregation.

        Parameters
        ----------
        folder_path : Path | str
            Path to folder containing video files
        model_config : dict
            Model configuration dictionary
        batch_size : int
            Number of frames to process at once

        Returns
        -------
        dict[str, Any]
            Dictionary with 'metrics' (aggregated metrics) and 'per_video_details' (list of per-video results)
        """
        folder_path = Path(folder_path)
        logger.info(f"Evaluating videos in: {folder_path}")

        video_files = self._get_video_files_in_folder(folder_path)
        if not video_files:
            return {"metrics": {}, "per_video_details": []}

        # Process each video
        per_video_metrics = []
        for video_path in video_files:
            logger.info(f"Processing {video_path.name}...")
            video_metrics = self.evaluate_single_video(
                video_path, model_config, batch_size
            )
            per_video_metrics.append(video_metrics)
            metrics_dict = video_metrics.get("metrics", {})
            if isinstance(metrics_dict, dict):
                num_tracks = metrics_dict.get("num_tracks", 0)
            else:
                num_tracks = 0
            logger.info(
                f"  {video_path.name}: {video_metrics['frame_count']} frames, "
                f"{num_tracks} tracks"
            )

        # Compute and return aggregated metrics
        logger.info("Computing aggregated metrics...")
        weighted_metrics = self._compute_weighted_aggregates(per_video_metrics)
        logger.success("Calculated aggregated metrics")

        return {
            "metrics": weighted_metrics,
            "per_video_details": per_video_metrics,
        }

    def _parse_param_values(self, param_config: Any) -> list[Any]:
        """Parse parameter config into list of values.

        Parameters
        ----------
        param_config : any
            Parameter configuration (list, dict, or single value)

        Returns
        -------
        list
            List of parameter values
        """
        if isinstance(param_config, list):
            return param_config
        return [param_config]

    def _flatten_search_space(
        self, search_space_config: dict[str, Any]
    ) -> dict[str, list[Any]]:
        """Flatten nested config structure into flat search space.

        Parameters
        ----------
        search_space_config : dict
            Nested configuration with model_config and botsort_config

        Returns
        -------
        dict[str, list]
            Flattened search space with composite keys
        """
        flattened_space = {}

        for top_key, top_value in search_space_config.items():
            if isinstance(top_value, dict):
                # Nested config - create composite keys
                for param_key, param_value in top_value.items():
                    composite_key = f"{top_key}.{param_key}"
                    flattened_space[composite_key] = self._parse_param_values(
                        param_value
                    )
            else:
                flattened_space[top_key] = self._parse_param_values(top_value)

        return flattened_space

    def _reconstruct_nested_config(self, flat_params: dict[str, Any]) -> dict[str, Any]:
        """Reconstruct nested config structure from flattened parameters.

        Parameters
        ----------
        flat_params : dict
            Flattened parameters with composite keys

        Returns
        -------
        dict
            Nested configuration with model_config and botsort_config
        """
        model_config = {
            key.replace("model_config.", ""): value
            for key, value in flat_params.items()
            if key.startswith("model_config.")
        }

        # Add non-prefixed params to model_config
        model_config.update(
            {
                key: value
                for key, value in flat_params.items()
                if not key.startswith(("model_config.", "botsort_config."))
            }
        )

        botsort_config = {
            key.replace("botsort_config.", ""): value
            for key, value in flat_params.items()
            if key.startswith("botsort_config.")
        }

        final_config = {"model_config": model_config}
        if botsort_config:
            final_config["botsort_config"] = botsort_config

        return final_config

    def _log_params_to_mlflow(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Parameters
        ----------
        params : dict
            Parameters to log (must contain 'model_config', optionally 'botsort_config')
        """
        # Logging the model path
        mlflow.log_param("model_path", str(self.model_path))

        if "model_config" in params:
            mlflow.log_params(params["model_config"])

        if "botsort_config" in params:
            mlflow.log_params(params["botsort_config"])

    def _run_single_experiment(
        self,
        item_path: Path,
        final_config: dict[str, Any],
        experiment_counter: int,
    ) -> None:
        """Run single hyperparameter experiment and log to MLflow.

        Parameters
        ----------
        item_path : Path
            Path to video file or folder
        final_config : dict
            Nested config for evaluation
        experiment_counter : int
            Experiment number
        """
        with mlflow.start_run(run_name=f"exp_{experiment_counter:04d}"):
            try:
                # Log parameters first so they're available even if experiment fails
                self._log_params_to_mlflow(final_config)

                # Validate path exists before processing
                if not item_path.exists():
                    raise ValueError(f"Path does not exist: {item_path}")

                # Select evaluation method based on path type
                if item_path.is_file():
                    result = self.evaluate_single_video(item_path, final_config)
                elif item_path.is_dir():
                    result = self.evaluate_unlabeled_videos(item_path, final_config)
                else:
                    raise ValueError(
                        f"Invalid path type (must be file or directory): {item_path}"
                    )

                # Log metrics on success
                mlflow.log_metrics(result["metrics"])
                logger.success(f"Logged experiment {experiment_counter}")

            except Exception as e:
                logger.error(f"Experiment {experiment_counter} failed: {e}")
                mlflow.log_params({"status": "failed", "error": str(e)})

    def search(
        self,
        search_space_config: dict[str, Any],
        path: str | Path,
        mlflow_uri: str | None = None,
        experiment_name: str = "botsort_hyperparam_search",
    ) -> None:
        """Hyperparameter search with logging to MLflow.

        Parameters
        ----------
        search_space_config : dict
            Configuration containing hyperparameter search space
        path : str | Path
            Path to single video file or folder
        mlflow_uri : str, optional
            MLflow tracking URI. If None, defaults to '<project_root>/output/mlruns'.
            Can be a relative path (resolved from current working directory) or absolute path.
        experiment_name : str
            MLflow experiment name

        Note
        ----
        If using a relative mlflow_uri, ensure you run this from the expected working
        directory (typically the model-training directory).
        """
        item_path = Path(path)

        # Set default MLflow URI if not provided
        if mlflow_uri is None:
            mlflow_uri = DEFAULT_MLFLOW_URI

        # Flatten search space
        flattened_space = self._flatten_search_space(search_space_config)
        hp_keys = list(flattened_space.keys())
        hp_values = list(flattened_space.values())

        # Validate search space is not empty
        if not flattened_space:
            raise ValueError(
                "Search space is empty. Please provide valid hyperparameters."
            )

        # Check for empty parameter lists
        empty_params = [k for k, v in flattened_space.items() if not v]
        if empty_params:
            raise ValueError(
                f"The following parameters have empty value lists: {', '.join(empty_params)}"
            )

        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)

        # Calculate and log search info
        total_combinations = int(np.prod([len(v) for v in hp_values]))
        logger.info(f"Starting search with {total_combinations} parameter combinations")

        # Run experiments
        for experiment_counter, hp_combo in enumerate(product(*hp_values)):
            flat_params = dict(zip(hp_keys, hp_combo, strict=True))
            final_config = self._reconstruct_nested_config(flat_params)

            logger.info(f"Experiment {experiment_counter + 1}/{total_combinations}")
            self._run_single_experiment(
                item_path=item_path,
                final_config=final_config,
                experiment_counter=experiment_counter,
            )

        logger.success(f"Completed {total_combinations} experiments")
