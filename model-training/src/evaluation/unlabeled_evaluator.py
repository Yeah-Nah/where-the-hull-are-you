"""Evaluator for unlabeled data using confidence-based metrics."""

from pathlib import Path

import cv2
from loguru import logger
from itertools import product
import mlflow
from tracking_metrics import MetricsCalculator, TrackingMetricsCollector
from tracking_metrics.inference import ModelInference
import numpy as np


class UnlabeledEvaluator:
    """Evaluate model performance on unlabeled data."""

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

    def _create_inference(self, config) -> ModelInference:
        """Create model infernce for given config.

        Parameters
        ----------
        model_config : config
            Model config

        Returns
        -------
        ModelInference
            Configured inference instance
        """

        if "botsort_config" in config:
            botsort_config = config.get("botsort_config")
            model_config = config.get("model_config")
        else:
            botsort_config = None

        return ModelInference(
            model_path=str(self.model_path),
            model_config=model_config,
            tracker_config=botsort_config,
        )

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

        cap = self._initiate_cap(video_path)
        props = self._get_video_properties(cap)

        frame_batch = []
        frame_ids = []

        for frame_id, frame in self._read_video(cap):
            frame_batch.append(frame)
            frame_ids.append(frame_id)

            # Process batch when full
            if len(frame_batch) == frame_batch_size:
                batch_detections = inference.predict_batch_frames(frame_batch)

                for fid, detections in zip(frame_ids, batch_detections, strict=True):
                    self.collector.add_batch_detection_with_track(detections, fid)
                    self.collector.frame_count += 1

                frame_batch = []
                frame_ids = []

            if frame_id % 500 == 0:
                logger.debug(f"Processed frame {frame_id}")

        # Process remaining frames
        if frame_batch:
            batch_detections = inference.predict_batch_frames(frame_batch)
            for fid, detections in zip(frame_ids, batch_detections, strict=True):
                self.collector.add_batch_detection_with_track(detections, fid)
                self.collector.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        return props["frame_count"]

    def evaluate_single_video(
        self,
        video_path: Path,
        model_config: dict,
        batch_size: int = 16,  # Adjust based on GPU memory
    ) -> dict[str, float]:
        """Evaluate model on single video.

        Parameters
        ----------
        video_path : Path
            Path to video file
        model_config : dict
            Config for model
        batch_size : int
            Number of frames to process at once

        Returns
        -------
        Dict[str, float]
            Computed metrics
        """
        self.collector.reset()

        # Create inference class with given config
        inference = self._create_inference(model_config)

        # Convert to Path if string
        video_path = Path(video_path)

        frame_count = self._process_single_video(video_path, inference, batch_size)

        logger.info("Computing metrics...")
        metrics = self.calculator.compute_all_metrics(total_frames=frame_count)
        metrics["frame_count"] = frame_count
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
        self, folder_path: Path, model_config: dict, batch_size: int = 16
    ) -> dict[str, float]:
        """Evaluate model on multiple unlabeled videos with weighted aggregation.

        Parameters
        ----------
        folder_path : Path
            Path to folder containing video files
        model_config : dict
            Model config as a dictionary
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
            video_metrics = self.evaluate_single_video(video_path, model_config, batch_size)
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
        final_metrics["metrics"] = weighted_metrics
        final_metrics["per_video_details"] = per_video_metrics

        # Log results
        logger.success("Calculated aggregated metrics")

        return final_metrics

    def _parse_param_values(self, param_config) -> list:
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
        elif isinstance(param_config, dict):
            return [param_config]
        else:
            return [param_config]

    def _flatten_search_space(self, search_space_config: dict) -> dict[str, list]:
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
                    flattened_space[composite_key] = self._parse_param_values(param_value)
            else:
                flattened_space[top_key] = self._parse_param_values(top_value)
        
        return flattened_space

    def _reconstruct_nested_config(self, flat_params: dict) -> dict:
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
        model_config = {}
        botsort_config = {}
        
        for key, value in flat_params.items():
            if key.startswith("model_config."):
                param_name = key.replace("model_config.", "")
                model_config[param_name] = value
            elif key.startswith("botsort_config."):
                param_name = key.replace("botsort_config.", "")
                botsort_config[param_name] = value
            else:
                model_config[key] = value
        
        # Create final config structure
        final_config = {"model_config": model_config}
        if botsort_config:
            final_config["botsort_config"] = botsort_config
        
        return final_config

    def _log_params_to_mlflow(self, params: dict) -> None:
        """Log parameters to MLflow.
        
        Parameters
        ----------
        params : dict
            Parameters to log
        """
        print(params['model_config'])
        print(params["botsort_config"])
        mlflow.log_params(params['model_config'])
        if 'botsort_config' in params:
            mlflow.log_params(params['botsort_config'])

    def _run_single_experiment(
        self, 
        item_path: Path, 
        flat_params: dict, 
        final_config: dict,
        experiment_counter: int
    ) -> None:
        """Run single hyperparameter experiment and log to MLflow.
        
        Parameters
        ----------
        item_path : Path
            Path to video file or folder
        flat_params : dict
            Flattened parameters for MLflow logging
        final_config : dict
            Nested config for evaluation
        experiment_counter : int
            Experiment number
        """
        with mlflow.start_run(run_name=f"exp_{experiment_counter:04d}"):
            
            try:
                if item_path.is_file():
                    print(final_config)
                    result = self.evaluate_single_video(item_path, final_config)
                    print(final_config)
                    self._log_params_to_mlflow(final_config)
                    mlflow.log_metrics(result["metrics"])
                    
                elif item_path.is_dir():
                    result = self.evaluate_unlabeled_videos(item_path, final_config)
                    self._log_params_to_mlflow(final_config)
                    mlflow.log_metrics(result["metrics"])
                else:
                    raise ValueError(f"Invalid path: {item_path}")
                
                logger.success(f"Logged experiment {experiment_counter}")
                
            except Exception as e:
                logger.error(f"Experiment {experiment_counter} failed: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))

    def search(self, search_space_config: dict, path: str | Path) -> None:
        """Hyperparameter search with logging to MLflow.
        
        Parameters
        ----------
        search_space_config : dict
            Config containing hyperparameter search space
        path : str or Path
            Path to single video file or folder
        """
        
        # Flatten nested search space
        flattened_space = self._flatten_search_space(search_space_config)
        hp_keys = list(flattened_space.keys())
        hp_values = list(flattened_space.values())
        
        # Setup MLflow
        item_path = Path(path)
        mlflow.set_tracking_uri("file:../output/mlruns")
        mlflow.set_experiment("botsort_hyperparam_search")
        
        # Log search info
        total_combinations = np.prod([len(v) for v in hp_values])
        logger.info(f"Starting search with {total_combinations} parameter combinations")
        
        # Run experiments
        for experiment_counter, hp_combo in enumerate(product(*hp_values)):
            flat_params = dict(zip(hp_keys, hp_combo, strict=False))
            final_config = self._reconstruct_nested_config(flat_params)
            
            logger.info(f"Experiment {experiment_counter + 1}/{total_combinations}")
            
            self._run_single_experiment(
                item_path=item_path,
                flat_params=flat_params,
                final_config=final_config,
                experiment_counter=experiment_counter
            )
        
        logger.success(f"Completed {total_combinations} experiments")
