"""MLflow experiment runner for boat tracking hyperparameter tuning."""

import mlflow
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from itertools import product
import json
from loguru import logger

from src.models.detector import Detector
from src.models.tracker import Tracker
from src.processors.batch_processor import BatchProcessor
from src.processors.video_preprocessor import VideoPreprocessor


class MLflowRunner:
    """
    Orchestrate MLflow experiment runs with hyperparameter configurations.
    
    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment
    tracking_uri : str, optional
        MLflow tracking URI (default: 'file:./mlruns')
    """
    
    def __init__(self, experiment_name: str, tracking_uri: str = "file:./mlruns"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.setup_experiment()
    
    def setup_experiment(self):
        """Initialize MLflow experiment."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow experiment set: {self.experiment_name}")
        logger.info(f"Tracking URI: {self.tracking_uri}")
    
    def load_hyperparameter_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load hyperparameter configuration from YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def run_experiment(self, hyperparameters: Dict[str, Any], 
                      input_dir: str, output_dir: str,
                      model_path: str = "models/yolov8n.pt",
                      boat_classes: list = None,
                      run_name: Optional[str] = None):
        """
        Run a single experiment with given hyperparameters.
        
        Parameters
        ----------
        hyperparameters : Dict[str, Any]
            Dictionary of hyperparameters
        input_dir : str
            Input directory with preprocessed videos
        output_dir : str
            Output directory for processed videos
        model_path : str
            Path to YOLO model
        boat_classes : list, optional
            List of boat class names to detect
        run_name : str, optional
            Name for this MLflow run
        """
        boat_classes = boat_classes or ["boat", "ship"]
        
        with mlflow.start_run(run_name=run_name):
            logger.info(f"\nStarting MLflow run: {run_name or 'unnamed'}")
            logger.info("="*60)
            
            # Log hyperparameters
            self.log_hyperparameters(hyperparameters)
            
            # Extract hyperparameters
            confidence_threshold = hyperparameters.get('confidence_threshold', 0.5)
            target_height = hyperparameters.get('target_height', 720)
            botsort_config = hyperparameters.get('botsort_config', {})
            tracker_type = hyperparameters.get('tracker_type', 'botsort')
            
            # Preprocess videos if needed
            if 'preprocess' in hyperparameters and hyperparameters['preprocess']:
                logger.info(f"Preprocessing videos to height {target_height}...")
                preprocessor = VideoPreprocessor(
                    input_dir=hyperparameters.get('raw_input_dir', input_dir),
                    output_dir=input_dir,
                    target_height=target_height
                )
                preprocessor.preprocess_all()
            
            # Initialize detector and tracker
            detector = Detector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                target_classes=boat_classes
            )
            
            tracker = Tracker(
                tracker_type=tracker_type,
                config=botsort_config
            )
            
            # Run batch processing with metrics collection
            batch_processor = BatchProcessor(
                input_dir=input_dir,
                output_dir=output_dir,
                detector=detector,
                tracker=tracker,
                collect_metrics=True
            )
            
            logger.info(f"Processing videos with hyperparameters:")
            logger.info(f"  Confidence: {confidence_threshold}")
            logger.info(f"  Tracker: {tracker_type}")
            logger.info(f"  BOTSORT config: {botsort_config}")
            
            metrics = batch_processor.run()
            
            # Log metrics
            if metrics:
                self.log_metrics(metrics['aggregated'])
                
                # Save per-video metrics as artifact
                per_video_path = Path(output_dir) / "per_video_metrics.json"
                with open(per_video_path, 'w') as f:
                    json.dump(metrics['per_video'], f, indent=2)
                mlflow.log_artifact(str(per_video_path))
            
            logger.info("="*60)
            logger.info(f"✓ MLflow run completed")
            
            return metrics
    
    def run_experiment_from_config(self, config_path: str, 
                                   input_dir: str, output_dir: str,
                                   model_path: str = "models/yolov8n.pt",
                                   boat_classes: list = None):
        """
        Run experiment from YAML configuration file.
        
        Parameters
        ----------
        config_path : str
            Path to experiment configuration YAML file
        input_dir : str
            Input directory with preprocessed videos
        output_dir : str
            Output directory for processed videos
        model_path : str
            Path to YOLO model
        boat_classes : list, optional
            List of boat class names to detect
        """
        config = self.load_hyperparameter_config(config_path)
        run_name = config.get('experiment_name', Path(config_path).stem)
        
        return self.run_experiment(
            hyperparameters=config,
            input_dir=input_dir,
            output_dir=output_dir,
            model_path=model_path,
            boat_classes=boat_classes,
            run_name=run_name
        )
    
    def run_grid_search(self, config_path: str, 
                       input_dir: str, output_dir: str,
                       model_path: str = "models/yolov8n.pt",
                       boat_classes: list = None):
        """
        Run grid search from configuration file.
        
        Parameters
        ----------
        config_path : str
            Path to grid search configuration YAML file
        input_dir : str
            Input directory with preprocessed videos
        output_dir : str
            Output directory for processed videos
        model_path : str
            Path to YOLO model
        boat_classes : list, optional
            List of boat class names to detect
        """
        config = self.load_hyperparameter_config(config_path)
        
        if config.get('search_type') != 'grid':
            raise ValueError("Config file must have search_type: grid")
        
        parameters = config.get('parameters', {})
        
        # Extract parameter values for grid search
        confidence_thresholds = parameters.get('confidence_threshold', [0.5])
        target_heights = parameters.get('target_height', [720])
        
        # Handle nested botsort_config
        botsort_params = parameters.get('botsort_config', {})
        track_high_threshs = botsort_params.get('track_high_thresh', [0.5])
        track_low_threshs = botsort_params.get('track_low_thresh', [0.1])
        new_track_threshs = botsort_params.get('new_track_thresh', [0.4])
        track_buffers = botsort_params.get('track_buffer', [30])
        match_threshs = botsort_params.get('match_thresh', [0.8])
        fuse_scores = botsort_params.get('fuse_score', [True])
        gmc_methods = botsort_params.get('gmc_method', ['sparseOptFlow'])
        proximity_threshs = botsort_params.get('proximity_thresh', [0.5])
        appearance_threshs = botsort_params.get('appearance_thresh', [0.25])
        with_reids = botsort_params.get('with_reid', [False])
        models = botsort_params.get('model', ['auto'])
        
        # Generate all combinations
        combinations = list(product(
            confidence_thresholds,
            target_heights,
            track_high_threshs,
            track_low_threshs,
            new_track_threshs,
            track_buffers,
            match_threshs,
            fuse_scores,
            gmc_methods,
            proximity_threshs,
            appearance_threshs,
            with_reids,
            models
        ))
        
        logger.info(f"Running grid search with {len(combinations)} combinations")
        
        all_results = []
        
        for i, (conf, height, track_high_th, track_low_th, new_track_th, track_buf, 
                match_th, fuse_score, gmc_method, prox_th, appear_th, with_reid, model) in enumerate(combinations, 1):
            hyperparams = {
                'confidence_threshold': conf,
                'target_height': height,
                'tracker_type': 'botsort',
                'botsort_config': {
                    'track_high_thresh': track_high_th,
                    'track_low_thresh': track_low_th,
                    'new_track_thresh': new_track_th,
                    'track_buffer': track_buf,
                    'match_thresh': match_th,
                    'fuse_score': fuse_score,
                    'gmc_method': gmc_method,
                    'proximity_thresh': prox_th,
                    'appearance_thresh': appear_th,
                    'with_reid': with_reid,
                    'model': model
                }
            }
            
            run_name = f"grid_search_{i}_conf{conf}_h{height}_tht{track_high_th}_tlt{track_low_th}_ntt{new_track_th}_tb{track_buf}_mt{match_th}_fs{int(fuse_score)}_gmc{gmc_method}_pt{prox_th}_at{appear_th}_reid{int(with_reid)}"
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Grid Search Run {i}/{len(combinations)}")
            logger.info(f"{'='*60}")
            
            try:
                result = self.run_experiment(
                    hyperparameters=hyperparams,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    model_path=model_path,
                    boat_classes=boat_classes,
                    run_name=run_name
                )
                all_results.append({
                    'hyperparameters': hyperparams,
                    'metrics': result
                })
            except Exception as e:
                logger.info(f"✗ Error in run {i}: {e}")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Grid search completed: {len(all_results)}/{len(combinations)} successful")
        logger.info(f"{'='*60}")
        
        return all_results
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters to MLflow.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of hyperparameters
        """
        # Flatten nested parameters for logging
        flat_params = {}
        
        for key, value in params.items():
            if isinstance(value, dict):
                # Flatten nested dicts (e.g., botsort_config)
                for nested_key, nested_value in value.items():
                    flat_params[f"{key}.{nested_key}"] = nested_value
            else:
                flat_params[key] = value
        
        # Log all parameters
        mlflow.log_params(flat_params)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log metrics to MLflow.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of metrics
        """
        # Log all numeric metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
    
    def log_artifacts(self, artifact_paths: list):
        """
        Log artifacts to MLflow.
        
        Parameters
        ----------
        artifact_paths : list
            List of file paths to log as artifacts
        """
        for path in artifact_paths:
            if Path(path).exists():
                mlflow.log_artifact(path)
