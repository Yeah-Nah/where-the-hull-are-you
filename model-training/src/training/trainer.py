"""Model training orchestration."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from ultralytics import YOLO


class ModelTrainer:
    """Train custom YOLO models for maritime object detection."""

    def __init__(self, training_config: dict[str, Any]):
        """Initialize trainer with configuration.

        Parameters
        ----------
        training_config : Dict[str, Any]
            Training configuration including model, data, hyperparameters
        """
        self.model_config = training_config.get("model_config", {})
        self.output_config = training_config.get("output_config", {})
        self.base_dir = Path(__file__).parent.parent.parent
        self.base_model = self._load_base_model()

    def _create_data_yaml(self, output_dir: Path) -> Path:
        """Create data.yaml file for YOLO training in run-specific directory.

        Parameters
        ----------
        output_dir : Path
            Directory where data.yaml should be created (run-specific)

        Returns
        -------
        Path
            Path to created data.yaml file
        """
        data_dir = self.base_dir / "data"

        # Read classes from classes.txt
        classes_file = data_dir / "classes.txt"
        with open(classes_file) as f:
            classes = [line.strip() for line in f if line.strip()]

        # Create data.yaml content with relative path
        # Use relative path if possible, otherwise use absolute path
        try:
            path_value = str(data_dir.relative_to(self.base_dir))
        except ValueError:
            # If data_dir is not relative to base_dir, use absolute path
            path_value = str(data_dir.absolute())

        data_config = {
            "path": path_value,
            "train": "images",
            "val": "images",  # Using same as train - no validation split
            "names": dict(enumerate(classes)),
            "nc": len(classes),
        }

        # Write to run-specific output directory to avoid dirty working tree
        output_file = output_dir / "data.yaml"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created data.yaml at: {output_file}")
        return output_file

    def _create_training_run_name(self) -> str:
        """Create unique training run name with datetime.

        Returns
        -------
        str
            Unique training run name
        """
        base_name = self.output_config.get("training_run_name", "custom_model")
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{date_str}"

    def _create_kwargs(self, output_dir: Path) -> dict[str, Any]:
        """Create model keyword arguments from model configuration.

        Parameters
        ----------
        output_dir : Path
            Directory for run-specific outputs (data.yaml, etc.)

        Returns
        -------
        dict[str, Any]
            Keyword arguments for model.train()
        """
        kwargs = {}

        # Create data.yaml file for training in run-specific directory
        data_yaml = self._create_data_yaml(output_dir)
        kwargs["data"] = str(data_yaml)

        # Use the training run name that was already set
        kwargs["name"] = self.training_run_name

        # Add model config parameters (remove None values and exclude 'model' key)
        if self.model_config:
            kwargs.update(
                {
                    k: v
                    for k, v in self.model_config.items()
                    if v is not None and k != "model"
                }
            )

        return kwargs

    def _load_base_model(self) -> YOLO:
        """Load base model for transfer learning.

        Returns
        -------
        YOLO
            Loaded YOLO model for transfer learning

        Raises
        ------
        ValueError
            If model name/path is not specified or empty
        FileNotFoundError
            If the specified model file does not exist
        """
        base_model = self.model_config.get("model", "")

        # Validate that model is specified and not empty
        if not base_model or not str(base_model).strip():
            raise ValueError(
                "Base model name or path is not specified in training configuration."
            )

        base_model_path = self.base_dir / "models" / base_model

        # Check if the model path exists
        if not base_model_path.exists():
            msg = (
                f"Base model not found: {base_model_path}. "
                f"Please ensure the model exists in the models directory."
            )
            raise FileNotFoundError(msg)

        logger.success(f"Base model loaded: {base_model_path}")
        return YOLO(str(base_model_path))

    def _save_to_custom_folder(
        self,
        best_weights_path: Path,
    ) -> None:
        """Save trained model to custom models folder.

        Parameters
        ----------
        best_weights_path : Path
            Path to best.pt from training

        Returns
        -------
        None
        """
        custom_dir = self.base_dir / "models" / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        dest_path = custom_dir / f"{self.training_run_name}.pt"

        # Copy best weights to custom folder
        shutil.copy2(best_weights_path, dest_path)

        logger.info(f"Model saved to: {dest_path}")

    def train(self) -> None:
        """Train the model.

        Returns
        -------
        None
        """
        # Get output directory from config or use default
        project_dir = self.base_dir / "runs" / "train"

        # Create unique run name
        self.training_run_name = self._create_training_run_name()

        # Create run-specific directory for data.yaml and other outputs
        output_dir = project_dir / self.training_run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build training kwargs (includes training_run_name and data.yaml in run dir)
        training_kwargs = self._create_kwargs(output_dir)

        logger.info("Starting training with configuration:")
        for k, v in training_kwargs.items():
            logger.info(f"  {k}: {v}")

        # Train the model (progress is automatically displayed by Ultralytics)
        results = self.base_model.train(**training_kwargs)

        # Get the best weights path from results
        best_weights = Path(results.save_dir) / "weights" / "best.pt"

        logger.success(f"Training completed! Best weights: {best_weights}")

        # Save best weights to custom models folder
        self._save_to_custom_folder(best_weights)
