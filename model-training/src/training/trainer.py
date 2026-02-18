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

    def _create_data_yaml(self) -> Path:
        """Create data.yaml file for YOLO training in data directory.

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

        # Use relative path (. = current directory where data.yaml sits)
        data_config = {
            "path": ".",  # ✓ Relative to data.yaml location
            "train": "images",
            "val": "images",
            "names": dict(enumerate(classes)),
            "nc": len(classes),
        }

        # Write to data directory
        output_file = data_dir / "data.yaml"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

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

    def _create_kwargs(self) -> dict[str, Any]:
        """Create model keyword arguments from model configuration.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for model.train()
        """
        kwargs = {}

        # Create data.yaml file for training in shared data directory
        data_yaml = self._create_data_yaml()
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
        # Create unique run name
        self.training_run_name = self._create_training_run_name()

        # Build training kwargs (includes training_run_name and data.yaml in run dir)
        training_kwargs = self._create_kwargs()

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
