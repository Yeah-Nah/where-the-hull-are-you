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
        self.base_model = self._load_base_model()

    def _create_data_yaml(self) -> str:
        """Create data.yaml file for YOLO training.

        Parameters
        ----------
        output_path : Path
            Path where data.yaml should be created

        Returns
        -------
        Path
            Path to created data.yaml file
        """
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "data"

        # Read classes from classes.txt
        classes_file = data_dir / "classes.txt"
        with open(classes_file) as f:
            classes = [line.strip() for line in f if line.strip()]

        # Create data.yaml content
        data_config = {
            "path": str(data_dir.absolute()),
            "train": "images",
            "val": "images",  # Using same as train - no validation split
            "names": dict(enumerate(classes)),
            "nc": len(classes),
        }

        # Write to file
        output_file = data_dir / "data.yaml"
        with open(output_file, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created data.yaml at: {output_file}")
        return str(output_file)

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
            Keyword arguments for model.track()
        """
        kwargs = {}

        # Create data.yaml file for training
        data_yaml = self._create_data_yaml()
        kwargs["data"] = data_yaml

        # Create custom training run name with datetime for uniqueness
        self.training_run_name = self._create_training_run_name()
        kwargs["name"] = self.training_run_name

        # Add model config parameters (remove None values)
        if self.model_config:
            kwargs.update({k: v for k, v in self.model_config.items() if v is not None})

        return kwargs

    def _load_base_model(self) -> YOLO:
        """Load base model for transfer learning.

        Parameters
        ----------
        base_model_path : Path
            Path to the base model for transfer learning
        """
        base_dir = Path(__file__).parent.parent.parent
        base_model = self.model_config.get("model", "")
        base_model_path = base_dir / "models" / base_model
        # Check the model path exists
        if base_model_path is None:
            raise ValueError(
                "Base model path is not specified in training configuration."
            )
        if not base_model_path.exists():
            msg = (
                f"Base model not found: {base_model_path}. "
                f"Please ensure the model exists in the models directory."
            )
            raise FileNotFoundError(msg)
        logger.success(f"Base model loaded: {base_model_path}")

        return YOLO(str(base_model_path))

    def train(self) -> None:
        """Train the model.

        Returns
        -------
        None
        """
        # Build training kwargs (includes training_run_name)
        training_kwargs = self._create_kwargs()

        logger.info("Starting training with configuration:")
        for k, v in training_kwargs.items():
            logger.info(f"  {k}: {v}")

        # Train the model (progress is automatically displayed by Ultralytics)
        self.base_model.train(**training_kwargs)

        logger.success("Training completed!")

    def save_to_custom_folder(
        self, best_weights_path: Path, custom_name: str | None = None
    ) -> Path:
        """Save trained model to custom models folder.

        Parameters
        ----------
        best_weights_path : Path
            Path to best.pt from training
        custom_name : str, optional
            Custom name for saved model (uses training name if None)

        Returns
        -------
        Path
            Path to saved model in custom folder
        """
        base_dir = Path(__file__).parent.parent.parent
        custom_dir = base_dir / "models" / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if custom_name is not None:
            dest_path = custom_dir / f"{custom_name}.pt"
        else:
            dest_path = custom_dir / f"{self.training_run_name}.pt"

        # Copy best weights to custom folder
        shutil.copy2(best_weights_path, dest_path)

        logger.info(f"\nModel saved to: {dest_path}")
        return dest_path
