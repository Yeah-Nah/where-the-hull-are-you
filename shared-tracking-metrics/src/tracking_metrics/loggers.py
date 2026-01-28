"""MLflow integration for logging tracking metrics."""

from typing import Any


class MLflowMetricsLogger:
    """Log tracking metrics to MLflow."""

    def __init__(self, experiment_name: str, run_name: str | None = None):
        """Initialize MLflow logger.

        Parameters
        ----------
        experiment_name : str
            Name of MLflow experiment
        run_name : Optional[str]
            Name for this run (auto-generated if None)
        """
        self.experiment_name = experiment_name
        self.run_name = run_name

    def start_run(self) -> None:
        """Start a new MLflow run."""
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow.

        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics to log
        step : Optional[int]
            Step number (for time series logging)
        """
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to log (model config, tracker config, etc.)
        """
        pass

    def log_artifact(self, artifact_path: str) -> None:
        """Log artifact (video, plot, etc.) to MLflow.

        Parameters
        ----------
        artifact_path : str
            Path to artifact file
        """
        pass

    def end_run(self) -> None:
        """End the current MLflow run."""
        pass
