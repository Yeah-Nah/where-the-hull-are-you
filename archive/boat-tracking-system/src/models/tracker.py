import cv2
import yaml
from pathlib import Path
import tempfile
from loguru import logger


class Tracker:
    """Track objects across video frames using YOLO's built-in tracking."""

    def __init__(self, tracker_type="botsort", config=None):
        """
        Initialize tracker.

        Parameters
        ----------
        tracker_type : str
            Tracking algorithm: 'botsort', 'bytetrack', or 'deepocsort'
        config : dict, optional
            Dictionary of tracker configuration parameters
        """
        self.tracker_type = tracker_type
        self.config = config or {}

        # If config dict provided, create temporary YAML file
        if self.config:
            # Create a temporary YAML file with config
            self.tracker_config = self._create_config_file(self.config)
            logger.info(
                f"Using custom tracker config with {len(self.config)} parameters"
            )
        else:
            # Use default built-in config
            self.tracker_config = f"{tracker_type}.yaml"
            logger.info(f"Using default {tracker_type} config")

    def _create_config_file(self, config_dict):
        """Create temporary YAML config file from dictionary."""
        # Create temp file that persists
        temp_dir = Path(tempfile.gettempdir()) / "boat_tracking"
        temp_dir.mkdir(exist_ok=True)

        config_file = temp_dir / f"{self.tracker_type}_custom.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        return str(config_file)

    def track(self, model, frame, confidence_threshold=0.5, target_class_ids=None):
        """Track objects in frame using YOLO's built-in tracking."""
        results = model.track(
            frame,
            conf=confidence_threshold,
            persist=True,
            tracker=self.tracker_config,
            classes=target_class_ids,
            verbose=False,
        )
        return results

    def draw_tracks(self, frame, results):
        """Draw bounding boxes with track IDs on frame."""
        annotated_frame = frame.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            names = results[0].names

            for box, track_id, conf, cls_id in zip(
                boxes, track_ids, confidences, class_ids
            ):
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                class_name = names[cls_id]
                label = f"ID:{track_id} {class_name} {conf:.2f}"

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        return annotated_frame

    def extract_tracks(self, results):
        """Extract tracking information as a list of dictionaries."""
        tracks = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            names = results[0].names

            for box, track_id, conf, cls_id in zip(
                boxes, track_ids, confidences, class_ids
            ):
                x1, y1, x2, y2 = map(int, box)
                tracks.append(
                    {
                        "track_id": int(track_id),
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": names[cls_id],
                    }
                )

        return tracks
