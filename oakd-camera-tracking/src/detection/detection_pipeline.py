#!/usr/bin/env python3
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from src.config.model_settings import CLASS_IDS, COCO_CLASSES, CONFIDENCE_THRESHOLD
from src.processors.camera_feed_proccessor import CameraFeedProcessor
from src.processors.video_replay_processor import VideoFeedProcessor


class DetectionPipeline:
    """Runs required detection pipeline on given input feed. Either video replay or camera feed."""

    def __init__(
        self, input_video: str = None, use_camera: bool = False, model_path: Path = None
    ):
        """Initiate detection pipeline.

        Args:
            input_video (str, optional): Path to video to process. Defaults to None.
            use_camera (bool, optional): Whether to use camera input. Defaults to False.
            model_path (Path, optional): Path to the model file. Defaults to None.
        """
        self.input_video = input_video
        self.use_camera = use_camera
        self.model_path = model_path
        self.pipeline = None
        self.video_queue = None
        self.nn_queue = None

    def run_detection(self):
        """Run the required detection pipeline."""
        # Get pipeline and queues from appropriate processor
        if self.use_camera:
            processor = CameraFeedProcessor()
            self.pipeline, cam_out, self.video_queue = processor.camera_feed_connect()

        elif self.input_video is not None:
            processor = VideoFeedProcessor(video_path=self.input_video)
            self.pipeline, cam_out, self.video_queue = processor.video_feed_connect()
        else:
            raise ValueError(
                "Either input_video must be provided or use_camera must be True."
            )

        # Add neural network to the pipeline
        nn = self.pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(self.model_path))

        # Link input to neural network
        cam_out.link(nn.input)
        self.nn_queue = nn.out.createOutputQueue()

        # Start pipeline
        self.pipeline.start()

        # FPS calculation
        prev_time = time.time()

        try:
            while self.pipeline.isRunning():
                # Get video frame
                video_frame = self.video_queue.tryGet()
                in_nn_data = self.nn_queue.tryGet()

                if video_frame is not None and in_nn_data is not None:
                    # Calculate FPS
                    current_time = time.time()
                    fps = (
                        1 / (current_time - prev_time)
                        if (current_time - prev_time) > 0
                        else 0
                    )
                    prev_time = current_time

                    assert isinstance(video_frame, dai.ImgFrame)
                    frame = video_frame.getCvFrame()
                    frame_height, frame_width = frame.shape[:2]

                    tensor = in_nn_data.getFirstTensor()
                    assert isinstance(tensor, np.ndarray)

                    # Reshape tensor to (1, 84, 4032)
                    tensor = tensor.reshape(1, 84, 4032)

                    # Process detections
                    valid_detections = self.process_detections(tensor)

                    # Draw results
                    frame = self.draw_detections(
                        frame, valid_detections, frame_width, frame_height, fps
                    )

                    # Show frame
                    cv2.imshow("Object Detection", frame)

                    print(f"Valid detections: {len(valid_detections)} | FPS: {fps:.1f}")

                if cv2.waitKey(1) == ord("q"):
                    break

        finally:
            cv2.destroyAllWindows()

    def process_detections(self, tensor: np.ndarray) -> list:
        """
        Process neural network output tensor and filter detections.

        Args:
            tensor: Neural network output tensor of shape (1, 84, 4032)

        Returns
        -------
            List of valid detections
        """
        valid_detections = []
        for i in range(4032):  # For each grid cell
            # Get bounding box and class scores for this grid cell
            bbox = tensor[0, :4, i]
            class_scores = tensor[0, 4:, i]

            # Find max confidence and corresponding class
            max_confidence = np.max(class_scores)
            predicted_class = np.argmax(class_scores)

            # Filter by confidence threshold and class IDs
            if max_confidence > CONFIDENCE_THRESHOLD and predicted_class in CLASS_IDS:
                valid_detections.append(
                    {
                        "grid_cell": i,
                        "class": predicted_class,
                        "confidence": max_confidence,
                        "bbox": bbox,
                    }
                )

        return valid_detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list,
        frame_width: int,
        frame_height: int,
        fps: float,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: Input frame
            detections: List of detection dictionaries
            frame_width: Frame width
            frame_height: Frame height
            fps: Current FPS

        Returns
        -------
            Frame with drawn detections
        """
        for det in detections:
            # YOLOv8 outputs are in absolute pixel coordinates
            x_center, y_center, width, height = det["bbox"]

            # Convert center coordinates to top-left corner
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label with class name and confidence
            class_name = COCO_CLASSES.get(det["class"], f"Class {det['class']}")
            label = f"{class_name}: {det['confidence']:.2f}"

            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1,
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        # Display detection count and FPS
        info_text = f"Detections: {len(detections)} | FPS: {fps:.1f}"
        cv2.putText(
            frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        return frame
