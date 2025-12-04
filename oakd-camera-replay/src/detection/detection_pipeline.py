from pathlib import Path
from src.processors.camera_feed_proccessor import camera_detection_pipeline

class detectionPipeline():
    """Runs required detection pipeline on given input feed. Either video reply or camera feed."""

    def __init__(self, input_video: str = None, use_camera: bool = False, model_path: Path = None):
        """Initiate detection pipeline.

        Args:
            input_video (str, optional): Path to video to process. Defaults to None.
            use_camera (bool, optional): Whether to use camera input. Defaults to False.
            model_path (Path, optional): Path to the model file. Defaults to None.
        """
        self.input_video = input_video
        self.use_camera = use_camera
        self.model_path = model_path

    def run_detection(self):
        """Run the required detection pipeline."""
        
        if self.use_camera == True:
            
            camera_detection_pipeline(
                use_camera=self.use_camera,
                model_path=self.model_path
            )
        
        elif self.input_video is not None:
            print("Placeholder for now")

        else:
            raise ValueError("Either input_video must be provided or use_camera must be True.")