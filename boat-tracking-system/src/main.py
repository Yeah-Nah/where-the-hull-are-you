# Contents of /boat-tracking-system/boat-tracking-system/src/main.py

import os
import yaml
from config.settings import Config
from processors.batch_processor import BatchProcessor
from utils.logger import Logger

def main():
    logger = Logger()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize the batch processor
    batch_processor = BatchProcessor(config['input_folder'], config['output_folder'], logger)
    
    # Process videos
    logger.info("Starting video processing...")
    batch_processor.process_videos()
    logger.info("Video processing completed.")

if __name__ == "__main__":
    main()