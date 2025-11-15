from src.processors.video_processor import VideoProcessor
from src.processors.batch_processor import BatchProcessor

def test_video_processor():
    # Initialize the VideoProcessor with a sample video file
    video_processor = VideoProcessor(input_path='data/input/sample_video.mp4', output_path='data/output/processed_video.mp4')
    
    # Process the video
    result = video_processor.process()
    
    # Check if the output file was created
    assert result is True
    assert os.path.exists('data/output/processed_video.mp4')

def test_batch_processor():
    # Initialize the BatchProcessor with input and output directories
    batch_processor = BatchProcessor(input_dir='data/input', output_dir='data/output')
    
    # Process all videos in the input directory
    results = batch_processor.process_all()
    
    # Check if the results contain processed files
    assert isinstance(results, list)
    assert len(results) > 0
    for output_file in results:
        assert os.path.exists(output_file)