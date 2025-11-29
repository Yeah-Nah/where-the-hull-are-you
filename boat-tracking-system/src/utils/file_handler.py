from pathlib import Path
import cv2


def read_video_files(input_folder):
    """
    Reads all video files from the specified input folder.

    Args:
        input_folder (str): Path to the input folder containing video files.

    Returns:
        list: A list of paths to the video files.
    """
    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    input_path = Path(input_folder)
    return [file for file in input_path.glob("*") if file.suffix in video_extensions]


def save_output_video(output_path, frame, fourcc, fps, width, height):
    """
    Saves the processed video frame to the specified output path.

    Args:
        output_path (str): Path to save the output video.
        frame: The frame to be saved.
        fourcc: Codec for the video writer.
        fps (int): Frames per second of the video.
        width (int): Width of the video frame.
        height (int): Height of the video frame.
    """
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    out.write(frame)
    out.release()
