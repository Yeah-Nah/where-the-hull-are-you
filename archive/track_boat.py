"""
Boat tracking script using YOLOv8 with tracking.
This script tracks boats across video frames.
"""

from ultralytics import YOLO
import cv2
import os


def track_boats_in_video(video_path, output_path=None, confidence_threshold=0.5, show_video=False):
    """
    Track boats in a video using YOLOv8 with built-in tracking.
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path to save output video with tracking (optional)
        confidence_threshold (float): Minimum confidence score for detections (0-1)
        show_video (bool): Whether to display video while processing
    
    Returns:
        dict: Summary statistics including total frames, boats detected, etc.
    """
    # Load pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Tracking statistics
    stats = {
        'total_frames': 0,
        'frames_with_boats': 0,
        'total_detections': 0,
        'unique_boat_ids': set()
    }
    
    print("Processing video...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run tracking on the frame
        # persist=True enables tracking between frames
        results = model.track(frame, conf=confidence_threshold, persist=True, classes=[8])
        
        # Process results
        boats_in_frame = 0
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == 8:  # Boat class
                    boats_in_frame += 1
                    # Track ID if available
                    if box.id is not None:
                        track_id = int(box.id[0])
                        stats['unique_boat_ids'].add(track_id)
        
        # Update statistics
        stats['total_frames'] += 1
        if boats_in_frame > 0:
            stats['frames_with_boats'] += 1
            stats['total_detections'] += boats_in_frame
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Write frame to output video
        if out is not None:
            out.write(annotated_frame)
        
        # Display frame if requested
        if show_video:
            cv2.imshow('Boat Tracking', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user")
                break
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
        print(f"\nOutput video saved to: {output_path}")
    if show_video:
        cv2.destroyAllWindows()
    
    # Convert set to count for final stats
    stats['unique_boat_ids'] = len(stats['unique_boat_ids'])
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Boat Tracking Script")
    print("=" * 50)
    print("This script uses YOLOv8 to track boats in videos.")
    print("\nUsage:")
    print("  from track_boat import track_boats_in_video")
    print("  stats = track_boats_in_video('path/to/video.mp4', 'output.mp4')")
    print("\nFor your first run, place a video file in the current directory")
    print("and modify the example below.")
