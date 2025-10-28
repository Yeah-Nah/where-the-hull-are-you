"""
Demo script showing how to use the boat detection and tracking system.
This is a simple example to help you get started.
"""

import os
import sys
from detect_boat import detect_boats
from track_boat import track_boats_in_video


def demo_detection(image_path):
    """
    Demo: Detect boats in a single image.
    """
    print("\n" + "=" * 60)
    print("DEMO: Boat Detection in Image")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Detect boats
    output_path = "output/detected_boats.jpg"
    detections = detect_boats(image_path, output_path, confidence_threshold=0.5)
    
    # Print results
    print(f"\nInput image: {image_path}")
    print(f"Found {len(detections)} boat(s)")
    
    for i, detection in enumerate(detections, 1):
        bbox = detection['bbox']
        conf = detection['confidence']
        print(f"  Boat {i}: confidence={conf:.2f}, bbox={bbox}")
    
    if len(detections) > 0:
        print(f"\nOutput saved to: {output_path}")
    else:
        print("\nNo boats detected in the image.")


def demo_tracking(video_path):
    """
    Demo: Track boats in a video.
    """
    print("\n" + "=" * 60)
    print("DEMO: Boat Tracking in Video")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Track boats
    output_path = "output/tracked_boats.mp4"
    print(f"\nInput video: {video_path}")
    print("This may take a while depending on video length...")
    
    stats = track_boats_in_video(video_path, output_path, confidence_threshold=0.5)
    
    # Print statistics
    print("\n" + "-" * 60)
    print("TRACKING STATISTICS")
    print("-" * 60)
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Frames with boats: {stats['frames_with_boats']}")
    print(f"Total boat detections: {stats['total_detections']}")
    print(f"Unique boats tracked: {stats['unique_boat_ids']}")
    
    if stats['total_frames'] > 0:
        detection_rate = (stats['frames_with_boats'] / stats['total_frames']) * 100
        print(f"Detection rate: {detection_rate:.1f}%")


def main():
    """
    Main demo function with simple menu.
    """
    print("\n" + "=" * 60)
    print("BOAT DETECTION AND TRACKING DEMO")
    print("=" * 60)
    print("\nThis demo shows how to use the boat detection and tracking scripts.")
    print("\nAvailable options:")
    print("  1. Detect boats in an image")
    print("  2. Track boats in a video")
    print("\nNote: On first run, the YOLOv8 model will be downloaded (~6MB)")
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        # Determine file type by extension
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            demo_detection(file_path)
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            demo_tracking(file_path)
        else:
            print(f"\nError: Unsupported file type: {file_path}")
            print("Supported formats:")
            print("  Images: .jpg, .jpeg, .png, .bmp")
            print("  Videos: .mp4, .avi, .mov, .mkv")
    else:
        print("\n" + "=" * 60)
        print("USAGE")
        print("=" * 60)
        print("\nTo run the demo, provide a file path as an argument:")
        print("\n  python demo.py path/to/your/image.jpg")
        print("  python demo.py path/to/your/video.mp4")
        print("\nThe script will automatically detect whether it's an image or video.")
        print("Results will be saved in the 'output/' directory.")
        print("\n" + "=" * 60)
        print("\nExample usage in your own code:")
        print("\n  # For images:")
        print("  from detect_boat import detect_boats")
        print("  detections = detect_boats('boat_image.jpg', 'output.jpg')")
        print("\n  # For videos:")
        print("  from track_boat import track_boats_in_video")
        print("  stats = track_boats_in_video('boat_video.mp4', 'output.mp4')")


if __name__ == "__main__":
    main()
