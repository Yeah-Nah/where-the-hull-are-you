"""
Basic boat detection script using YOLOv8.
This script detects boats in a single image or frame.
"""

from ultralytics import YOLO
import cv2


def detect_boats(image_path, output_path=None, confidence_threshold=0.5):
    """
    Detect boats in an image using YOLOv8.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output image with detections (optional)
        confidence_threshold (float): Minimum confidence score for detections (0-1)
    
    Returns:
        list: List of detection dictionaries with bbox, confidence, and class info
    """
    # Load pre-trained YOLOv8 model (will download on first run)
    model = YOLO('yolov8n.pt')  # 'n' is nano - smallest and fastest model
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Run inference
    # Class 8 in COCO dataset is 'boat'
    results = model(image, conf=confidence_threshold, classes=[8])
    
    # Extract boat detections
    boat_detections = []
    for result in results:
        boxes = result.boxes
        # Check if any detections were found
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Filter for boats (class 8 in COCO dataset)
                if class_id == 8:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boat_detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': 'boat'
                    })
    
    # Save annotated image if output path provided
    if output_path and len(boat_detections) > 0:
        annotated_image = results[0].plot()
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to: {output_path}")
    
    return boat_detections


if __name__ == "__main__":
    # Example usage
    print("Boat Detection Script")
    print("=" * 50)
    print("This script uses YOLOv8 to detect boats in images.")
    print("\nUsage:")
    print("  from detect_boat import detect_boats")
    print("  detections = detect_boats('path/to/image.jpg', 'output.jpg')")
    print("\nFor your first run, place an image in the current directory")
    print("and modify the example below.")
