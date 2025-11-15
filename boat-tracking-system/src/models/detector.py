from ultralytics import YOLO
class Detector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0].cpu().numpy())
                confidence_score = box.conf[0].cpu().numpy()
                if confidence_score >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence_score),
                        'class_id': class_id,
                        'class_name': self.model.names[class_id]
                    })
        return detections

    def load_model(self):
        return self.model

    def get_classes(self):
        return self.model.names