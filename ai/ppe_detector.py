from ultralytics import YOLO
import cv2

# Class ID constants — must match training exactly
CLASS_HELMET    = 0
CLASS_NO_HELMET = 1
CLASS_VEST      = 2
CLASS_NO_VEST   = 3

# Violation classes — anything in this set is a PPE violation
VIOLATION_CLASSES = {CLASS_NO_HELMET, CLASS_NO_VEST}

CONFIDENCE_THRESHOLD = 0.4


def analyze_detections(results):
    """
    Given raw YOLO results, return simple compliance and violation tags.
    This helper operates directly on class IDs.
    """
    violations = []
    has_helmet = False
    has_vest = False

    if not results:
        return True, violations

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        if cls_id == CLASS_HELMET:
            has_helmet = True
        elif cls_id == CLASS_NO_HELMET:
            violations.append("no_helmet")
        elif cls_id == CLASS_VEST:
            has_vest = True
        elif cls_id == CLASS_NO_VEST:
            violations.append("no_vest")

    is_compliant = len(violations) == 0
    return is_compliant, violations


class PPEDetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        print(f"[PPEDetector] Model loaded → {model_path}")

    def detect(self, frame):
        """Runs detection and returns list of detected objects (ID-based)."""
        results    = self.model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                bbox       = box.xyxy[0].tolist()
                class_name = (
                    self.model.names[class_id]
                    if isinstance(self.model.names, dict) and class_id in self.model.names
                    else str(class_id)
                )

                detections.append({
                    "class_id":    class_id,
                    "class_name":  class_name,
                    "confidence":  round(confidence, 2),
                    "bbox":        [int(b) for b in bbox],
                    "violation":   class_id in VIOLATION_CLASSES,
                })

        return detections

    def check_ppe_compliance(self, detections):
        """
        Checks helmet and vest presence using class IDs.
        Returns compliance dict used by main_ai.py.
        """
        has_helmet = False
        has_vest   = False

        for det in detections:
            cid = det["class_id"]
            if cid == CLASS_HELMET:
                has_helmet = True
            elif cid == CLASS_VEST:
                has_vest = True

        missing = []
        if not has_helmet:
            missing.append("Helmet")
        if not has_vest:
            missing.append("Safety Vest")

        return {
            "has_helmet": has_helmet,
            "has_vest":   has_vest,
            "missing":    missing
        }

    def draw_boxes(self, frame, detections):
        """Draws detection boxes on frame, coloring violations by class ID."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f'{det["class_name"]} {det["confidence"]}'

            is_violation = det.get("violation", False)
            color = (0, 0, 255) if is_violation else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

        return frame