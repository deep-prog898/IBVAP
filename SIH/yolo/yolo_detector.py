from ultralytics import YOLO


class YOLODetector:

    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def track(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=320,
            classes=[0, 2, 3, 5, 7],
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                class_name = self.model.names[class_id]

                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -1

                detections.append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    )
                })

        return detections
