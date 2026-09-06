import os

os.environ["FLAGS_use_onednn"] = "0"

from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2


# Load YOLO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt"))

# Load PaddleOCR
ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)

# Open video
video_file = os.path.join(BASE_DIR, "test4.mp4")
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished")
        break

    frame_count += 1

    # YOLO detection
    results = model(frame, conf=0.25, verbose=False)

    # Draw YOLO results
    annotated_frame = results[0].plot()

    # Run OCR only every 10th frame
    if frame_count % 10 == 0:

        for box in results[0].boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Keep coordinates inside image
            h, w = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Crop plate
            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            # Resize plate
            plate = cv2.resize(
                plate,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            # Improve contrast while keeping 3-channel image
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # Convert back to 3-channel BGR for PaddleOCR
            plate = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # OCR
            try:

                result = ocr.predict(plate)

                for res in result:

                    # PaddleOCR 3.x result
                    data = res.json

                    if isinstance(data, str):
                        import json
                        data = json.loads(data)

                    ocr_data = data.get("res", {})

                    texts = ocr_data.get("rec_texts", [])
                    scores = ocr_data.get("rec_scores", [])

                    for i in range(len(texts)):

                        text = texts[i]

                        if i < len(scores):
                            score = scores[i]
                        else:
                            score = 0.0

                        print(
                            f"Plate: {text} | Confidence: {score:.2f}"
                        )

            except Exception as e:

                print("OCR error:", e)

    # Display video
    cv2.imshow("ANPR", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
