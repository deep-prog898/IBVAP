import time
import cv2

from camera_manager import CameraManager
from yolo_detector import YOLODetector


import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "test2.mp4")


# -----------------------------
# Camera
# -----------------------------

camera = CameraManager(
    source=VIDEO_PATH,
    camera_id="CAM-03"
)

if not camera.connect():
    exit()


# -----------------------------
# YOLO + ByteTrack
# -----------------------------

detector = YOLODetector("yolo11n.pt")


# -----------------------------
# Video FPS
# -----------------------------

fps = camera.cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

print(f"Source FPS: {fps}")


# -----------------------------
# Main Loop
# -----------------------------

while True:

    start_time = time.time()

    success, frame = camera.read_frame()

    if not success:
        print("Video ended.")
        break


    # -------------------------
    # YOLO + ByteTrack
    # -------------------------

    detections = detector.track(frame)


    # -------------------------
    # Draw detections
    # -------------------------

    for detection in detections:

        class_name = detection["class_name"]
        confidence = detection["confidence"]
        track_id = detection["track_id"]

        x1, y1, x2, y2 = detection["bbox"]


        # Bounding box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ID + class + confidence

        label = f"ID:{track_id} {class_name} {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # -------------------------
    # Calculate AI FPS
    # -------------------------

    processing_time = time.time() - start_time

    if processing_time > 0:
        current_fps = 1 / processing_time
    else:
        current_fps = 0


    # -------------------------
    # Camera Status
    # -------------------------

    cv2.putText(
        frame,
        "CAM-03 | ONLINE | AI TRACKING",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # -------------------------
    # AI FPS
    # -------------------------

    cv2.putText(
        frame,
        f"AI FPS: {current_fps:.1f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # -------------------------
    # Display
    # -------------------------

    cv2.imshow(
        "IBVAP - AI Video Analytics",
        frame
    )


    # Press Q to exit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

camera.release()

cv2.destroyAllWindows()