import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime
from ultralytics import YOLO


class PerimeterSurveillance:

    def __init__(
        self,
        source=0,
        model_weight="yolov8n.pt",
        fence_polygon=None,
        db_name="detections.db",
        evidence_dir="evidence",
        confidence_threshold=0.40
    ):
        """
        YOLO-based Perimeter Surveillance System

        Features:
        - Person detection
        - Fence boundary detection
        - Person tracking
        - Intrusion logging
        - Screenshot evidence
        - Entry/exit event detection
        """

        self.source = source
        self.db_name = db_name
        self.evidence_dir = evidence_dir
        self.confidence_threshold = confidence_threshold

        # --------------------------------------------------
        # Create evidence directory
        # --------------------------------------------------
        os.makedirs(self.evidence_dir, exist_ok=True)

        # --------------------------------------------------
        # Open video source
        # --------------------------------------------------
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open video source: {self.source}"
            )

        # --------------------------------------------------
        # Load YOLO model
        # --------------------------------------------------
        print(f"[INFO] Loading YOLO model: {model_weight}")

        try:
            self.model = YOLO(model_weight)
        except Exception as e:
            raise RuntimeError(
                f"YOLO model could not be loaded.\n"
                f"Make sure '{model_weight}' exists or internet is available.\n\n"
                f"Original error: {e}"
            )

        # --------------------------------------------------
        # Fence polygon
        # --------------------------------------------------
        if fence_polygon is None:

            self.fence_polygon = np.array(
                [
                    [150, 100],
                    [650, 100],
                    [650, 450],
                    [150, 450]
                ],
                dtype=np.int32
            )

        else:

            self.fence_polygon = np.array(
                fence_polygon,
                dtype=np.int32
            )

        if len(self.fence_polygon) < 3:
            raise ValueError(
                "Fence polygon must contain at least 3 points."
            )

        # --------------------------------------------------
        # Tracking state
        # --------------------------------------------------

        # Current state of each tracked person
        self.person_states = {}

        # Prevent duplicate intrusion screenshots
        self.last_intrusion_time = {}

        # --------------------------------------------------
        # Database
        # --------------------------------------------------
        self._init_db()

    # ======================================================
    # DATABASE
    # ======================================================

    def _init_db(self):

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trespass_logs (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,

                person_id INTEGER NOT NULL,

                person_label TEXT NOT NULL,

                confidence REAL NOT NULL,

                centroid_x INTEGER NOT NULL,

                centroid_y INTEGER NOT NULL,

                status TEXT NOT NULL,

                evidence_image TEXT

            )
        """)

        conn.commit()
        conn.close()

        print(f"[INFO] Database ready: {self.db_name}")

    # ======================================================
    # DATABASE LOGGING
    # ======================================================

    def _log_to_db(
        self,
        person_id,
        confidence,
        centroid,
        status,
        evidence_image=None
    ):

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        cursor.execute(
            """
            INSERT INTO trespass_logs
            (
                timestamp,
                person_id,
                person_label,
                confidence,
                centroid_x,
                centroid_y,
                status,
                evidence_image
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                current_time,
                int(person_id),
                f"Person_{person_id}",
                float(confidence),
                int(centroid[0]),
                int(centroid[1]),
                status,
                evidence_image
            )
        )

        conn.commit()
        conn.close()

    # ======================================================
    # SAVE EVIDENCE IMAGE
    # ======================================================

    def _save_evidence(self, frame, person_id):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"intrusion_person_{person_id}_"
            f"{timestamp}.jpg"
        )

        filepath = os.path.join(
            self.evidence_dir,
            filename
        )

        success = cv2.imwrite(filepath, frame)

        if success:
            return filepath

        return None

    # ======================================================
    # DRAW FENCE
    # ======================================================

    def _draw_fence(self, frame):

        cv2.polylines(
            frame,
            [self.fence_polygon],
            isClosed=True,
            color=(255, 0, 0),
            thickness=3
        )

        x, y = self.fence_polygon[0]

        cv2.putText(
            frame,
            "PERMITTED ZONE",
            (int(x), max(25, int(y) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2
        )

    # ======================================================
    # RUN SURVEILLANCE
    # ======================================================

    def run(self, output_path=None):

        if not self.cap.isOpened():

            print(
                f"[ERROR] Could not open source: {self.source}"
            )

            return

        # --------------------------------------------------
        # Video properties
        # --------------------------------------------------

        width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25

        print(
            f"[INFO] Video resolution: "
            f"{width}x{height}"
        )

        print(
            f"[INFO] FPS: {fps:.2f}"
        )

        # --------------------------------------------------
        # Validate fence coordinates
        # --------------------------------------------------

        max_x = width - 1
        max_y = height - 1

        if (
            np.any(self.fence_polygon[:, 0] < 0)
            or np.any(self.fence_polygon[:, 0] > max_x)
            or np.any(self.fence_polygon[:, 1] < 0)
            or np.any(self.fence_polygon[:, 1] > max_y)
        ):

            print(
                "[WARNING] Some fence points are outside "
                "the video frame."
            )

        # --------------------------------------------------
        # Output video writer
        # --------------------------------------------------

        writer = None

        if output_path:

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            if not writer.isOpened():

                print(
                    "[WARNING] Output video could not be created."
                )

                writer = None

        print()
        print("=" * 60)
        print(" PERIMETER SURVEILLANCE STARTED")
        print("=" * 60)
        print(" Press Q to stop")
        print("=" * 60)

        # ==================================================
        # FRAME LOOP
        # ==================================================

        while self.cap.isOpened():

            ret, frame = self.cap.read()

            if not ret:
                break

            # ------------------------------------------------
            # Draw fence
            # ------------------------------------------------

            self._draw_fence(frame)

            # ------------------------------------------------
            # YOLO Tracking
            # ------------------------------------------------

            try:

                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=self.confidence_threshold,
                    verbose=False
                )

            except Exception as e:

                print(
                    f"[ERROR] YOLO tracking error: {e}"
                )

                break

            outside_count = 0
            inside_count = 0

            current_ids = set()

            # ------------------------------------------------
            # Process detections
            # ------------------------------------------------

            for result in results:

                if result.boxes is None:
                    continue

                boxes = result.boxes

                for box in boxes:

                    # ----------------------------------------
                    # Confidence
                    # ----------------------------------------

                    try:
                        conf = float(
                            box.conf[0].cpu().item()
                        )

                    except Exception:
                        continue

                    if conf < self.confidence_threshold:
                        continue

                    # ----------------------------------------
                    # Bounding box
                    # ----------------------------------------

                    try:

                        x1, y1, x2, y2 = (
                            box.xyxy[0]
                            .cpu()
                            .numpy()
                            .astype(int)
                        )

                    except Exception:
                        continue

                    # ----------------------------------------
                    # Tracking ID
                    # ----------------------------------------

                    if box.id is not None:

                        try:
                            person_id = int(
                                box.id[0].cpu().item()
                            )

                        except Exception:
                            person_id = 0

                    else:

                        person_id = 0

                    # ----------------------------------------
                    # If tracking ID unavailable
                    # ----------------------------------------

                    if person_id <= 0:

                        person_id = (
                            len(current_ids) + 1
                        )

                    current_ids.add(person_id)

                    # ----------------------------------------
                    # Foot point
                    # ----------------------------------------

                    foot_x = int(
                        (x1 + x2) / 2
                    )

                    foot_y = int(y2)

                    foot_point = (
                        foot_x,
                        foot_y
                    )

                    # ----------------------------------------
                    # Fence check
                    # ----------------------------------------

                    distance = cv2.pointPolygonTest(
                        self.fence_polygon,
                        foot_point,
                        False
                    )

                    is_inside = distance >= 0

                    # ----------------------------------------
                    # Person state
                    # ----------------------------------------

                    previous_state = (
                        self.person_states.get(
                            person_id,
                            None
                        )
                    )

                    # ========================================
                    # INTRUDER
                    # ========================================

                    if not is_inside:

                        outside_count += 1

                        box_color = (
                            0,
                            0,
                            255
                        )

                        label = (
                            f"INTRUDER "
                            f"ID:{person_id} "
                            f"{conf:.2f}"
                        )

                        # ------------------------------------
                        # ENTRY EVENT
                        # ------------------------------------

                        if previous_state != "outside":

                            print(
                                f"[ALERT] Person "
                                f"{person_id} crossed "
                                f"the perimeter!"
                            )

                            evidence_path = (
                                self._save_evidence(
                                    frame,
                                    person_id
                                )
                            )

                            self._log_to_db(
                                person_id=person_id,
                                confidence=conf,
                                centroid=foot_point,
                                status="Intrusion",
                                evidence_image=evidence_path
                            )

                            self.last_intrusion_time[
                                person_id
                            ] = datetime.now()

                        self.person_states[
                            person_id
                        ] = "outside"

                    # ========================================
                    # SAFE
                    # ========================================

                    else:

                        inside_count += 1

                        box_color = (
                            0,
                            255,
                            0
                        )

                        label = (
                            f"SAFE "
                            f"ID:{person_id} "
                            f"{conf:.2f}"
                        )

                        # ------------------------------------
                        # EXIT EVENT
                        # ------------------------------------

                        if previous_state == "outside":

                            print(
                                f"[INFO] Person "
                                f"{person_id} returned "
                                f"inside the permitted zone."
                            )

                        self.person_states[
                            person_id
                        ] = "inside"

                    # ----------------------------------------
                    # Bounding box
                    # ----------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    # ----------------------------------------
                    # Foot point
                    # ----------------------------------------

                    cv2.circle(
                        frame,
                        foot_point,
                        5,
                        (0, 255, 255),
                        -1
                    )

                    # ----------------------------------------
                    # Label
                    # ----------------------------------------

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2
                    )

            # ==================================================
            # HUD
            # ==================================================

            if outside_count > 0:

                alert_text = (
                    "ALERT: PERIMETER BREACH!"
                )

                hud_color = (
                    0,
                    0,
                    255
                )

            else:

                alert_text = (
                    "STATUS: SECURE"
                )

                hud_color = (
                    0,
                    255,
                    0
                )

            # ----------------------------------------------
            # Status
            # ----------------------------------------------

            cv2.putText(
                frame,
                alert_text,
                (20, 35),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                hud_color,
                2
            )

            # ----------------------------------------------
            # Inside count
            # ----------------------------------------------

            cv2.putText(
                frame,
                f"Inside Zone: {inside_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            # ----------------------------------------------
            # Outside count
            # ----------------------------------------------

            cv2.putText(
                frame,
                f"Intruders: {outside_count}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            # ----------------------------------------------
            # Active tracked people
            # ----------------------------------------------

            cv2.putText(
                frame,
                f"Tracked People: {len(current_ids)}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # ==================================================
            # SAVE OUTPUT
            # ==================================================

            if writer is not None:
                writer.write(frame)

            # ==================================================
            # DISPLAY
            # ==================================================

            cv2.imshow(
                "YOLO Perimeter Surveillance",
                frame
            )

            # Press Q
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        # ======================================================
        # CLEANUP
        # ======================================================

        self.cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        print()
        print("=" * 60)
        print(" SURVEILLANCE STOPPED")
        print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Video / Webcam
    # --------------------------------------------------------

    video_input = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test4.mp4")

    # Webcam ke liye:
    # video_input = 0

    # --------------------------------------------------------
    # Fence coordinates
    # IMPORTANT:
    # Coordinates video resolution ke according hone chahiye.
    # --------------------------------------------------------

    custom_fence = [
        [200, 150],
        [700, 150],
        [750, 500],
        [150, 500]
    ]

    # --------------------------------------------------------
    # Create application
    # --------------------------------------------------------

    try:

        app = PerimeterSurveillance(
            source=video_input,
            model_weight="yolov8n.pt",
            fence_polygon=custom_fence,
            db_name="detections.db",
            evidence_dir="evidence",
            confidence_threshold=0.40
        )

        # ----------------------------------------------------
        # Start surveillance
        # ----------------------------------------------------

        app.run(
            output_path="output_detected.mp4"
        )

    except Exception as e:

        print()
        print("=" * 60)
        print(" APPLICATION ERROR")
        print("=" * 60)
        print(e)
        print("=" * 60)
