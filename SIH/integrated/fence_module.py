import cv2
import sqlite3
import os
from datetime import datetime

class FenceModule:
    def __init__(self, fence_polygon, db_name, evidence_dir):
        self.fence_polygon = fence_polygon
        self.db_name = db_name
        self.evidence_dir = evidence_dir
        
        os.makedirs(self.evidence_dir, exist_ok=True)
        # To ensure the directory containing the db exists
        os.makedirs(os.path.dirname(self.db_name), exist_ok=True)
        
        self.person_states = {}
        self._init_db()

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

    def _log_to_db(self, person_id, confidence, centroid, status, evidence_image=None):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO trespass_logs
            (timestamp, person_id, person_label, confidence, centroid_x, centroid_y, status, evidence_image)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (current_time, int(person_id), f"Person_{person_id}", float(confidence), 
             int(centroid[0]), int(centroid[1]), status, evidence_image)
        )
        conn.commit()
        conn.close()

    def _save_evidence(self, frame, person_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intrusion_person_{person_id}_{timestamp}.jpg"
        filepath = os.path.join(self.evidence_dir, filename)
        success = cv2.imwrite(filepath, frame)
        return filepath if success else None

    def process_detections(self, frame, detections):
        outside_count = 0
        inside_count = 0
        
        results = []

        for det in detections:
            track_id = det["track_id"]
            if track_id == -1:
                continue

            x1, y1, x2, y2 = det["bbox"]
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            foot_point = (foot_x, foot_y)

            distance = cv2.pointPolygonTest(self.fence_polygon, foot_point, False)
            is_inside = distance >= 0
            
            previous_state = self.person_states.get(track_id, None)

            if is_inside:
                outside_count += 1
                status = "inside"
                if previous_state != "inside":
                    print(f"[ALERT] {det['class_name'].upper()} {track_id} entered the RESTRICTED ZONE!")
                    evidence_path = self._save_evidence(frame, track_id)
                    self._log_to_db(track_id, det["confidence"], foot_point, "Intrusion", evidence_path)
            else:
                inside_count += 1
                status = "outside"
                if previous_state == "inside":
                    print(f"[INFO] {det['class_name'].upper()} {track_id} left the restricted zone.")

            self.person_states[track_id] = status
            
            results.append({
                "track_id": track_id,
                "class_name": det["class_name"],
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "status": status,
                "foot_point": foot_point
            })

        return {
            "outside_count": outside_count,
            "inside_count": inside_count,
            "person_results": results
        }
