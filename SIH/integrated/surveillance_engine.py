import os
import sys
import time
import sqlite3
from datetime import datetime
import cv2
import numpy as np
from collections import deque

# Ensure integrated directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
from yolo_tracker import YOLOTracker
from fence_module import FenceModule
from drawer import Drawer
from video_stream import VideoStream

# Safe imports for Streamlit caching if available
try:
    import streamlit as st
    cache_resource_decorator = st.cache_resource
except Exception:
    def cache_resource_decorator(func):
        return func

# Directory configurations
BASE_DIR = config.BASE_DIR
EVIDENCE_DIR = config.EVIDENCE_DIR
DB_PATH = config.DB_NAME
UPLOAD_DIR = os.path.join(CURRENT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------------------------

@cache_resource_decorator
def load_yolo_tracker(model_path=None, confidence=0.40):
    """Load and cache the YOLOv11 tracker model to prevent reload on UI reruns."""
    if model_path is None:
        model_path = config.YOLO_MODEL_PATH
    if not os.path.exists(model_path):
        # Fallback to local file in project root if relative path discrepancy
        alt_path = os.path.join(BASE_DIR, "yolo11n.pt")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"YOLO model not found at {model_path} or {alt_path}")
    print(f"[ENGINE] Loading YOLOTracker from: {model_path} (conf={confidence})")
    return YOLOTracker(model_path=model_path, confidence=confidence)


@cache_resource_decorator
def load_anpr_module(model_path=None, confidence=0.25, ocr_interval=10):
    """
    Safely load and cache the ANPR YOLO + PaddleOCR module.
    Returns: (anpr_module_instance, status_message_or_warning)
    Does NOT crash if the ANPR model or PaddleOCR is unavailable.
    """
    if model_path is None:
        model_path = config.ANPR_MODEL_PATH

    if not os.path.exists(model_path):
        warning_msg = (
            f"ANPR weights file not found at '{model_path}'. "
            "License Plate recognition will run in simulation or bypass mode."
        )
        print(f"[ENGINE WARN] {warning_msg}")
        return None, warning_msg

    try:
        from anpr_module import ANPRModule
        print(f"[ENGINE] Initializing ANPRModule from: {model_path}")
        anpr = ANPRModule(model_path=model_path, confidence=confidence, ocr_interval=ocr_interval)
        return anpr, "ANPR and PaddleOCR initialized successfully."
    except Exception as e:
        warning_msg = f"Failed to initialize ANPR / PaddleOCR ({str(e)}). Plate recognition disabled."
        print(f"[ENGINE ERROR] {warning_msg}")
        return None, warning_msg


# ---------------------------------------------------------------------------
# Reusable Surveillance Engine Coordinator
# ---------------------------------------------------------------------------

class SurveillanceEngine:
    """
    Unified AI surveillance coordinator managing YOLO tracking, digital fence
    intrusion detection, ANPR license plate extraction, overlay rendering,
    SQLite logging, and snapshot captures.
    """
    def __init__(self, camera_label="CAM-01", yolo_tracker=None, anpr_module=None):
        self.camera_label = camera_label
        self.yolo_tracker = yolo_tracker or load_yolo_tracker()
        self.anpr_module = anpr_module  # May be None if ANPR disabled or missing

        self.custom_polygon = config.FENCE_POLYGON.copy()
        self.fence_module = FenceModule(self.custom_polygon, DB_PATH, EVIDENCE_DIR)
        self.drawer = Drawer(self.custom_polygon)

        self.tracked_vehicle_plates = {}
        self.target_logged_cooldown = {}
        self.recent_alerts = deque(maxlen=30)
        self.session_counts = {
            "persons": 0,
            "vehicles": 0,
            "active_tracks": 0,
            "plates": 0,
            "intrusions": 0
        }

    def reset_state(self):
        """Reset transient tracking & state caches (e.g., when switching video/camera)."""
        self.tracked_vehicle_plates.clear()
        self.target_logged_cooldown.clear()
        self.fence_module.person_states.clear()
        self.recent_alerts.clear()

    def process_frame(self, frame, toggles=None, fps=30.0, save_evidence_db=True, is_live=False):
        """
        Process a single video frame with toggles for granular AI module activation:
        toggles = {
            'human': True/False,
            'vehicle': True/False,
            'tracking': True/False,
            'anpr': True/False,
            'ocr': True/False,
            'fence': True/False,
            'intrusion': True/False
        }
        """
        if frame is None:
            return None, {}

        if toggles is None:
            toggles = {
                "human": True,
                "vehicle": True,
                "tracking": True,
                "anpr": True,
                "ocr": True,
                "fence": True,
                "intrusion": True
            }

        h, w = frame.shape[:2]

        # 1. Adapt Fence Polygon to frame aspect ratio
        active_polygon = np.array([
            [int(w * 0.15), int(h * 0.18)],
            [int(w * 0.85), int(h * 0.18)],
            [int(w * 0.90), int(h * 0.90)],
            [int(w * 0.10), int(h * 0.90)]
        ], dtype=np.int32)
        self.fence_module.fence_polygon = active_polygon
        self.drawer.fence_polygon = active_polygon

        # 2. YOLO Object Detection & Tracking
        raw_detections = self.yolo_tracker.track(frame)

        filtered_detections = []
        person_count = 0
        vehicle_count = 0
        active_track_ids = set()

        for det in raw_detections:
            c_name = det.get("class_name", "").lower()
            is_person = (c_name == "person")
            is_vehicle = (c_name in ["car", "motorcycle", "bus", "truck", "bicycle"])

            if is_person:
                if not toggles.get("human", True):
                    continue
                person_count += 1
            elif is_vehicle:
                if not toggles.get("vehicle", True):
                    continue
                vehicle_count += 1
            else:
                # Discard non-target classes (e.g. dog, backpack, etc.)
                continue

            det_copy = dict(det)
            if not toggles.get("tracking", True):
                det_copy["track_id"] = -1
            else:
                if det_copy["track_id"] != -1:
                    active_track_ids.add(det_copy["track_id"])

            filtered_detections.append(det_copy)

        # 3. ANPR Plate Detection & OCR
        anpr_boxes = []
        last_ocr = []
        if toggles.get("anpr", True) and self.anpr_module is not None:
            try:
                anpr_boxes, last_ocr = self.anpr_module.process_frame(frame)
                if not toggles.get("ocr", True):
                    # Suppress OCR text recognition
                    for p in anpr_boxes:
                        p["text"] = None
                    last_ocr = []
            except Exception as e:
                print(f"[ENGINE ANPR ERROR] {e}")

        # 4. Associate detected plates with tracked vehicles
        for p_box in anpr_boxes:
            px1, py1, px2, py2 = p_box["bbox"]
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            p_text = p_box.get("text")
            p_conf = p_box.get("confidence", 0.0)
            if not p_text and last_ocr:
                p_text = last_ocr[0].get("text")
                p_conf = last_ocr[0].get("confidence", 0.0)

            for det in filtered_detections:
                if det.get("class_name", "").lower() in ["car", "motorcycle", "bus", "truck"]:
                    vx1, vy1, vx2, vy2 = det["bbox"]
                    if (vx1 - 15) <= pcx <= (vx2 + 15) and (vy1 - 15) <= pcy <= (vy2 + 15):
                        vid = det["track_id"]
                        if vid != -1 and p_text:
                            self.tracked_vehicle_plates[vid] = {
                                "plate": p_text,
                                "confidence": p_conf,
                                "last_seen": time.time()
                            }
                        break

        # 5. Digital Fence & Intrusion Breaches
        if toggles.get("fence", True):
            fence_results = self.fence_module.process_detections(
                frame,
                filtered_detections,
                vehicle_plates=self.tracked_vehicle_plates,
                camera_label=self.camera_label
            )
            # If intrusion alerts are toggled OFF, mask breaches
            if not toggles.get("intrusion", True):
                fence_results["outside_count"] = 0
                for item in fence_results.get("person_results", []):
                    item["status"] = "outside"
        else:
            # Fence disabled: all detected targets marked safe/outside
            fence_results = {
                "outside_count": 0,
                "inside_count": len(filtered_detections),
                "person_results": [
                    {
                        "track_id": d["track_id"],
                        "class_name": d["class_name"],
                        "display_label": f"{d['class_name'].upper()} #{d['track_id']}",
                        "bbox": d["bbox"],
                        "confidence": d["confidence"],
                        "status": "outside",
                        "foot_point": (int((d["bbox"][0] + d["bbox"][2]) / 2), d["bbox"][3])
                    }
                    for d in filtered_detections
                ]
            }

        # 6. Real-time Telemetry & Incident Logging to SQLite
        now = time.time()
        new_alerts = []

        if save_evidence_db:
            for det in filtered_detections:
                tid = det["track_id"]
                if tid == -1:
                    continue
                vclass = det.get("class_name", "obj").lower()
                conf = det.get("confidence", 0.0)

                last_logged = self.target_logged_cooldown.get(tid, 0)
                # Log on first appearance, then cooldown of 12 seconds per active track
                if (now - last_logged) > 12:
                    plate_info = self.tracked_vehicle_plates.get(tid)
                    has_plate = plate_info and plate_info.get("plate")
                    is_intruder = any(
                        p["track_id"] == tid and p["status"] == "inside"
                        for p in fence_results.get("person_results", [])
                    )

                    if is_live:
                        inc_type = "Live Camera Detection"
                        obj_label = f"[{self.camera_label}] {vclass.upper()} #{tid} (Live Cam)"
                    elif has_plate:
                        inc_type = "ANPR Detection"
                        obj_label = f"[{self.camera_label}] {vclass.upper()} #{tid} [{plate_info['plate']}]"
                        conf = plate_info.get("confidence", conf)
                    elif is_intruder:
                        inc_type = "Perimeter Intrusion"
                        obj_label = f"[{self.camera_label}] {vclass.upper()} #{tid}"
                    else:
                        inc_type = "Surveillance Telemetry"
                        obj_label = f"[{self.camera_label}] {vclass.upper()} #{tid}"

                    self.target_logged_cooldown[tid] = now
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    time_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    snap_filename = f"{self.camera_label}_{vclass}_{tid}_{timestamp_str}.jpg"
                    snap_path = os.path.join(EVIDENCE_DIR, snap_filename)
                    cv2.imwrite(snap_path, frame)

                    vx1, vy1, vx2, vy2 = det["bbox"]
                    cx, cy = int((vx1 + vx2) / 2), int(vy2)

                    try:
                        self.fence_module._log_to_db(
                            person_id=tid,
                            label=obj_label,
                            confidence=conf,
                            centroid=(cx, cy),
                            status="VALID",
                            evidence_image=snap_path,
                            incident_type=inc_type
                        )
                    except Exception as e:
                        print(f"[ENGINE DB ERROR] {e}")

                    alert_entry = {
                        "id": int(time.time() * 1000) % 10000000,
                        "time": time_display,
                        "camera": self.camera_label,
                        "event": inc_type,
                        "object": obj_label,
                        "confidence": round(conf * 100, 1),
                        "status": "VALID",
                        "evidence_path": snap_path,
                        "evidence_filename": snap_filename
                    }
                    self.recent_alerts.appendleft(alert_entry)
                    new_alerts.append(alert_entry)

        # 7. Render Overlays (bounding boxes, polygons, ANPR text, HUD)
        annotated_frame = self.drawer.draw(
            frame.copy(),
            filtered_detections,
            fence_results,
            anpr_boxes,
            last_ocr,
            fps
        )

        # Add Camera Watermark
        cv2.putText(
            annotated_frame,
            self.camera_label,
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (59, 130, 246),
            2,
            cv2.LINE_AA
        )

        # Telemetry metrics package
        metrics = {
            "persons": person_count,
            "vehicles": vehicle_count,
            "active_tracks": len(active_track_ids),
            "anpr_plates": len(anpr_boxes),
            "intrusions": fence_results.get("outside_count", 0),
            "fps": round(fps, 1),
            "camera": self.camera_label,
            "new_alerts": new_alerts,
            "last_ocr": last_ocr,
            "anpr_boxes": anpr_boxes
        }

        return annotated_frame, metrics


# ---------------------------------------------------------------------------
# Database Incident Log Queries & Operations
# ---------------------------------------------------------------------------

def fetch_incident_logs(db_path=None, limit=50, camera_filter=None, status_filter=None):
    """
    Safely query the trespass_logs table in detections.db.
    Returns a list of incident dictionaries.
    """
    if db_path is None:
        db_path = DB_PATH

    if not os.path.exists(db_path):
        return []

    incidents = []
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()

        # Check table columns
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trespass_logs'")
        if not cursor.fetchone():
            conn.close()
            return []

        cursor.execute("PRAGMA table_info(trespass_logs)")
        cols = [c[1] for c in cursor.fetchall()]
        has_incident_type = "incident_type" in cols

        query = (
            "SELECT id, timestamp, person_label, confidence, status, evidence_image, incident_type "
            "FROM trespass_logs ORDER BY id DESC LIMIT ?"
            if has_incident_type else
            "SELECT id, timestamp, person_label, confidence, status, evidence_image, 'Perimeter Intrusion' "
            "FROM trespass_logs ORDER BY id DESC LIMIT ?"
        )
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            label_str = str(row[2])
            cam_tag = "CAM-01"
            if "[CAM-02]" in label_str:
                cam_tag = "CAM-02"
            elif "[CAM-01]" in label_str:
                cam_tag = "CAM-01"

            raw_status = str(row[4]).upper() if row[4] else "VALID"
            ui_status = "FALSE_ALARM" if "FALSE" in raw_status else "VALID"

            # Camera filter check
            if camera_filter and camera_filter != "ALL" and cam_tag != camera_filter:
                continue

            # Status filter check
            if status_filter and status_filter != "ALL" and ui_status != status_filter:
                continue

            ev_path = row[5]
            ev_filename = os.path.basename(ev_path) if ev_path else None

            incidents.append({
                "id": row[0],
                "time": row[1],
                "camera": cam_tag,
                "object": label_str,
                "confidence": round(row[3] * 100, 1) if row[3] else 0.0,
                "status": ui_status,
                "evidence_path": ev_path,
                "evidence_filename": ev_filename,
                "incident_type": row[6] if row[6] else "Perimeter Intrusion"
            })
    except Exception as e:
        print(f"[ENGINE DB FETCH ERROR] {e}")

    return incidents


def update_incident_status(incident_id, new_status="FALSE_ALARM", db_path=None):
    """Update status of a specific log entry in detections.db."""
    if db_path is None:
        db_path = DB_PATH
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        cursor.execute("UPDATE trespass_logs SET status = ? WHERE id = ?", (new_status, incident_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ENGINE DB UPDATE ERROR] {e}")
        return False


def get_incident_summary(db_path=None):
    """Return total count and breakdown from detections.db."""
    if db_path is None:
        db_path = DB_PATH

    summary = {"total": 0, "valid": 0, "false_alarms": 0}
    if not os.path.exists(db_path):
        return summary

    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trespass_logs'")
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(*) FROM trespass_logs")
            summary["total"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trespass_logs WHERE status LIKE '%FALSE%'")
            summary["false_alarms"] = cursor.fetchone()[0]

            summary["valid"] = summary["total"] - summary["false_alarms"]
        conn.close()
    except Exception as e:
        print(f"[ENGINE DB SUMMARY ERROR] {e}")
    return summary


# ---------------------------------------------------------------------------
# Evidence & Media Helpers
# ---------------------------------------------------------------------------

def list_sample_videos():
    """Discover available sample clips in project root."""
    samples = []
    if os.path.exists(BASE_DIR):
        for fname in sorted(os.listdir(BASE_DIR)):
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                fpath = os.path.join(BASE_DIR, fname)
                if os.path.isfile(fpath):
                    size_mb = os.path.getsize(fpath) / (1024 * 1024)
                    samples.append({
                        "filename": fname,
                        "path": fpath,
                        "size_mb": round(size_mb, 2)
                    })
    return samples


def list_evidence_images(limit=30):
    """Retrieve list of recent evidence captures."""
    evidence_files = []
    if os.path.exists(EVIDENCE_DIR):
        for fname in sorted(os.listdir(EVIDENCE_DIR), reverse=True):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(EVIDENCE_DIR, fname)
                if os.path.isfile(fpath):
                    cam_tag = "CAM-01" if "CAM-01" in fname else ("CAM-02" if "CAM-02" in fname else "CAM-MAIN")
                    evidence_files.append({
                        "filename": fname,
                        "path": fpath,
                        "camera": cam_tag,
                        "mtime": datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
                    })
            if len(evidence_files) >= limit:
                break
    return evidence_files


def save_uploaded_file(uploaded_file):
    """Save user-uploaded video to integrated/uploads directory."""
    if uploaded_file is None:
        return None
    safe_name = f"{int(time.time())}_{uploaded_file.name.replace(' ', '_')}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path
