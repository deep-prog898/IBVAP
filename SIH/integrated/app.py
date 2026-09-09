import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import time
from datetime import datetime
import sqlite3
import threading
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import config
from video_stream import VideoStream
from yolo_tracker import YOLOTracker
from fence_module import FenceModule
from anpr_module import ANPRModule
from drawer import Drawer
import importlib

app = Flask(__name__)

# Global variables to share data between the AI thread and the web server
latest_frame_encoded = None
system_state = {
    "active_tracks": 0,
    "incidents": 0,
    "system_health": "ONLINE",
    "model": "yolo11n.pt",
    "recent_alert": None
}

def processing_thread():
    global latest_frame_encoded, system_state
    
    # Initialize Video
    current_video_path = config.VIDEO_PATH
    video = VideoStream(current_video_path, camera_id="MAIN-CAM")
    if not video.connect():
        print(f"[ERROR] Cannot connect to video: {current_video_path}")
        return

    # Use default polygon from config for the web version (headless)
    custom_polygon = config.FENCE_POLYGON

    # Initialize Modules
    yolo_tracker = YOLOTracker(config.YOLO_MODEL_PATH, confidence=config.YOLO_CONFIDENCE)
    fence_module = FenceModule(custom_polygon, config.DB_NAME, config.EVIDENCE_DIR)
    anpr_module = ANPRModule(config.ANPR_MODEL_PATH, confidence=config.ANPR_CONFIDENCE, ocr_interval=config.OCR_INTERVAL)
    drawer = Drawer(custom_polygon)

    # Persistent caches for tracking plates and preventing log spam
    tracked_vehicle_plates = {}
    anpr_logged_cooldown = {}

    frame_counter = 0

    while True:
        start_time = time.time()
        frame_counter += 1
        
        # Check every 20 frames if the video was changed in config.py
        if frame_counter % 20 == 0:
            try:
                importlib.reload(config)
                if os.path.normpath(config.VIDEO_PATH) != os.path.normpath(current_video_path):
                    print(f"\n[HOT-RELOAD] Switched video to: {config.VIDEO_PATH}")
                    video.release()
                    current_video_path = config.VIDEO_PATH
                    video = VideoStream(current_video_path, camera_id="MAIN-CAM")
                    video.connect()
            except Exception as e:
                print(f"[HOT-RELOAD ERROR]: {e}")
        
        ret, frame = video.read_frame()
        if not ret:
            # Check if video was changed before looping
            try:
                importlib.reload(config)
                if os.path.normpath(config.VIDEO_PATH) != os.path.normpath(current_video_path):
                    print(f"\n[HOT-RELOAD] Switched video to: {config.VIDEO_PATH}")
                    video.release()
                    current_video_path = config.VIDEO_PATH
                    video = VideoStream(current_video_path, camera_id="MAIN-CAM")
                    video.connect()
                    continue
            except Exception:
                pass
            # Loop the video for continuous web dashboard testing
            if video.cap:
                video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.01)
            continue

        # 1. Run Unified YOLO Tracker
        yolo_detections = yolo_tracker.track(frame)

        # 2. Run ANPR Logic
        anpr_boxes, last_ocr = anpr_module.process_frame(frame)

        # 3. Associate ANPR plate boxes with vehicle detections
        for p_box in anpr_boxes:
            px1, py1, px2, py2 = p_box["bbox"]
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            p_text = p_box.get("text")
            p_conf = p_box.get("confidence", 0.0)

            # Fallback to last_ocr if available
            if not p_text and last_ocr:
                p_text = last_ocr[0].get("text")
                p_conf = last_ocr[0].get("confidence", 0.0)

            for det in yolo_detections:
                if det.get("class_name", "").lower() in ["car", "motorcycle", "bus", "truck"]:
                    vx1, vy1, vx2, vy2 = det["bbox"]
                    # Check if plate center is within vehicle bbox (with slight padding)
                    if (vx1 - 15) <= pcx <= (vx2 + 15) and (vy1 - 15) <= pcy <= (vy2 + 15):
                        vid = det["track_id"]
                        if vid != -1 and p_text:
                            tracked_vehicle_plates[vid] = {
                                "plate": p_text,
                                "confidence": p_conf,
                                "last_seen": time.time()
                            }
                        break

        # 4. Run Fence Logic with tracked vehicle plates
        fence_results = fence_module.process_detections(frame, yolo_detections, vehicle_plates=tracked_vehicle_plates)

        # 5. Log Passing Vehicles with Recognized Plates (even outside fence)
        for det in yolo_detections:
            vid = det["track_id"]
            vclass = det.get("class_name", "").lower()
            if vid != -1 and vclass in ["car", "motorcycle", "bus", "truck"] and vid in tracked_vehicle_plates:
                plate_info = tracked_vehicle_plates[vid]
                plate_str = plate_info.get("plate")
                if plate_str:
                    cooldown_key = f"{vid}_{plate_str}"
                    now = time.time()
                    last_logged = anpr_logged_cooldown.get(cooldown_key, 0)
                    if (now - last_logged) > 25:
                        # Check if already inside restricted zone (fence_module logs those as intrusions)
                        is_intruder = any(p["track_id"] == vid and p["status"] == "inside" for p in fence_results.get("person_results", []))
                        if not is_intruder:
                            anpr_logged_cooldown[cooldown_key] = now
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            snap_filename = f"anpr_{vclass}_{vid}_{timestamp_str}.jpg"
                            snap_path = os.path.join(config.EVIDENCE_DIR, snap_filename)
                            cv2.imwrite(snap_path, frame)
                            
                            obj_label = f"{vclass.upper()} #{vid} [{plate_str}]"
                            vx1, vy1, vx2, vy2 = det["bbox"]
                            cx, cy = int((vx1 + vx2) / 2), int((vy1 + vy2) / 2)
                            fence_module._log_to_db(
                                person_id=vid,
                                label=obj_label,
                                confidence=plate_info.get("confidence", det["confidence"]),
                                centroid=(cx, cy),
                                status="VALID",
                                evidence_image=snap_path,
                                incident_type="ANPR Detection"
                            )
                            print(f"[ANPR LOG] Recorded passing vehicle: {obj_label}")

        # 6. Calculate FPS
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        # 7. Draw HUD & Overlays
        annotated_frame = drawer.draw(frame, yolo_detections, fence_results, anpr_boxes, last_ocr, fps)

        # 8. Update global state for the dashboard
        system_state["active_tracks"] = len(yolo_detections)
        system_state["incidents"] = fence_results.get("outside_count", 0) # intruder count
        
        # Check for active intruders right now
        active_intruders = [p for p in fence_results.get("person_results", []) if p["status"] == "inside"]
        if active_intruders:
            system_state["recent_alert"] = {
                "track_id": active_intruders[0]["track_id"],
                "class_name": active_intruders[0].get("display_label", active_intruders[0].get("class_name", "OBJ")),
                "confidence": active_intruders[0]["confidence"]
            }
        else:
            system_state["recent_alert"] = None

        # Encode frame for web streaming with optimized JPEG quality
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ret:
            latest_frame_encoded = buffer.tobytes()

def generate_frames():
    global latest_frame_encoded
    while True:
        if latest_frame_encoded is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_encoded + b'\r\n')
        time.sleep(0.03) # Smooth ~30 fps stream

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/evidence/<path:filename>')
def get_evidence(filename):
    """Serve captured intrusion and ANPR snapshot images directly to the web dashboard."""
    return send_from_directory(config.EVIDENCE_DIR, filename)

@app.route('/api/incidents/<int:incident_id>/status', methods=['POST'])
def update_incident_status(incident_id):
    """Update incident verification status (e.g. Mark False / Mark Valid)."""
    data = request.get_json() or {}
    new_status = data.get('status', 'FALSE_ALARM')
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        cursor = conn.cursor()
        cursor.execute("UPDATE trespass_logs SET status = ? WHERE id = ?", (new_status, incident_id))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "id": incident_id, "status": new_status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/state')
def get_state():
    # Fetch latest database logs
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        cursor = conn.cursor()
        
        # Total historical incidents
        cursor.execute("SELECT COUNT(*) FROM trespass_logs")
        total_incidents = cursor.fetchone()[0]
        
        # Check if incident_type column exists
        cursor.execute("PRAGMA table_info(trespass_logs)")
        cols = [c[1] for c in cursor.fetchall()]
        has_incident_type = "incident_type" in cols

        query = (
            "SELECT id, timestamp, person_label, confidence, status, evidence_image, incident_type "
            "FROM trespass_logs ORDER BY id DESC LIMIT 15"
            if has_incident_type else
            "SELECT id, timestamp, person_label, confidence, status, evidence_image, 'Perimeter Intrusion' "
            "FROM trespass_logs ORDER BY id DESC LIMIT 15"
        )
        cursor.execute(query)
        logs = []
        for row in cursor.fetchall():
            ev_path = row[5]
            ev_filename = os.path.basename(ev_path) if ev_path else None
            ev_url = f"/evidence/{ev_filename}" if ev_filename else None

            raw_status = str(row[4]).upper() if row[4] else "VALID"
            ui_status = "FALSE_ALARM" if "FALSE" in raw_status else "VALID"

            logs.append({
                "id": row[0],
                "time": row[1],
                "object": row[2],
                "confidence": round(row[3] * 100, 1) if row[3] else 0,
                "status": ui_status,
                "raw_status": row[4],
                "evidence_url": ev_url,
                "evidence_filename": ev_filename,
                "incident": row[6] if row[6] else "Perimeter Intrusion"
            })
        conn.close()
    except Exception as e:
        total_incidents = 0
        logs = []

    state = system_state.copy()
    state["total_incidents"] = total_incidents
    state["logs"] = logs
    return jsonify(state)

if __name__ == '__main__':
    # Ensure evidence dir exists
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    
    # Start AI processing in a background thread
    t = threading.Thread(target=processing_thread, daemon=True)
    t.start()
    
    # Start Web Server
    print("\n" + "="*50)
    print("🚀 DASHBOARD LIVE: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

