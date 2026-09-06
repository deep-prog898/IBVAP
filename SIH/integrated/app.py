import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import time
import sqlite3
import threading
from flask import Flask, render_template, Response, jsonify
import config
from video_stream import VideoStream
from yolo_tracker import YOLOTracker
from fence_module import FenceModule
from anpr_module import ANPRModule
from drawer import Drawer

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

import importlib

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

        # 2. Run Fence Logic
        fence_results = fence_module.process_detections(frame, yolo_detections)

        # 3. Run ANPR Logic
        anpr_boxes, last_ocr = anpr_module.process_frame(frame)

        # 4. Calculate FPS
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        # 5. Draw
        annotated_frame = drawer.draw(frame, yolo_detections, fence_results, anpr_boxes, last_ocr, fps)

        # 6. Update global state for the dashboard
        system_state["active_tracks"] = len(yolo_detections)
        system_state["incidents"] = fence_results.get("outside_count", 0) # intruder count
        
        # Check for active intruders right now
        active_intruders = [p for p in fence_results.get("person_results", []) if p["status"] == "inside"]
        if active_intruders:
            system_state["recent_alert"] = {
                "track_id": active_intruders[0]["track_id"],
                "class_name": active_intruders[0].get("class_name", "obj"),
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

@app.route('/api/state')
def get_state():
    # Fetch latest database logs
    try:
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()
        
        # Total historical incidents
        cursor.execute("SELECT COUNT(*) FROM trespass_logs")
        total_incidents = cursor.fetchone()[0]
        
        # Last 10 logs
        cursor.execute("SELECT timestamp, person_label, confidence, status, evidence_image FROM trespass_logs ORDER BY id DESC LIMIT 10")
        logs = []
        for row in cursor.fetchall():
            logs.append({
                "time": row[0],
                "incident": "Perimeter Intrusion",
                "object": row[1],
                "confidence": round(row[2] * 100, 1) if row[2] else 0,
                "status": row[3],
                "evidence": row[4]
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
