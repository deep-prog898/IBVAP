import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# Set SIH and integrated paths
SIH_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATED_DIR = os.path.join(SIH_DIR, "integrated")
if SIH_DIR not in sys.path:
    sys.path.insert(0, SIH_DIR)
if INTEGRATED_DIR not in sys.path:
    sys.path.insert(0, INTEGRATED_DIR)

import config
import surveillance_engine
from surveillance_engine import (
    load_yolo_tracker,
    load_anpr_module,
    SurveillanceEngine,
    VideoStream,
    fetch_incident_logs,
    update_incident_status,
    get_incident_summary,
    list_sample_videos,
    list_evidence_images,
    save_uploaded_file,
    BASE_DIR,
    EVIDENCE_DIR,
    DB_PATH
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IBVAP — Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for Sleek Defense / Dark Surveillance Aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Surveillance Base */
    .stApp {
        background-color: #0c1322;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Top Header Bar */
    .ibvap-header {
        background: linear-gradient(135deg, #111a2e 0%, #172033 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .ibvap-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .ibvap-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 2px;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .status-online {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .status-alert {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.5);
        animation: pulseRed 1.8s infinite;
    }
    
    @keyframes pulseRed {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background: #141c2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 2px;
    }
    
    /* Standby Frame */
    .standby-box {
        background: #0f172a;
        border: 2px dashed #1e293b;
        border-radius: 12px;
        height: 480px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: #64748b;
    }
    
    /* Table & Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111a2e;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #60a5fa !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "is_paused" not in st.session_state:
    st.session_state.is_paused = False
if "camera_select" not in st.session_state:
    st.session_state.camera_select = "CAM-01"
if "live_alerts" not in st.session_state:
    st.session_state.live_alerts = []
if "live_anpr_records" not in st.session_state:
    st.session_state.live_anpr_records = []
if "latest_metrics" not in st.session_state:
    st.session_state.latest_metrics = {
        "persons": 0, "vehicles": 0, "active_tracks": 0, "anpr_plates": 0, "intrusions": 0, "fps": 0.0
    }
if "last_annotated_frame" not in st.session_state:
    st.session_state.last_annotated_frame = None
if "webcam_start_time" not in st.session_state:
    st.session_state.webcam_start_time = None

# ---------------------------------------------------------------------------
# Cached Model Initialization (Shared across runs)
# ---------------------------------------------------------------------------
try:
    cached_yolo = load_yolo_tracker(config.YOLO_MODEL_PATH, confidence=0.40)
    yolo_loaded = True
    yolo_status_msg = "YOLOv11 Object Tracker loaded."
except Exception as e:
    cached_yolo = None
    yolo_loaded = False
    yolo_status_msg = f"YOLO Load Warning: {e}"

cached_anpr, anpr_status_msg = load_anpr_module(config.ANPR_MODEL_PATH, confidence=0.25, ocr_interval=10)
anpr_loaded = (cached_anpr is not None)

# Per-camera engine registry
if "cam_engines" not in st.session_state:
    st.session_state.cam_engines = {
        "CAM-01": SurveillanceEngine("CAM-01", yolo_tracker=cached_yolo, anpr_module=cached_anpr),
        "CAM-02": SurveillanceEngine("CAM-02", yolo_tracker=cached_yolo, anpr_module=cached_anpr)
    }

# ---------------------------------------------------------------------------
# Sidebar UI: Controls, Sources & AI Modules
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ IBVAP Surveillance")
    st.caption("AI-Based Border Surveillance Matrix")
    
    # 1. Camera Selection
    st.markdown("---")
    st.markdown("#### 📹 Camera Selection")
    active_cam = st.radio(
        "Active Camera Channel",
        options=["CAM-01", "CAM-02"],
        index=0 if st.session_state.camera_select == "CAM-01" else 1,
        format_func=lambda c: f"{c} • Sector {'A (Primary)' if c == 'CAM-01' else 'B (Secondary)'}",
        help="Switch between dual surveillance feeds"
    )
    st.session_state.camera_select = active_cam
    current_engine = st.session_state.cam_engines[active_cam]

    # 2. Video Source
    st.markdown("---")
    st.markdown("#### 📺 Video Source")
    source_type = st.selectbox(
        "Source Type",
        options=["Sample Video", "Upload Video", "Webcam (Live Camera)", "RTSP Camera"],
        index=0
    )

    resolved_source = None
    is_live_stream = False

    if source_type == "Sample Video":
        samples = list_sample_videos()
        if samples:
            sample_names = [s["filename"] for s in samples]
            default_idx = sample_names.index("test6.mp4") if "test6.mp4" in sample_names else 0
            chosen_sample = st.selectbox("Select Sample Clip", options=sample_names, index=default_idx)
            resolved_source = os.path.join(BASE_DIR, chosen_sample)
            selected_info = next((s for s in samples if s["filename"] == chosen_sample), None)
            if selected_info:
                st.caption(f"📁 Size: `{selected_info['size_mb']} MB` | Status: Ready")
        else:
            st.warning("No sample video files found in project root.")
            resolved_source = os.path.join(BASE_DIR, "test6.mp4")

    elif source_type == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload Surveillance Footage",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Supports standard formats up to 500MB"
        )
        if uploaded_file is not None:
            save_path = save_uploaded_file(uploaded_file)
            resolved_source = save_path
            st.success(f"Uploaded: `{uploaded_file.name}`")
        else:
            st.info("Upload a video to begin analysis.")

    elif source_type == "Webcam (Live Camera)":
        live_mode = st.radio(
            "Live Camera Mode",
            options=["Continuous Stream (Webcam)", "Browser Camera Snapshot"],
            help="Continuous stream runs directly via OpenCV VideoStream. Browser snapshot works in any browser and on Streamlit Cloud."
        )
        if live_mode == "Continuous Stream (Webcam)":
            device_idx = st.number_input("Camera Device Index", min_value=0, max_value=4, value=0, step=1, help="Default internal/USB camera is index 0")
            resolved_source = int(device_idx)
            is_live_stream = True
            st.info("⏱️ Live Prototype Session: 60-second limit. Uses OpenCV with DirectShow on Windows, with automatic simulated fallback if physical camera is busy.")
        else:
            resolved_source = "browser_snapshot"
            is_live_stream = True
            st.info("📷 Browser Snapshot Mode: Use your browser camera directly below to capture and analyze frames in real-time.")

    elif source_type == "RTSP Camera":
        rtsp_url = st.text_input("RTSP Stream URL", placeholder="rtsp://admin:pass@192.168.1.100:554/h264")
        if rtsp_url:
            resolved_source = rtsp_url.strip()
            is_live_stream = True
        st.caption("ℹ️ RTSP streams require direct network route / edge gateway access.")

    # 3. AI Modules Toggles
    st.markdown("---")
    st.markdown("#### 🧠 AI Analytics Layer")
    enable_human = st.checkbox("👤 Human Detection", value=True)
    enable_vehicle = st.checkbox("🚗 Vehicle Detection", value=True)
    enable_tracking = st.checkbox("🎯 ByteTrack Multi-Target", value=True)
    
    # ANPR & OCR toggles with disabled check if weights missing
    anpr_disabled = not anpr_loaded
    enable_anpr = st.checkbox(
        "🔢 ANPR Plate Detection",
        value=anpr_loaded,
        disabled=anpr_disabled,
        help="Requires runs/detect/train/weights/best.pt" if anpr_disabled else "Detect vehicle license plates"
    )
    enable_ocr = st.checkbox(
        "🔤 PaddleOCR Recognition",
        value=anpr_loaded,
        disabled=anpr_disabled or not enable_anpr,
        help="Extract license plate characters"
    )
    if anpr_disabled:
        st.caption("⚠️ ANPR model offline (best.pt not present or OCR uninitialized).")

    enable_fence = st.checkbox("🚧 Digital Fence Boundary", value=True)
    enable_intrusion = st.checkbox("⚠️ Intrusion Detection", value=True)

    ai_toggles = {
        "human": enable_human,
        "vehicle": enable_vehicle,
        "tracking": enable_tracking,
        "anpr": enable_anpr and anpr_loaded,
        "ocr": enable_ocr and anpr_loaded,
        "fence": enable_fence,
        "intrusion": enable_intrusion
    }

    # 4. Performance & Tuning
    st.markdown("---")
    with st.expander("⚙️ Pipeline Performance & Tuning", expanded=False):
        yolo_conf = st.slider("YOLO Confidence", min_value=0.15, max_value=0.85, value=0.40, step=0.05)
        current_engine.yolo_tracker.confidence = yolo_conf
        
        if current_engine.anpr_module:
            anpr_conf = st.slider("ANPR Confidence", min_value=0.10, max_value=0.70, value=0.25, step=0.05)
            current_engine.anpr_module.confidence = anpr_conf
            
        frame_stride = st.select_slider("Frame Processing Stride", options=[1, 2, 3], value=1, help="Skip frames to maximize CPU throughput")
        loop_playback = st.checkbox("Loop Video Footage", value=True)

# ---------------------------------------------------------------------------
# Main Dashboard Header
# ---------------------------------------------------------------------------
active_breach = st.session_state.latest_metrics["intrusions"] > 0
header_status_class = "status-alert" if active_breach else "status-online"
header_status_text = "🚨 PERIMETER BREACH ACTIVE" if active_breach else "● SYSTEM ONLINE"

st.markdown(f"""
<div class="ibvap-header">
    <div>
        <div class="ibvap-title">
            <span>🛡️ IBVAP</span>
            <span style="font-size: 14px; font-weight: 600; color: #60a5fa; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.4); padding: 3px 10px; border-radius: 6px;">
                COMMAND CENTER
            </span>
            <span style="font-size: 13px; font-weight: 700; color: #e2e8f0; background: #1e293b; padding: 3px 10px; border-radius: 6px;">
                CHANNEL: {active_cam}
            </span>
        </div>
        <div class="ibvap-subtitle">
            AI-Based Intelligent Video Analytics Platform for Border Surveillance
        </div>
    </div>
    <div class="status-badge {header_status_class}">
        {header_status_text}
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top KPI Metric Cards
# ---------------------------------------------------------------------------
m = st.session_state.latest_metrics
kpi_cols = st.columns(6)

with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid #3b82f6;">
        <div class="kpi-label">Active Channel</div>
        <div class="kpi-value" style="color: #60a5fa;">{active_cam}</div>
        <div class="kpi-sub">Sector {'A (Primary)' if active_cam == 'CAM-01' else 'B (Secondary)'}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid #10b981;">
        <div class="kpi-label">Persons Detected</div>
        <div class="kpi-value">{m['persons']}</div>
        <div class="kpi-sub">In Current Field of View</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid #06b6d4;">
        <div class="kpi-label">Vehicles Detected</div>
        <div class="kpi-value">{m['vehicles']}</div>
        <div class="kpi-sub">Cars, Trucks, Cycles</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid #8b5cf6;">
        <div class="kpi-label">Active Tracks</div>
        <div class="kpi-value">{m['active_tracks']}</div>
        <div class="kpi-sub">ByteTrack Trajectories</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[4]:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
        <div class="kpi-label">ANPR Plates</div>
        <div class="kpi-value" style="color: #fbbf24;">{m['anpr_plates']}</div>
        <div class="kpi-sub">License Plates Read</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[5]:
    breach_color = "#ef4444" if m['intrusions'] > 0 else "#64748b"
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid {breach_color};">
        <div class="kpi-label">Intrusion Alerts</div>
        <div class="kpi-value" style="color: {breach_color};">{m['intrusions']}</div>
        <div class="kpi-sub">Restricted Zone Breaches</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Video Viewport & Playback Toolbar
# ---------------------------------------------------------------------------
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5 = st.columns([1.5, 1.2, 1.2, 1.5, 2.6])

with ctrl_c1:
    is_source_ready = (resolved_source is not None) and (not isinstance(resolved_source, str) or resolved_source.strip() != "")
    is_browser_snap = (resolved_source == "browser_snapshot")
    start_label = "▶ Start Surveillance" if not st.session_state.is_running else "▶ Running..."
    start_btn = st.button(
        start_label,
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_running or not is_source_ready or is_browser_snap
    )

with ctrl_c2:
    pause_label = "▶ Resume" if st.session_state.is_paused else "⏸ Pause"
    pause_btn = st.button(pause_label, use_container_width=True, disabled=not st.session_state.is_running)

with ctrl_c3:
    stop_btn = st.button("⏹ Stop Feed", use_container_width=True, disabled=not st.session_state.is_running)

with ctrl_c4:
    reset_state_btn = st.button("🔄 Reset Alerts", use_container_width=True)
    if reset_state_btn:
        current_engine.reset_state()
        st.session_state.live_alerts.clear()
        st.session_state.live_anpr_records.clear()
        st.session_state.latest_metrics = {"persons": 0, "vehicles": 0, "active_tracks": 0, "anpr_plates": 0, "intrusions": 0, "fps": 0.0}
        st.rerun()

with ctrl_c5:
    status_str = f"Pipeline FPS: **{m['fps']}** | Channel: **{active_cam}**"
    if is_live_stream and st.session_state.webcam_start_time:
        rem_sec = max(0, int(60 - (time.time() - st.session_state.webcam_start_time)))
        status_str += f" | ⏱️ **{rem_sec}s** Left"
    st.markdown(f"<div style='text-align: right; padding-top: 8px; color: #94a3b8; font-size: 13px;'>{status_str}</div>", unsafe_allow_html=True)

# State transitions
if start_btn:
    st.session_state.is_running = True
    st.session_state.is_paused = False
    if is_live_stream:
        st.session_state.webcam_start_time = time.time()
    st.rerun()

if pause_btn:
    st.session_state.is_paused = not st.session_state.is_paused
    st.rerun()

if stop_btn:
    st.session_state.is_running = False
    st.session_state.is_paused = False
    st.session_state.webcam_start_time = None
    st.rerun()

# ---------------------------------------------------------------------------
# Video Stream Processing Loop
# ---------------------------------------------------------------------------
video_box = st.empty()

if resolved_source == "browser_snapshot":
    st.markdown("##### 📷 Live Browser Camera Input")
    st.caption("Capture a live frame using your connected webcam or mobile/laptop camera:")
    camera_pic = st.camera_input("Capture Live Surveillance Snapshot")
    if camera_pic is not None:
        bytes_data = camera_pic.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            annotated, metrics = current_engine.process_frame(
                frame=frame,
                toggles=ai_toggles,
                fps=30.0,
                save_evidence_db=True,
                is_live=True
            )
            st.session_state.latest_metrics = metrics
            st.session_state.last_annotated_frame = annotated
            if metrics.get("new_alerts"):
                for a in metrics["new_alerts"]:
                    st.session_state.live_alerts.insert(0, a)
                    if a.get("event") == "ANPR Detection":
                        st.session_state.live_anpr_records.insert(0, a)
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_box.image(rgb_frame, use_container_width=True)

elif st.session_state.is_running and resolved_source is not None:
    # Open Video Source using VideoStream coordinator
    stream = VideoStream(resolved_source, camera_id=active_cam)
    if not stream.connect():
        st.error(f"Failed to connect to video source: `{resolved_source}`. Please verify camera device or path.")
        st.session_state.is_running = False
    else:
        frame_idx = 0
        t_start = time.time()
        
        while st.session_state.is_running:
            if st.session_state.is_paused:
                time.sleep(0.1)
                break

            # 60-Second Live Camera window check
            if is_live_stream and st.session_state.webcam_start_time:
                elapsed = time.time() - st.session_state.webcam_start_time
                if elapsed >= 60:
                    st.warning("⏱️ 1-Minute Live Camera prototype session completed. Stream paused.")
                    st.session_state.is_running = False
                    st.session_state.webcam_start_time = None
                    break

            ret, frame = stream.read_frame()
            if not ret or frame is None:
                if is_live_stream:
                    # Retry for camera warm-up or temporary glitch
                    retried = False
                    for _ in range(5):
                        time.sleep(0.04)
                        ret, frame = stream.read_frame()
                        if ret and frame is not None:
                            retried = True
                            break
                    if not retried:
                        st.session_state.is_running = False
                        break
                else:
                    if loop_playback and stream.cap:
                        stream.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = stream.read_frame()
                        if not ret or frame is None:
                            st.session_state.is_running = False
                            break
                    else:
                        st.session_state.is_running = False
                        break

            frame_idx += 1
            if frame_stride > 1 and (frame_idx % frame_stride != 0):
                continue

            fps_calc = 1.0 / (time.time() - t_start) if (time.time() - t_start) > 0 else 30.0
            t_start = time.time()

            # Process frame with reusable surveillance engine
            annotated, metrics = current_engine.process_frame(
                frame=frame,
                toggles=ai_toggles,
                fps=fps_calc,
                save_evidence_db=True,
                is_live=is_live_stream
            )

            # Overlay countdown on frame for live camera
            if is_live_stream and st.session_state.webcam_start_time:
                rem_sec = max(0, int(60 - (time.time() - st.session_state.webcam_start_time)))
                mins, secs = divmod(rem_sec, 60)
                timer_str = f"LIVE CAM: {mins:02d}:{secs:02d}"
                box_color = (0, 0, 220) if rem_sec <= 10 else (0, 140, 255)
                cv2.rectangle(annotated, (annotated.shape[1] - 240, 12), (annotated.shape[1] - 15, 44), (0, 0, 0), -1)
                cv2.rectangle(annotated, (annotated.shape[1] - 240, 12), (annotated.shape[1] - 15, 44), box_color, 2)
                cv2.putText(annotated, timer_str, (annotated.shape[1] - 225, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

            # Accumulate alerts
            if metrics.get("new_alerts"):
                for a in metrics["new_alerts"]:
                    st.session_state.live_alerts.insert(0, a)
                    if a.get("event") == "ANPR Detection":
                        st.session_state.live_anpr_records.insert(0, a)

            st.session_state.latest_metrics = metrics
            st.session_state.last_annotated_frame = annotated

            # Render in Streamlit viewport
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_box.image(rgb_frame, use_container_width=True)

            # Cooperative yield for smooth UI interactions
            time.sleep(0.015)
            
        stream.release()

elif st.session_state.last_annotated_frame is not None:
    # Render last captured frame when paused
    rgb_frame = cv2.cvtColor(st.session_state.last_annotated_frame, cv2.COLOR_BGR2RGB)
    video_box.image(rgb_frame, use_container_width=True)
else:
    # Sleek standby surveillance card
    video_box.markdown(f"""
    <div class="standby-box">
        <div style="font-size: 40px; margin-bottom: 12px;">🛡️</div>
        <div style="font-size: 18px; font-weight: 700; color: #cbd5e1;">CHANNEL {active_cam} • SURVEILLANCE STANDBY</div>
        <div style="font-size: 13px; color: #64748b; margin-top: 6px;">Select a video source or start live camera session above</div>
        <div style="margin-top: 20px; font-size: 11px; color: #475569; font-family: monospace; letter-spacing: 1px;">
            YOLOv11n TRACKER • BYTETRACK • ANPR PADDLEOCR • PERIMETER FENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Lower Analytical Panels: 5 Tactical Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "🚨 Recent Alerts",
    "🚗 ANPR & Plates",
    "📋 Incident Logs (SQLite)",
    "🖼️ Evidence Vault",
    "⚙️ Telemetry & Cloud Diagnostics"
])

# Tab 1: Recent Alerts
with tabs[0]:
    st.markdown("#### Real-time Threat Matrix & Breach Alerts")
    alerts_data = st.session_state.live_alerts
    if alerts_data:
        df_alerts = pd.DataFrame(alerts_data[:20])
        # Clean columns for display
        disp_cols = [c for c in ["time", "camera", "event", "object", "confidence", "status"] if c in df_alerts.columns]
        st.dataframe(
            df_alerts[disp_cols].rename(columns={
                "time": "Timestamp",
                "camera": "Channel",
                "event": "Event Type",
                "object": "Target Object",
                "confidence": "Confidence %",
                "status": "Audit Status"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No active breach alerts recorded in current session. Perimeter secure.")

# Tab 2: ANPR & Plates
with tabs[1]:
    st.markdown("#### Automatic Number Plate Recognition (ANPR)")
    if not anpr_loaded:
        st.warning(f"ANPR Module Status: {anpr_status_msg}")
    
    anpr_records = st.session_state.live_anpr_records
    if anpr_records:
        df_anpr = pd.DataFrame(anpr_records[:20])
        disp_cols = [c for c in ["time", "camera", "object", "confidence"] if c in df_anpr.columns]
        st.dataframe(
            df_anpr[disp_cols].rename(columns={
                "time": "Timestamp",
                "camera": "Channel",
                "object": "License Plate & Vehicle ID",
                "confidence": "OCR Confidence %"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No license plates recognized in the active stream yet. Plates will appear here when detected.")

# Tab 3: Incident Logs (SQLite Audit)
with tabs[2]:
    st.markdown("#### Permanent Security Audit Trail (`detections.db`)")
    
    summary = get_incident_summary()
    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns([1.5, 1.5, 1.5, 2.5])
    with stat_c1:
        st.metric("Total Recorded Incidents", summary["total"])
    with stat_c2:
        st.metric("Validated Intrusions", summary["valid"])
    with stat_c3:
        st.metric("Flagged False Alarms", summary["false_alarms"])
    with stat_c4:
        db_cam_filter = st.selectbox("Filter Channel", options=["ALL", "CAM-01", "CAM-02"], index=0)

    db_incidents = fetch_incident_logs(limit=50, camera_filter=db_cam_filter)
    if db_incidents:
        df_db = pd.DataFrame(db_incidents)
        display_df = df_db[["id", "time", "camera", "incident_type", "object", "confidence", "status"]].rename(columns={
            "id": "ID",
            "time": "Timestamp",
            "camera": "Camera",
            "incident_type": "Incident Type",
            "object": "Target Object",
            "confidence": "Confidence %",
            "status": "Audit Status"
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        col_act1, col_act2 = st.columns([2, 1])
        with col_act1:
            inc_to_toggle = st.number_input("Target Incident ID to Mark", min_value=1, step=1, value=db_incidents[0]["id"] if db_incidents else 1)
        with col_act2:
            new_stat = st.selectbox("New Audit Status", ["FALSE_ALARM", "VALID"])
            if st.button("Update Incident Status"):
                if update_incident_status(inc_to_toggle, new_stat):
                    st.success(f"Incident #{inc_to_toggle} updated to {new_stat}")
                    st.rerun()
                else:
                    st.error("Failed to update incident in database.")
    else:
        st.info("No records found in SQLite database `detections.db`.")

# Tab 4: Evidence Vault
with tabs[3]:
    st.markdown("#### Evidence Vault & Intrusion Snapshots (`integrated/evidence/`)")
    evidence_items = list_evidence_images(limit=18)
    if evidence_items:
        grid_cols = st.columns(3)
        for idx, item in enumerate(evidence_items):
            col = grid_cols[idx % 3]
            with col:
                st.image(item["path"], caption=f"{item['camera']} • {item['mtime']}", use_container_width=True)
                with open(item["path"], "rb") as file:
                    st.download_button(
                        label=f"⬇️ Download {item['filename'][:18]}...",
                        data=file,
                        file_name=item["filename"],
                        mime="image/jpeg",
                        key=f"dl_{item['filename']}_{idx}"
                    )
    else:
        st.info("Evidence directory `integrated/evidence/` has no captures yet.")

# Tab 5: Telemetry & Cloud Diagnostics
with tabs[4]:
    st.markdown("#### System Telemetry & Streamlit Community Cloud Readiness")
    diag_c1, diag_c2 = st.columns(2)
    
    with diag_c1:
        st.markdown("##### 📦 AI Models & Weights Status")
        st.write(f"- **YOLOv11 Tracker**: `{'🟢 Online' if yolo_loaded else '🔴 Missing'}` (`{config.YOLO_MODEL_PATH}`)")
        st.write(f"- **ANPR License Plate Detector**: `{'🟢 Online' if anpr_loaded else '🟡 Offline / Simulated'}` (`{config.ANPR_MODEL_PATH}`)")
        st.write(f"- **PaddleOCR Engine**: `{'🟢 Initialized' if anpr_loaded else '🟡 Disabled/Bypassed'}`")
        st.write(f"- **SQLite Incident Database**: `🟢 Connected` (`{config.DB_NAME}`)")
        st.write(f"- **Evidence Storage**: `🟢 Accessible` (`{config.EVIDENCE_DIR}`)")

    with diag_c2:
        st.markdown("##### ☁️ Cloud Deployment Specs")
        st.write(f"- **Python Runtime**: `{sys.version.split()[0]}`")
        st.write(f"- **OpenCV Version**: `{cv2.__version__}`")
        st.write(f"- **Streamlit Version**: `{st.__version__}`")
        st.write("- **Cloud Compatibility**: Relative paths active, non-blocking model fallbacks, headless OpenCV configured.")
        st.write("- **Edge / Private Sources**: RTSP and physical webcams clearly labeled for local or edge deployment.")