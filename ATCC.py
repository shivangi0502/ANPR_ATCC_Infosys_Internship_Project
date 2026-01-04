# ==================================================
# 🔧 DYNAMIC MODULE RESOLUTION (STREAMLIT-CLOUD SAFE)
# ==================================================
import sys
import os

def add_module_path(module_filename: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for root, _, files in os.walk(base_dir):
        if module_filename in files:
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    raise ModuleNotFoundError(f"{module_filename} not found in project")

# Dynamically locate violation_engine.py
VIOLATION_ENGINE_DIR = add_module_path("violation_engine.py")

# ==================================================
# IMPORTS
# ==================================================
import streamlit as st
import cv2
import tempfile
import pandas as pd
import plotly.express as px

from traffic_engine import TrafficLightManager
from violation_engine import ViolationEngine

# ==================================================
# CONFIGURATION
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAFFIC_MODEL_PATH = os.path.join(
    BASE_DIR,
    "UVH_Traffic_Project",
    "yolov9e_a6000_run",
    "weights",
    "best.pt"
)

VIOLATION_MODEL_PATH = os.path.join(
    VIOLATION_ENGINE_DIR,
    "model",
    "best2.pt"
)

BIKE_SCAN_INTERVAL = 5
VIOLATION_SCAN_INTERVAL = 20

# ==================================================
# STREAMLIT SETUP
# ==================================================
st.set_page_config(
    page_title="AI Smart Traffic & Violation System",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦🛵 AI Smart Traffic & Two-Wheeler Violation Detection")

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header("System Settings")

conf_thresh = st.sidebar.slider(
    "Detection Confidence Threshold",
    0.1, 1.0, 0.35, 0.05
)

# ==================================================
# SESSION STATE
# ==================================================
if "run_simulation" not in st.session_state:
    st.session_state.run_simulation = False

if "violation_stats" not in st.session_state:
    st.session_state.violation_stats = {
        "bikes": 0,
        "helmet": 0,
        "mobile": 0,
        "triple": 0,
        "total": 0,
        "screenshots": []
    }

# ==================================================
# CONTROLS
# ==================================================
def start_simulation():
    st.session_state.run_simulation = True

def stop_simulation():
    st.session_state.run_simulation = False

# ==================================================
# LOAD MODELS
# ==================================================
@st.cache_resource(show_spinner=True)
def load_engines():
    traffic_engine = TrafficLightManager(TRAFFIC_MODEL_PATH)
    violation_engine = ViolationEngine(VIOLATION_MODEL_PATH)
    return traffic_engine, violation_engine

try:
    traffic_system, violation_engine = load_engines()
    traffic_system.conf_threshold = conf_thresh
    st.sidebar.success("✅ Models Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Model Load Error: {e}")
    st.stop()

# ==================================================
# VIDEO UPLOAD
# ==================================================
uploaded_video = st.file_uploader(
    "Upload CCTV Traffic Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)

    # ==================================================
    # UI LAYOUT
    # ==================================================
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Traffic & Violation Analysis")
        frame_box = st.empty()

    with col2:
        st.subheader("🚦 Signal Status")
        signal_box = st.empty()
        alert_box = st.empty()
        chart_box = st.empty()

    # ==================================================
    # LANE WAIT METRICS
    # ==================================================
    st.markdown("---")
    st.subheader("⏱ Lane Wait Time")

    lanes = ["UP", "DOWN", "LEFT", "RIGHT"]
    lane_cols = st.columns(4)
    lane_metrics = {lane: lane_cols[i].empty() for i, lane in enumerate(lanes)}

    # ==================================================
    # CONTROL BUTTONS
    # ==================================================
    btn1, btn2 = st.columns([1, 10])
    btn1.button("▶ Start Simulation", on_click=start_simulation, type="primary")
    btn2.button("⏹ Stop", on_click=stop_simulation)

    # ==================================================
    # MAIN LOOP
    # ==================================================
    if st.session_state.run_simulation:

        frame_id = 0
        bikes = []

        while cap.isOpened():

            if not st.session_state.run_simulation:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            frame = cv2.resize(frame, (1280, 720))

            # ---------------- TRAFFIC SYSTEM ----------------
            (
                frame,
                lane_data,
                active_lane,
                green_time,
                congestion_msg,
                wait_times
            ) = traffic_system.process_frame(frame)

            if congestion_msg:
                alert_box.warning(congestion_msg, icon="🚨")
            else:
                alert_box.empty()

            # ---------------- BIKE DETECTION ----------------
            if frame_id % BIKE_SCAN_INTERVAL == 0:
                bikes = violation_engine.detect_bikes(frame)
                st.session_state.violation_stats["bikes"] += len(bikes)

            # ---------------- VIOLATION DETECTION ----------------
            if frame_id % VIOLATION_SCAN_INTERVAL == 0:
                for idx, (x1, y1, x2, y2) in enumerate(bikes):
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    violations, screenshot, _ = violation_engine.detect_violations_on_bike(
                        roi,
                        frame_id,
                        f"{frame_id}_{idx}",
                        center
                    )

                    if violations:
                        st.session_state.violation_stats["total"] += 1

                        if "No Helmet" in violations:
                            st.session_state.violation_stats["helmet"] += 1
                        if "Mobile Usage" in violations:
                            st.session_state.violation_stats["mobile"] += 1
                        if "Triple Riding" in violations:
                            st.session_state.violation_stats["triple"] += 1

                        if screenshot is not None:
                            st.session_state.violation_stats["screenshots"].append(screenshot)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            frame,
                            "VIOLATION",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

            # ---------------- DISPLAY FRAME ----------------
            frame_box.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

            # ---------------- SIGNAL STATUS ----------------
            with signal_box.container():
                st.markdown(
                    f"""
                    ### Active Lane: **{active_lane}** 🟢  
                    **Green Time:** `{int(green_time)} sec`
                    """
                )

            # ---------------- TRAFFIC DENSITY CHART (FIXED) ----------------
            if lane_data:
                df = pd.DataFrame({
                    "Lane": list(lane_data.keys()),
                    "Density": list(lane_data.values())
                })

                fig = px.bar(
                    df,
                    x="Lane",
                    y="Density",
                    color="Lane",
                    title="Live Traffic Density"
                )

                chart_box.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"traffic_chart_{frame_id}"  # ✅ FIXED
                )

            # ---------------- WAIT TIME METRICS ----------------
            for lane, metric_box in lane_metrics.items():
                sec = wait_times.get(lane, 0.0)
                metric_box.metric(
                    label=f"{lane} Lane",
                    value=f"{sec:.1f} sec",
                    delta="Stopped" if sec > 5 else "Moving",
                    delta_color="inverse" if sec > 5 else "normal"
                )

        cap.release()
        st.session_state.run_simulation = False

# ==================================================
# VIOLATION EVIDENCE
# ==================================================
st.markdown("---")
st.subheader("📸 Violation Evidence")

cols = st.columns(4)
for i, img in enumerate(st.session_state.violation_stats["screenshots"][-8:]):
    cols[i % 4].image(img)
