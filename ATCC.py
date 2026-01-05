import streamlit as st
import cv2
import tempfile
import pandas as pd
import plotly.express as px
from traffic_engine import TrafficLightManager
import os

# --- CONFIGURATION ---
MODEL_PATH = r"C:\Users\ADMIN\Documents\Dev_Projects\Traffic_Infosys\new_finetunning\FineTune_Project\scooter_fix_v1\weights\best.pt"

st.set_page_config(page_title="AI Traffic Manager", layout="wide")
st.title("🚦 AI Smart Traffic Light Control System")

# Sidebar
st.sidebar.header("System Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35)

if 'run_simulation' not in st.session_state:
    st.session_state['run_simulation'] = False

def start_logic(): st.session_state['run_simulation'] = True
def stop_logic(): st.session_state['run_simulation'] = False

@st.cache_resource
def load_engine():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return TrafficLightManager(MODEL_PATH)

try:
    traffic_system = load_engine()
    traffic_system.conf_threshold = conf_thresh
    st.sidebar.success("YOLOv9-E Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Traffic Video (CCTV Footage)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    
    # UI Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Analysis Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("Traffic Signal Status")
        signal_placeholder = st.empty()
        alert_placeholder = st.empty()
        chart_placeholder = st.empty()
    
    # --- SECTION: Bottom Stats ---
    st.markdown("---")
    st.subheader("⏱️ Lane Wait Time Analytics")
    #  4 columns for the main directions at the bottom
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_placeholders = {
        'UP': stat_col1.empty(),
        'DOWN': stat_col2.empty(),
        'LEFT': stat_col3.empty(),
        'RIGHT': stat_col4.empty()
    }
    
    btn_col1, btn_col2 = st.columns([1, 10])
    with btn_col1: st.button("Start Simulation", on_click=start_logic, type="primary")
    with btn_col2: st.button("Stop", on_click=stop_logic, type="secondary")

    if st.session_state['run_simulation']:
        frame_count = 0
        while cap.isOpened():
            if not st.session_state['run_simulation']: break

            ret, frame = cap.read()
            if not ret:
                st.session_state['run_simulation'] = False
                break
            
            frame_count += 1
            frame = cv2.resize(frame, (1280, 720))
            
            processed_frame, lane_data, active_lane, green_time, congestion_msg, wait_times = traffic_system.process_frame(frame)
            
            # 1. Alert
            if congestion_msg:
                alert_placeholder.warning(congestion_msg, icon="🚨")
            else:
                alert_placeholder.empty()

            # 2. Video
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # 3. Signal Panel
            colors = {k: '🔴' for k in lane_data.keys()}
            if active_lane in colors: colors[active_lane] = '🟢'
            
            with signal_placeholder.container():
                st.markdown(f"""
                ### Active Lane: **{active_lane}** {colors.get(active_lane, '🟢')}
                **Predicted Green Time:** `{int(green_time)} seconds`
                """)
            
            # 4. Chart
            active_lanes_data = {k: v for k, v in lane_data.items() if v > 0}
            if active_lanes_data:
                df = pd.DataFrame({'Direction': list(active_lanes_data.keys()), 'Density': list(active_lanes_data.values())})
                fig = px.bar(df, x='Direction', y='Density', color='Direction', title="Real-time Traffic Load")
                fig.update_traces(marker_color=['#00CC00' if x==active_lane else '#FF3333' for x in df['Direction']])
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"traffic_chart_{frame_count}")

            # 5. BOTTOM METRICS: Wait Times
            # Update the 4 main metrics at the bottom
            for direction, ph in stat_placeholders.items():
                seconds = wait_times.get(direction, 0.0)
                # If wait time > 0, show in Red (Stopped). Else Blue (Moving/Empty).
                delta_color = "inverse" if seconds > 5 else "normal" 
                label_text = "Stopped" if seconds > 0 else "Moving/Empty"
                
                ph.metric(
                    label=f"{direction} Lane Status", 
                    value=f"{seconds:.1f}s", 
                    delta=label_text,
                    delta_color=delta_color
                )

    cap.release()