import streamlit as st
import cv2
import tempfile
import pandas as pd
import plotly.express as px
from traffic_engine import TrafficLightManager
import os
from collections import Counter

# --- CONFIGURATION ---
TRAFFIC_MODEL_PATH = "model1/best.pt"
VIOLATION_MODEL_PATH = "model2/best2.pt" 

st.set_page_config(
    page_title="AI Traffic Command Center", 
    layout="wide", 
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    .stAlert { padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    st.subheader("System Status")
    status_indicator = st.empty()
    status_indicator.info("Ready to Start")
    
    st.markdown("---")
    st.subheader("Simulation Controls")
    
    if 'run_simulation' not in st.session_state: st.session_state['run_simulation'] = False
    def start_logic(): st.session_state['run_simulation'] = True
    def stop_logic(): st.session_state['run_simulation'] = False

    col_start, col_stop = st.columns(2)
    with col_start: st.button("START", on_click=start_logic, type="primary", use_container_width=True)
    with col_stop: st.button("STOP", on_click=stop_logic, type="secondary", use_container_width=True)

    st.markdown("---")
    st.subheader("AI Sensitivity Settings")
    
    # --- NEW: DETAILED CONFIDENCE SLIDERS ---
    st.write("**Traffic Detection**")
    conf_traffic = st.slider("Vehicle Detection", 0.1, 1.0, 0.35)
    
    st.write("**Violation Specifics**")
    conf_helmet = st.slider("No Helmet Confidence", 0.1, 1.0, 0.80)
    conf_mobile = st.slider("Mobile Use Confidence", 0.1, 1.0, 0.50)
    conf_triple = st.slider("Triple Riding (Rider Detect)", 0.1, 1.0, 0.60)
    

# --- MAIN DASHBOARD ---
st.title("Smart Traffic Management System")

@st.cache_resource
def load_engine():
    if not os.path.exists(TRAFFIC_MODEL_PATH): raise FileNotFoundError(f"Missing Traffic Model")
    if not os.path.exists(VIOLATION_MODEL_PATH): raise FileNotFoundError(f"Missing Violation Model")
    return TrafficLightManager(TRAFFIC_MODEL_PATH, VIOLATION_MODEL_PATH)

try:
    traffic_system = load_engine()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

uploaded_file = st.file_uploader("📂 Upload CCTV Footage (MP4/AVI)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    cap = cv2.VideoCapture(tfile.name)
    
    # HUD
    st.subheader("📢 Live Network Status")
    col_sig, col_veh, col_viol, col_alert = st.columns(4)
    with col_sig:
        st.markdown("**🚥 Signal Status**")
        signal_metric = st.empty()
    with col_veh:
        st.markdown("**🚗 Total Vehicles**")
        vehicle_metric = st.empty()
    with col_viol:
        st.markdown("**👮 Total Violations**")
        violation_metric = st.empty()
    with col_alert:
        st.markdown("**⚠️ Alerts**")
        alert_metric = st.empty()

    st.markdown("---")
    video_placeholder = st.empty()

    tab_traffic, tab_viol, tab_debug = st.tabs(["📊 Traffic Analytics", "📸 Violation Evidence & OCR", "🛠️ System Diagnostics"])
    
    with tab_traffic:
        t_col1, t_col2 = st.columns([1, 2])
        with t_col1:
            st.write("**Lane Wait Times**")
            wait_placeholders = {'UP': st.empty(), 'DOWN': st.empty(), 'LEFT': st.empty(), 'RIGHT': st.empty()}
        with t_col2: chart_placeholder = st.empty()

    with tab_viol:
        evidence_header = st.empty()
        evidence_gallery = st.empty()

    with tab_debug:
        st.info("White Box = Raw Detection | Blue Box = ROI sent to Violation Model")
        debug_gallery = st.empty()

    if st.session_state['run_simulation']:
        status_indicator.success("Processing...")
        traffic_system.reset()
        
        # --- APPLY SLIDER SETTINGS TO ENGINE ---
        traffic_system.conf_threshold = conf_traffic
        traffic_system.violation_engine.helmet_conf = conf_helmet
        traffic_system.violation_engine.mobile_conf = conf_mobile
        traffic_system.violation_engine.rider_conf = conf_triple
        # ---------------------------------------

        frame_count = 0
        while cap.isOpened():
            if not st.session_state['run_simulation']: 
                status_indicator.warning("Simulation Stopped")
                break
            
            ret, frame = cap.read()
            if not ret:
                st.session_state['run_simulation'] = False
                status_indicator.info("Video Ended")
                break
            
            frame_count += 1
            frame = cv2.resize(frame, (1280, 720))
            
            processed_frame, lane_data, active_lane, green_time, congestion_msg, \
            wait_times, v_counts, total_v, viol_stats, recent_violations, debug_crops = \
            traffic_system.process_frame(frame, frame_count)
            
            colors = {k: '🔴' for k in lane_data.keys()}
            if active_lane in colors: colors[active_lane] = '🟢'
            signal_metric.markdown(f"<h2>{colors.get(active_lane, '🟢')} {active_lane} : {int(green_time)}s</h2>", unsafe_allow_html=True)
            vehicle_metric.metric("Count", total_v, label_visibility="collapsed")
            v_color = "normal" if viol_stats["Total Violations"] == 0 else "inverse"
            violation_metric.metric("Count", viol_stats["Total Violations"], delta_color=v_color, label_visibility="collapsed")
            
            if congestion_msg: alert_metric.error(congestion_msg, icon="🚨")
            else: alert_metric.success("Traffic Flow: Normal", icon="✅")

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Charts & Evidence (Same Logic)
            active_lanes_data = {k: v for k, v in lane_data.items() if v > 0}
            if active_lanes_data:
                df = pd.DataFrame({'Direction': list(active_lanes_data.keys()), 'Density': list(active_lanes_data.values())})
                fig = px.bar(df, x='Direction', y='Density', color='Direction', height=300)
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{frame_count}")
            
            for direction, ph in wait_placeholders.items():
                sec = wait_times.get(direction, 0.0)
                ph.metric(direction, f"{sec:.1f}s", delta="Stopped" if sec > 0 else "Moving", delta_color="inverse" if sec > 5 else "normal")

            if recent_violations:
                evidence_header.write(f"**Latest Evidence (Total: {len(recent_violations)})**")
                with evidence_gallery.container():
                    cols = st.columns(4)
                    recent_list = list(recent_violations)[-4:]
                    for i, item in enumerate(recent_list):
                        caption = f"Plate: {item['plate']}\n{', '.join(item['violations'])}"
                        cols[i].image(item['image'], caption=caption, use_container_width=True)
            else: evidence_header.info("No violations detected yet.")

            if debug_crops:
                with debug_gallery.container():
                    debug_list = list(debug_crops)
                    for i in range(0, len(debug_list), 4):
                        d_cols = st.columns(4)
                        for j in range(4):
                            if i + j < len(debug_list):
                                d_cols[j].image(debug_list[i+j], caption="Debug Crop", use_container_width=True)

        # --- FINAL REPORT---
        st.markdown("---")
        st.header("Final Session Report")
        
        unique_classes = traffic_system.unique_vehicle_classes
        if unique_classes:
            counts = Counter(unique_classes.values())
            df_final = pd.DataFrame(list(counts.items()), columns=['Vehicle Type', 'Count']).sort_values('Count', ascending=False)
            
            sum1, sum2, sum3, sum4 = st.columns(4)
            sum1.metric("Total Vehicles", sum(counts.values()))
            sum2.metric("Total Violations", viol_stats["Total Violations"])
            sum3.metric("Helmet Violations", viol_stats["No Helmet"])
            sum4.metric("Mobile Violations", viol_stats["Mobile Usage"])
            
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(df_final, names='Vehicle Type', values='Count', title='Vehicle Distribution', hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                fig_bar = px.bar(df_final, x='Vehicle Type', y='Count', color='Vehicle Type', title='Counts by Type')
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No vehicle data collected.")

    cap.release()
else:
    st.info("Please upload a video file to begin.")