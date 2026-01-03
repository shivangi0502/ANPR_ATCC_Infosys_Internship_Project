import streamlit as st
import cv2
import tempfile
from violation_engine import ViolationEngine

# ---------------- CONFIG ----------------

MODEL_PATH = "model/best2.pt"
BIKE_SCAN_INTERVAL = 5
VIOLATION_SCAN_INTERVAL = 20

st.set_page_config(page_title="Two-Wheeler Violation Detection", layout="wide")
st.title("Two-Wheeler Traffic Violation Detection")

engine = ViolationEngine(MODEL_PATH)


# ---------------- SESSION STATE ----------------
if "stats" not in st.session_state:
    st.session_state.stats = {
        "bikes": 0,
        "helmet": 0,
        "mobile": 0,
        "triple": 0,
        "total": 0,
        "screenshots": []
    }


# ---------------- VIDEO INPUT ----------------

uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])
frame_box = st.empty()

if uploaded_video:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_video.read())
    cap = cv2.VideoCapture(temp.name)

    frame_id = 0
    bikes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame = cv2.resize(frame, (1280, 720))

        # ---- Bike gate ----
        if frame_id % BIKE_SCAN_INTERVAL == 0:
            bikes = engine.detect_bikes(frame)
            if bikes:
                st.session_state.stats["bikes"] += len(bikes)

        # ---- Per-bike violation ----
        if frame_id % VIOLATION_SCAN_INTERVAL == 0:
            for idx, (x1, y1, x2, y2) in enumerate(bikes):
                bike_roi = frame[y1:y2, x1:x2]
                if bike_roi.size == 0:
                    continue

                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                violations, screenshot, _ = engine.detect_violations_on_bike(
                    bike_roi=bike_roi,
                    frame_id=frame_id,
                    bike_id=f"{frame_id}_{idx}",
                    global_center=center
                )

                if violations:
                    st.session_state.stats["total"] += 1
                    if "No Helmet" in violations:
                        st.session_state.stats["helmet"] += 1
                    if "Mobile Usage" in violations:
                        st.session_state.stats["mobile"] += 1
                    if "Triple Riding" in violations:
                        st.session_state.stats["triple"] += 1

                    if screenshot:
                        st.session_state.stats["screenshots"].append(screenshot)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        frame,
                        "VIOLATION",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        frame_box.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
        )

    cap.release()

# ---------------- EVIDENCE GALLERY ----------------
st.markdown("---")
st.subheader("Violation Evidence(Cropped image of the vehicle)")

cols = st.columns(4)
for i, img in enumerate(st.session_state.stats["screenshots"][-8:]):
    cols[i % 4].image(img)
