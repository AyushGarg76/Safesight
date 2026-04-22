import streamlit as st
import cv2
import tempfile
from detector import process_frame

import time

st.set_page_config(layout="wide")
st.title("🦺 Non-YOLO Helmet Detection System")

uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    alert_box = st.empty()  # 🚨 for alert display

    prev_time = time.time()  # ✅ FIX: initialize here

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 🔽 Resize for speed
        frame = cv2.resize(frame, (640, 480))

        # 🔍 Process frame
        annotated, violations = process_frame(frame)

        # ⏱ FPS calculation (FIXED)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # 🖥️ Display FPS
        cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 🚨 ALERT SYSTEM (FIXED)
        if violations > 0:
            alert_box.error("🚨 NO HELMET DETECTED!")

        # 🎥 Display frame
        stframe.image(annotated, channels="BGR")

    cap.release()