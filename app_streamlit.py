import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

st.title("🌩️ Cloud-Based Video Surveillance with Object Detection")
st.markdown("Upload an image or video to detect objects using YOLOv8.")

uploaded_file = st.file_uploader("Upload image/video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    file_type = uploaded_file.type
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if "image" in file_type:
        img = Image.open(tfile.name)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        results = model(tfile.name, conf=0.3)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detection Result", use_container_width=True)
    else:
        st.video(tfile.name)
        st.write("🔍 Detecting objects...")
        results = model.predict(tfile.name, conf=0.3, stream=True)
        save_path = os.path.join("detections_output.mp4")
        model.export(format="mp4")
        st.success("Video processed successfully!")

st.caption("Built with YOLOv8 + Streamlit")

