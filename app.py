# app.py
from flask import Flask, render_template, Response, send_from_directory, flash, redirect, url_for
from ultralytics import YOLO
import cv2
import os
import datetime
import glob
import time

USE_CLOUD = False  

if USE_CLOUD:
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
    except Exception as e:
        raise RuntimeError("pydrive2 is required for USE_CLOUD=True. Run: pip install pydrive2") from e

app = Flask(__name__)
app.secret_key = "replace-with-random-secret"

# -------------------------
# Paths
# -------------------------
ROOT = app.root_path
CLOUD_FOLDER = os.path.join(ROOT, "cloud_storage")
if not os.path.exists(CLOUD_FOLDER):
    os.makedirs(CLOUD_FOLDER, exist_ok=True)

# -------------------------
# Google Drive setup
# -------------------------
if USE_CLOUD:
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    print("Google Drive authenticated. - app.py:37")

# -------------------------
# Camera + Model
# -------------------------
CAM_INDEX = 0
camera = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if os.name == "nt" else 0)

# YOLOv8 small model for better accuracy
model = YOLO("yolov8s.pt")
try:
    model.fuse()
except:
    pass

frame_count = 0
SKIP_FRAMES = 1  # skip alternate frames

# CLAHE preprocessing
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# -------------------------
# Frame generator
# -------------------------
def generate_frames():
    global frame_count
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            success, frame = camera.read()
            if not success:
                break

        # Resize for better detection
        frame = cv2.resize(frame, (1280, 720))

        # Apply CLAHE for low-light enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        frame_count += 1
        if SKIP_FRAMES and (frame_count % (SKIP_FRAMES + 1) != 0):
            continue

        # YOLO detection with confidence and IoU tuning
        results = model(frame, conf=0.3, iou=0.45)[0]

        # Annotate frame
        annotated_frame = results.plot()

        # Overlay object count
        try:
            labels = results.boxes.cls
            object_count = len(labels)
        except:
            object_count = 0
        cv2.putText(annotated_frame, f"Objects: {object_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Save original detection image
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"detect_{now}.jpg"
        filepath = os.path.join(CLOUD_FOLDER, filename)
        cv2.imwrite(filepath, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        # Optional Google Drive upload
        if USE_CLOUD:
            try:
                f = drive.CreateFile({'title': filename})
                f.SetContentFile(filepath)
                f.Upload()
                print(f"Uploaded to Drive: {filename} - app.py:110")
            except Exception as e:
                print("Drive upload error: - app.py:112", e)

        # -------------------------
        # Augmentation
        # -------------------------
        aug_frames = []

        # Horizontal flip
        aug_frames.append(cv2.flip(annotated_frame, 1))
        # Brightness increase
        bright = cv2.convertScaleAbs(annotated_frame, alpha=1.2, beta=20)
        aug_frames.append(bright)
        # Contrast increase
        contrast = cv2.convertScaleAbs(annotated_frame, alpha=1.3, beta=0)
        aug_frames.append(contrast)
        # Gaussian blur
        blur = cv2.GaussianBlur(annotated_frame, (5,5), 0)
        aug_frames.append(blur)

        # Save augmented frames
        for i, aug in enumerate(aug_frames):
            aug_filename = f"detect_{now}_aug{i+1}.jpg"
            aug_filepath = os.path.join(CLOUD_FOLDER, aug_filename)
            cv2.imwrite(aug_filepath, aug, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            # Optional Google Drive upload
            if USE_CLOUD:
                try:
                    f = drive.CreateFile({'title': aug_filename})
                    f.SetContentFile(aug_filepath)
                    f.Upload()
                    print(f"Uploaded augmented: {aug_filename} - app.py:143")
                except Exception as e:
                    print("Drive upload error: - app.py:145", e)

        # Encode original frame for streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gallery')
def gallery():
    image_paths = sorted(glob.glob(os.path.join(CLOUD_FOLDER, "*.jpg")), reverse=True)[:30]
    image_names = [os.path.basename(p) for p in image_paths]
    return render_template('gallery.html', images=image_names)

@app.route('/cloud_storage/<path:filename>')
def serve_image(filename):
    return send_from_directory(CLOUD_FOLDER, filename)

@app.route('/clear_storage')
def clear_storage():
    files = glob.glob(os.path.join(CLOUD_FOLDER, '*.jpg'))
    for f in files:
        try:
            os.remove(f)
        except:
            pass
    flash("Cleared cloud_storage folder.")
    return redirect(url_for('gallery'))

# -------------------------
# Cleanup
# -------------------------
def cleanup():
    try:
        camera.release()
    except:
        pass

import atexit
atexit.register(cleanup)

# -------------------------
# Run app
# -------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
