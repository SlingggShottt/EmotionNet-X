from __future__ import annotations

import bz2
import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, List

import cv2
import dlib
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA  # placeholder if you want charts later
from tensorflow.keras.models import load_model

################################################################################
# CONFIG
################################################################################

ST_EMOTIONS: List[str] = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]

LANDMARK_BZ2 = "shape_predictor_5_face_landmarks.dat.bz2"  # < 25 MB
LANDMARK_DAT = LANDMARK_BZ2[:-4]                            # decompressed
MODEL_H5     = "emotion_on_grayscale.h5"                   # 1‑channel CNN

################################################################################
# UTILS
################################################################################

@st.cache_resource(show_spinner=False)
def ensure_landmark_model() -> Tuple[dlib.fhog_object_detector, dlib.shape_predictor]:
    """Decompress the .bz2 landmark model on first run and return dlib models."""

    if not Path(LANDMARK_DAT).exists():
        if not Path(LANDMARK_BZ2).exists():
            st.error(
                f"Missing {LANDMARK_BZ2}. Upload it to your repo / deployment directory.")
            st.stop()
        with st.spinner("Decompressing 5‑point landmark model …"):
            with bz2.open(LANDMARK_BZ2, "rb") as f_in, open(LANDMARK_DAT, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_DAT)
    return detector, predictor


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    if not Path(MODEL_H5).exists():
        st.error(f"Model file {MODEL_H5} not found.")
        st.stop()
    return load_model(MODEL_H5, compile=False)


def preprocess_face(face_gray: np.ndarray, expected_channels: int) -> np.ndarray:
    """Histogram‑equalise, normalise, and shape to (48,48,1|3)."""
    face = cv2.equalizeHist(face_gray)
    face = face.astype("float32") / 255.0
    if expected_channels == 1:
        face = np.expand_dims(face, axis=-1)        # (48,48,1)
    else:  # replicate to 3‑channel RGB
        face = np.stack([face]*3, axis=-1)          # (48,48,3)
    face = np.expand_dims(face, axis=0)             # add batch dim
    return face


def analyse_frame(frame_rgb: np.ndarray,
                  detector: dlib.fhog_object_detector,
                  predictor: dlib.shape_predictor,
                  model,
                  labels: List[str]):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    out  = frame_rgb.copy()
    faces = detector(gray)
    predictions = []

    for idx, face in enumerate(faces, 1):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        crop = gray[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (48, 48))
        tensor = preprocess_face(crop, model.input_shape[-1])
        probs  = model.predict(tensor, verbose=0)[0]
        label  = labels[int(np.argmax(probs))]

        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(out, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        predictions.append({"face_id": idx, "label": label, "probs": probs})

    return out, predictions

################################################################################
# STREAMLIT APP
################################################################################

def main():
    st.set_page_config(page_title="EmotionNet‑X", page_icon="😃", layout="wide")

    st.title("EmotionNet‑X – Facial Emotion Recognition")

    # Horizontal navigation bar via tabs
    about_tab, video_tab, image_tab = st.tabs(["About", "Video", "Image"])

    with about_tab:
        st.markdown(
            """
            ### About this project
            This demo showcases real‑time facial‑emotion recognition using a
            lightweight residual CNN trained on the FER‑2013 dataset.

            **Pipeline**
            1. Face detection (dlib HOG)
            2. 5‑point landmark alignment
            3. Pre‑processing: grayscale, 48×48, histogram equalisation
            4. Emotion classification (7 classes)
            5. Overlay of predictions in the UI
            """
        )

    with video_tab:
        run_video_page()

    with image_tab:
        run_image_page()

################################################################################
# VIDEO PAGE
################################################################################

def run_video_page():
    detector, predictor = ensure_landmark_model()
    model              = load_emotion_model()
    labels             = ST_EMOTIONS

    st.sidebar.header("Video source")
    source = st.sidebar.radio("Choose", ["Webcam", "Upload MP4/AVI"])

    frame_holder = st.empty()

    if source == "Webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam not accessible.")
            return
        stop_btn = st.sidebar.button("Stop webcam")
        while cap.isOpened() and not stop_btn:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vis, _ = analyse_frame(frame, detector, predictor, model, labels)
            frame_holder.image(vis, channels="RGB", use_column_width=True)
        cap.release()

    else:
        file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
        if not file:
            st.info("Please upload a video file.")
            return
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        pbar  = st.progress(0.0)
        idx   = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vis, _ = analyse_frame(frame, detector, predictor, model, labels)
            frame_holder.image(vis, channels="RGB", use_column_width=True)
            idx += 1
            pbar.progress(min(idx/total, 1.0))
        cap.release()
        os.unlink(tfile.name)

################################################################################
# IMAGE PAGE
################################################################################

def run_image_page():
    detector, predictor = ensure_landmark_model()
    model              = load_emotion_model()
    labels             = ST_EMOTIONS

    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if not file:
        return

    image = Image.open(file).convert("RGB")
    frame = np.array(image)

    vis, preds = analyse_frame(frame, detector, predictor, model, labels)

    st.image(vis, caption="Detection result", use_column_width=True)

    if preds:
        st.subheader("Emotion probabilities per face")
        for info in preds:
            st.write(f"**Face #{info['face_id']} – {info['label']}**")
            st.bar_chart({"emotion": labels, "prob": info["probs"]})
    else:
        st.info("No faces detected in the uploaded image.")

################################################################################

if __name__ == "__main__":
    main()
