import streamlit as st
import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os


def main():
    """Entry‑point for the Streamlit front‑end."""
    st.set_page_config(
        page_title="Emotion Detection App",
        page_icon="😀",
        layout="wide",
    )

    st.title("Real‑Time Emotion Detection")

    # ─────────────────────────────────────────────────────────────────────────────
    # NAVIGATION BAR (replaces the old sidebar <selectbox>)
    # ─────────────────────────────────────────────────────────────────────────────
    # Streamlit does not have a native top‑level nav bar, but `st.tabs` provides a
    # clean horizontal navigation UX that behaves like a navbar.
    #
    # Each tab hosts one of the three functional pages that were previously
    # selected from a dropdown.
    #
    # If you prefer a *side* navigation, swap `st.tabs` for `st.sidebar.radio`.
    # ─────────────────────────────────────────────────────────────────────────────
    about_tab, video_tab, image_tab = st.tabs([
        "About", "Run on Video", "Upload Image"
    ])

    with about_tab:
        show_about_page()

    with video_tab:
        run_on_video()

    with image_tab:
        run_on_image()


# ==============================================================================
# PAGE: ABOUT
# ==============================================================================

def show_about_page():
    st.markdown(
        """
        ## About This App
        
        This application uses deep learning to detect emotions in real‑time from
        your webcam feed or from uploaded images.
        
        ### How it works
        1. Face detection using **dlib**'s HOG‑based face detector  
        2. Facial landmark extraction  
        3. Face preprocessing (grayscale conversion, histogram equalisation)  
        4. Emotion classification using a pre‑trained Convolutional Neural Network
        
        ### Detected Emotions
        * Angry  
        * Disgust  
        * Fear  
        * Happy  
        * Sad  
        * Surprise  
        * Neutral  
        
        ### Tech Stack
        * **Streamlit** for the web interface  
        * **OpenCV** for image processing  
        * **dlib** for face detection  
        * **TensorFlow/Keras** for the emotion‑classification model
        """
    )


# ==============================================================================
# MODEL LOADING HELPERS  (cached so we only load once per session)
# ==============================================================================

@st.cache_resource
def load_detection_model():
    detector = dlib.get_frontal_face_detector()
    predictor_path = "face_landmarks.dat"

    if not os.path.exists(predictor_path):
        st.error(
            f"Error: {predictor_path} not found. Please download it and place it in the app directory.")
        st.stop()

    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor


@st.cache_resource
def load_emotion_model():
    model_path = "emotion.h5"
    if not os.path.exists(model_path):
        st.error(
            f"Error: {model_path} not found. Please download it and place it in the app directory.")
        st.stop()
    return load_model(model_path)


# ==============================================================================
# COMMON PROCESSING UTILITIES
# ==============================================================================

def preprocess_image(image):
    """Convert input face crop to 48×48 grayscale, normalised tensor."""
    if image is None or image.size == 0:
        return None
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.equalizeHist(image)
    image = cv2.resize(image, (48, 48))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)  # channel dim
    image = np.expand_dims(image, axis=0)   # batch dim
    return image


def process_image(image, detector, predictor, model, labels):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    result_image = image.copy()
    face_emotions = []

    for i, face in enumerate(detector(gray)):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = gray[y:y + h, x:x + w]
        if face_img.size == 0:
            continue
        tensor = preprocess_image(face_img)
        if tensor is None:
            continue
        probs = model.predict(tensor)[0]
        idx = int(np.argmax(probs))
        emotion = labels[idx]

        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        face_emotions.append({
            "face_id": i + 1,
            "emotion": emotion,
            "probabilities": {lab: float(p) for lab, p in zip(labels, probs)},
        })

    return result_image, face_emotions


# ==============================================================================
# PAGE: UPLOAD IMAGE
# ==============================================================================

def run_on_image():
    detector, predictor = load_detection_model()
    model = load_emotion_model()
    labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if not file:
        return

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Original Image", use_column_width=True)

    result, faces = process_image(image, detector, predictor, model, labels)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Emotions")
        st.image(result, channels="RGB", use_column_width=True)
    with col2:
        st.subheader("Emotion Analysis")
        if not faces:
            st.write("No faces detected.")
        else:
            for face in faces:
                st.write(f"### Face #{face['face_id']}: {face['emotion']}")
                st.bar_chart({"Emotion": list(face["probabilities"].keys()),
                              "Probability": list(face["probabilities"].values())})


# ==============================================================================
# PAGE: RUN ON VIDEO
# ==============================================================================

def run_on_video():
    detector, predictor = load_detection_model()
    model = load_emotion_model()
    labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    st.sidebar.header("Video Settings")
    source = st.sidebar.radio("Select video source", ["Webcam", "Upload Video"])

    if source == "Webcam":
        cap = cv2.VideoCapture(0)
        stop = st.sidebar.button("Stop Webcam")
        frame_holder = st.empty()
        while cap.isOpened() and not stop:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result, faces = process_image(frame, detector, predictor, model, labels)
            frame_holder.image(result, channels="RGB", use_column_width=True)
        cap.release()

    else:  # Upload Video
        file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
        if file:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(file.read())
            cap = cv2.VideoCapture(tmp.name)
            frame_holder = st.empty()
            progress = st.progress(0.0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            idx = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result, _ = process_image(frame, detector, predictor, model, labels)
                frame_holder.image(result, channels="RGB", use_column_width=True)
                idx += 1
                progress.progress(min(idx / total, 1.0))
            cap.release()
            os.unlink(tmp.name)


if __name__ == "__main__":
    main()
