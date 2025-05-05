import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk

# Load pre-trained model
model = load_model("emotion_on_grayscale.h5")  

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_landmarks.dat")

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Preprocess image
def preprocess_image(image):
    # Check if the image is empty
    if image is None or image.size == 0:
        return None
    
    # Check if the image is already grayscale (1 channel)
    if len(image.shape) == 2:  # Grayscale image
        pass  # No need to convert
    else:  # BGR image (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    image = cv2.equalizeHist(image)
    # Resize to 48x48
    image = cv2.resize(image, (48, 48))
    # Normalize pixel values to [0, 1]
    image = image.astype('float32') / 255.0
    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Function to draw text with background
def draw_text_with_background(frame, text, position, font_scale, font_color, bg_color, thickness=1):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    # Draw background rectangle
    cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width, y + 5), bg_color, -1)
    # Draw text
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

# Real-time emotion detection

# Global variables for frame skipping
frame_count = 0
cached_faces = []  # To store (face, prediction, emotion) tuples from the last detection

def detect_emotion():
    global frame_count, cached_faces
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        display_frame = frame.copy()  # Use a copy for UI display

        if frame_count % 5 == 0:
            # Perform detection on every third frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            # Cache the results to reuse in skipped frames
            cached_faces = []
            
            # Display the number of faces detected
            draw_text_with_background(frame, f"Number of Faces: {len(faces)}", (10, 30), 0.7, (0, 255, 0), (0, 0, 0))
            report_y = 60  # Start position for the report on the left side
            
            for i, face in enumerate(faces):
                landmarks = predictor(gray, face)
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_image = gray[y:y+h, x:x+w]
                
                if face_image.size == 0:
                    continue  # Skip if face image is empty
                
                processed_image = preprocess_image(face_image)
                if processed_image is None:
                    continue  # Skip if preprocessing fails
                
                prediction = model.predict(processed_image)
                emotion = emotion_labels[np.argmax(prediction)]
                
                # Cache detection result for use in skipped frames
                cached_faces.append((face, prediction, emotion))
                
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Display emotion report on the left side
                draw_text_with_background(frame, f"Face #{i+1}: {emotion}", (10, report_y), 0.5, (0, 255, 0), (0, 0, 0))
                report_y += 20
                for j, emotion_label in enumerate(emotion_labels):
                    draw_text_with_background(frame, f"{emotion_label}: {prediction[0][j]:.3f}", (10, report_y), 0.5, (0, 255, 0), (0, 0, 0))
                    report_y += 20
                # Display the emotion above the face
                draw_text_with_background(frame, f"Face #{i+1}: {emotion}", (x, y-10), 0.9, (0, 255, 0), (0, 0, 0))
        else:
            # For frames where detection is skipped, reuse the cached results if available
            if cached_faces:
                draw_text_with_background(frame, f"Number of Faces: {len(cached_faces)}", (10, 30), 0.7, (0, 255, 0), (0, 0, 0))
                report_y = 60
                for i, (face, prediction, emotion) in enumerate(cached_faces):
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    draw_text_with_background(frame, f"Face #{i+1}: {emotion}", (10, report_y), 0.5, (0, 255, 0), (0, 0, 0))
                    report_y += 20
                    for j, emotion_label in enumerate(emotion_labels):
                        draw_text_with_background(frame, f"{emotion_label}: {prediction[0][j]:.3f}", (10, report_y), 0.5, (0, 255, 0), (0, 0, 0))
                        report_y += 20
                    draw_text_with_background(frame, f"Face #{i+1}: {emotion}", (x, y-10), 0.9, (0, 255, 0), (0, 0, 0))
        
        # Resize frame to fit the Tkinter window and update the display
        frame = cv2.resize(frame, (video_label.winfo_width(), video_label.winfo_height()))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    
    video_label.after(10, detect_emotion)

# Initialize Tkinter
root = tk.Tk()
root.title("Real-Time Emotion Detection")

# Make the Tkinter window resizable
root.geometry("800x600")  # Initial window size
root.resizable(True, True)  # Allow resizing in both directions

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

# Video label
video_label = tk.Label(root)
video_label.pack(fill=tk.BOTH, expand=True)  # Allow the label to expand with the window

# Start detection
detect_emotion()

# Run the Tkinter event loop
root.mainloop()

# Release the camera
cap.release()