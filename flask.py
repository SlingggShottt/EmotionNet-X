from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model
import os
import base64
from PIL import Image
import io

app = Flask(__name__, static_folder='frontend/build', static_url_path='')

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_landmarks.dat")
model = load_model("emotion.h5")

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

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/detect', methods=['POST'])
def detect_emotion():
    # Get image from request
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    # Decode base64 image
    try:
        image_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400
    
    # Process image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Create result image for display
    result_image = image.copy()
    
    # Process each detected face
    results = []
    for i, face in enumerate(faces):
        # Get face landmarks
        landmarks = predictor(gray, face)
        
        # Get face bounding box
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = gray[y:y+h, x:x+w]
        
        # Skip if face image is empty
        if face_image.size == 0:
            continue
        
        # Preprocess face for emotion detection
        processed_image = preprocess_image(face_image)
        if processed_image is None:
            continue
        
        # Predict emotion
        prediction = model.predict(processed_image)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        
        # Draw rectangle around face
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw text with emotion
        cv2.putText(result_image, f"{emotion}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Add result to list
        face_result = {
            'face_id': i+1,
            'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'emotion': emotion,
            'probabilities': {label: float(prob) for label, prob in zip(emotion_labels, prediction[0])}
        }
        results.append(face_result)
    
    # Convert result image to base64
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_image)
    buffer = io.BytesIO()
    result_pil.save(buffer, format='JPEG')
    result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Return results
    return jsonify({
        'num_faces': len(results),
        'faces': results,
        'result_image': f'data:image/jpeg;base64,{result_base64}'
    })

@app.route('/api/video', methods=['POST'])
def process_video_frame():
    # Get image from request
    if 'frame' not in request.json:
        return jsonify({'error': 'No frame provided'}), 400
    
    # Decode base64 image
    try:
        image_data = request.json['frame'].split(',')[1] if ',' in request.json['frame'] else request.json['frame']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400
    
    # Process image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Create result image for display
    result_image = image.copy()
    
    # Process each detected face
    results = []
    for i, face in enumerate(faces):
        # Get face landmarks
        landmarks = predictor(gray, face)
        
        # Get face bounding box
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = gray[y:y+h, x:x+w]
        
        # Skip if face image is empty
        if face_image.size == 0:
            continue
        
        # Preprocess face for emotion detection
        processed_image = preprocess_image(face_image)
        if processed_image is None:
            continue
        
        # Predict emotion
        prediction = model.predict(processed_image)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        
        # Draw rectangle around face
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw text with emotion
        cv2.putText(result_image, f"{emotion}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Add result to list
        face_result = {
            'face_id': i+1,
            'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'emotion': emotion,
            'probabilities': {label: float(prob) for label, prob in zip(emotion_labels, prediction[0])}
        }
        results.append(face_result)
    
    # Convert result image to base64
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_image)
    buffer = io.BytesIO()
    result_pil.save(buffer, format='JPEG')
    result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Return results
    return jsonify({
        'num_faces': len(results),
        'faces': results,
        'result_image': f'data:image/jpeg;base64,{result_base64}'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Emotion detection API is running'})

@app.route('/api/models', methods=['GET'])
def get_models_info():
    # Return information about loaded models
    return jsonify({
        'face_detector': 'dlib HOG-based face detector',
        'landmark_predictor': 'dlib shape predictor',
        'emotion_model': 'Keras CNN model',
        'emotion_labels': emotion_labels
    })

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Check if models exist
    if not os.path.exists('face_landmarks.dat'):
        print("Error: face_landmarks.dat not found. Please download it.")
    
    if not os.path.exists('emotion.h5'):
        print("Error: emotion.h5 not found. Please download it.")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)