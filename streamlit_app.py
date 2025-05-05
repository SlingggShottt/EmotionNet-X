import os
import subprocess
import threading
import time

FLASK_PORT = os.getenv("PORT", 5000)  # Use Railway/Render assigned port or default to 5000
STREAMLIT_PORT = 8501  # Explicit Streamlit port

def start_flask_server():
    """Start Flask API server in a separate thread."""
    print(f"Starting Flask API server on port {FLASK_PORT}...")
    subprocess.Popen(["python", "flask.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def start_streamlit_app():
    """Start the Streamlit app in a separate thread."""
    print(f"Starting Streamlit app on port {STREAMLIT_PORT}...")
    subprocess.Popen(["streamlit", "run", "web.py", "--server.port", str(STREAMLIT_PORT)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    """Main function to start both servers."""
    print("🚀 Starting Emotion Detection Web Application")

    # Check if required files exist
    required_files = ["face_landmarks.dat", "emotion_on_grayscale.h5"]
    missing_files = [file for file in required_files if not os.path.exists(file)]
    
    if missing_files:
        print("❌ Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease download these files before running the application.")
        return

    # Start Flask and Streamlit in separate threads
    threading.Thread(target=start_flask_server, daemon=True).start()
    
    print("⌛ Waiting for Flask server to start...")
    time.sleep(5)  # Allow Flask time to start

    threading.Thread(target=start_streamlit_app, daemon=True).start()

    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
