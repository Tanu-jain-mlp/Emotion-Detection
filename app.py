import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from datetime import datetime

app = Flask(__name__)

# Load model only once
model = load_model("emotion_model.h5")

emotion_labels = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

emotion_history = []

def prepare_image(img, target_size=(48, 48)):
    # Convert PIL Image to numpy array grayscale
    if isinstance(img, Image.Image):
        img = img.convert('L')
        img = img.resize(target_size)
        img = np.array(img)
    # Convert OpenCV image to grayscale
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, target_size)
    
    # These lines must run for BOTH image types
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file:
                return "No file uploaded"
            img = Image.open(file)
            img = prepare_image(img)
            
            prediction = model.predict(img, verbose=0)[0]
            label = emotion_labels[np.argmax(prediction)]
            
            username = request.form.get('username', 'Unknown')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            emotion_history.append({
                'username': username,
                'emotion': label,
                'timestamp': timestamp
            })
            return f"Predicted Emotion: {label}"
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('upload.html')

@app.route('/webcam_page')
def webcam_page():
    return render_template('webcam.html')

@app.route('/webcam', methods=['POST'])
def webcam():
    try:
        file = request.files.get('file')
        if not file:
            return "No image captured"
        img = Image.open(file)
        img = prepare_image(img)
        
        prediction = model.predict(img, verbose=0)[0]
        label = emotion_labels[np.argmax(prediction)]
        
        username = request.form.get('username', 'Unknown')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        emotion_history.append({
            'username': username,
            'emotion': label,
            'timestamp': timestamp
        })
        return f"Predicted Emotion: {label}"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/view_log')
def view_log():
    return render_template('view_log.html', logs=emotion_history)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
