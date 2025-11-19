from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from datetime import datetime

app = Flask(__name__)

model = load_model('emotion_model.h5') 
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

emotion_history = []

def prepare_image(img, target_size=(48,48)):
    """
    Convert image to grayscale, resize, normalize, and add batch dimension
    """
    if isinstance(img, Image.Image):
        img = img.convert('L')  # Grayscale
        img = img.resize(target_size)
        img = np.array(img)
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, target_size)

    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # (48,48,1)
    img = np.expand_dims(img, axis=0)   # (1,48,48,1)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file uploaded"
        img = Image.open(file)
        img = prepare_image(img)
        prediction = model.predict(img)[0]
        label = emotion_labels[np.argmax(prediction)]

        username = request.form.get('username', 'Unknown')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emotion_history.append({'username': username, 'emotion': label, 'timestamp': timestamp})
        return f"Predicted Emotion: {label}"
    return render_template('upload.html')

@app.route('/webcam_page')
def webcam_page():
    return render_template('webcam.html')

@app.route('/webcam', methods=['POST'])
def webcam():
    file = request.files.get('file')
    if not file:
        return "No image captured"
    img = Image.open(file)
    img = prepare_image(img)
    prediction = model.predict(img)[0]
    label = emotion_labels[np.argmax(prediction)]

    username = request.form.get('username', 'Unknown')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emotion_history.append({'username': username, 'emotion': label, 'timestamp': timestamp})
    return f"Predicted Emotion: {label}"

@app.route('/view_log')
def view_log():
    return render_template('view_log.html', logs=emotion_history)

if __name__ == "__main__":
    app.run(debug=True)
