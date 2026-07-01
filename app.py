import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)

# Initialize the lightweight TFLite Interpreter safely
# Ensure emotion_model.tflite is uploaded to your main GitHub folder!
MODEL_PATH = "emotion_model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Map the internal input/output tensor arrays
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    # Convert OpenCV image matrix array to grayscale securely
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, target_size)
    
    # Process normalization and shape conversion for BOTH types
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
            
            # Tiny structural tensor computation to save server memory
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
            
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
        
        # Tiny structural tensor computation to save server memory
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        
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
