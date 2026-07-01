import os
# Force CPU environment parameters instantly
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from datetime import datetime

app = Flask(__name__)

# Load original model safely
model = load_model("emotion_model.h5")

emotion_labels = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

emotion_history = []

def prepare_image(img, target_size=(48, 48)):
    if isinstance(img, Image.Image):
        img = img.convert('L')
        img = img.resize(target_size)
        img = np.array(img)
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, target_size)
    
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
            
            # Convert raw numpy array array to strict input tensor data type
            tensor_img = tf.convert_to_tensor(img, dtype=tf.float32)
            
            # CRITICAL: Call the model as a direct function to prevent buffering deadlocks
            prediction = model(tensor_img, training=False).numpy()[0]
            label = emotion_labels[np.argmax(prediction)]
            
            K.clear_session()
            
            username = request.form.get('username', 'Unknown')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            emotion_history.append({
                'username': username, 'emotion': label, 'timestamp': timestamp
            })
            return f"Predicted Emotion: {label}"
        except Exception as e:
            K.clear_session()
            return f"Error during processing: {str(e)}"
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
        
        # Convert raw numpy array to tensor
        tensor_img = tf.convert_to_tensor(img, dtype=tf.float32)
        
        # CRITICAL: Run direct function call to bypass predict locks
        prediction = model(tensor_img, training=False).numpy()[0]
        label = emotion_labels[np.argmax(prediction)]
        
        K.clear_session()
        
        username = request.form.get('username', 'Unknown')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        emotion_history.append({
            'username': username, 'emotion': label, 'timestamp': timestamp
        })
        return f"Predicted Emotion: {label}"
    except Exception as e:
        K.clear_session()
        return f"Error during processing: {str(e)}"

@app.route('/view_log')
def view_log():
    return render_template('view_log.html', logs=emotion_history)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
