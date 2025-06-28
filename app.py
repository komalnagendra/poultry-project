from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model
model = load_model('Healthy_vs_rotten.h5')
class_names = ['Coccidiosis', 'Salmonella', 'New Castle Disease', 'Healthy']

# Create uploads folder if not exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to detect.html (e.g. triggered by "Get Start" button)
@app.route('/detect')
def detect_page():
    return render_template('detect.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence,
        'image_url': '/' + file_path  # To show image in frontend if needed
    })

if __name__ == '__main__':
    app.run(debug=True)
