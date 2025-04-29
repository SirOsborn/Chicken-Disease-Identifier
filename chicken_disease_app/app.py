from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import os

# Explicitly set the template folder to 'pages'
app = Flask(__name__, template_folder='pages')

# Debug the model file path
model_path = os.path.join('model', 'chicken_disease_model_efficientnetb0_final.h5')
print(f"Attempting to load model from: {os.path.abspath(model_path)}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load the model and label encoder
model = load_model(model_path)

label_encoder_path = os.path.join('model', 'label_encoder.pkl')
print(f"Attempting to load label encoder from: {os.path.abspath(label_encoder_path)}")
if not os.path.exists(label_encoder_path):
    raise FileNotFoundError(f"Label encoder file not found at: {label_encoder_path}")

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Define the upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    # Use the pages/ directory for templates
    return render_template('index.html')  # No need for 'pages/' prefix since template_folder is set

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    if file:
        # Save the uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction)) * 100
        predicted_label = label_encoder.classes_[predicted_class]

        # Render the result page
        return render_template('result.html', prediction=predicted_label, confidence=confidence, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)