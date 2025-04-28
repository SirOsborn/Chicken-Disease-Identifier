import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pickle
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'chicken_disease_app/static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and LabelEncoder
model = load_model('model/chicken_disease_model_efficientnetb0_final_v3.h5')
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    if file:
        # Save the uploaded image
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = preprocess_image(filepath)

        # Make prediction
        pred_prob = model.predict(img, verbose=0)
        pred_class = np.argmax(pred_prob, axis=1)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        # Get prediction confidence
        confidence = float(pred_prob[0][pred_class]) * 100

        return render_template('result.html', prediction=pred_label, confidence=confidence, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)