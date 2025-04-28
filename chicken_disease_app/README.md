## Chicken Disease Prediction Web App
### Overview

This directory contains a Flask web application for deploying a pre-trained deep learning model that predicts chicken health status (Coccidiosis, Healthy, New Castle Disease, Salmonella) based on fecal images. The app allows users to upload an image via a web interface and receive a prediction along with a confidence score.
Key Features
- Image Upload: Users can upload an image of chicken feces through a simple web interface.
- Prediction: The app uses a pre-trained EfficientNetB0 model to predict the chicken’s health status.
- Confidence Score: Displays the confidence level of the prediction.
- Deployment: Supports local deployment

---

## Instructions
### Directory Structure
```
chicken_disease_app/
├── app.py                           # Main Flask application
├── static/                          # Static files (CSS, uploaded images)
│   ├── css/
│   │   └── style.css                # CSS for styling the web app
│   └── uploads/                     # Directory to store uploaded images
├── pages/                           # HTML templates
│   ├── index.html                   # Home page for uploading images
│   └── result.html                  # Page to display prediction results
└── README.md                        # This documentation
```
---
1. Navigate to the Directory
```
cd ../chicken_disease_app
```
### Running the App Locally
1. Start the Flask Server
With the virtual environment activated, run:
```
python app.py
```
2. Access the App
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
---
### Usage
- Upload an Image: On the home page, click "Choose File" to select an image of chicken feces (e.g., JPG, PNG).
- Submit: Click the "Predict" button to upload the image and get a prediction.
- View Results: The app will display the predicted health status (e.g., "Coccidiosis") and a confidence score (e.g., 95.23%). The uploaded image will also be shown.
---