# Chicken Disease Identification by Fecal Images

## Project Description and Review

This project focuses on classifying chicken health status using fecal images, identifying four categories: Coccidiosis, Healthy, New Castle Disease, and Salmonella. The pipeline includes data loading, preprocessing, oversampling to handle class imbalance, data augmentation, training a deep learning model using transfer learning with EfficientNetB0, and evaluating the model’s performance. The final model has been deployed as a web application using Flask, allowing users to upload images and receive predictions.

## Key Achievements

    Model Performance: Achieved a test accuracy of 94% with balanced predictions across all classes (as shown in the confusion matrix).
    Deployment: Planing on deploy the model as a web app using Flask.
    Class Imbalance Handling: Addressed class imbalance through oversampling and class weights, improving prediction fairness across classes.
---

## Environment Setup and Folder Structure

### Prerequisites
- **Operating System:** Windows 11 (adaptable to macOS/Linux with minor changes).
- **Python Version:** 3.9.13 (compatible with TensorFlow 2.10.0).
- **GPU (Optional):** NVIDIA GPU for acceleration (requires CUDA setup).

### Setup Instructions
1. **Clone the Repository:**
   ```
   git clone https://github.com/SirOsborn/Chicken-Disease-Identifier.git
   cd Chicken-Disease-Identifier
   ```

2. **Set Up Virtual Environment:**
   ```
   python -m venv chick-env
   chick-env\Scripts\activate
   # On macOS/Linux: source chick-env/bin/activate
   ```

3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Verify TensorFlow Installation:**
   ```
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

5. **Set Up CUDA (Optional for GPU):**
   - Install NVIDIA drivers, CUDA Toolkit 11.2, and cuDNN 8.1.

### Updated Folder Structure
```
Chicken-Disease-Identifier/
├── dataset/                             # Raw dataset folder
│   ├── test_image/                      # For testing after the model is trained 
|   ├── Train/                           # jpg feces images of the dataset
│   └── train_data.csv                   # CSV with image labels
|      
├── model/                               # Directory for the model and LabelEncoder
│   ├── chicken_disease_model_efficientnetb0_final_v3.h5
│   └── label_encoder.pkl
│
├── chick_disease_prediction.ipynb       # Jupyter notebook for training and evaluation
├── requirements.txt                     # Dependency list for training
├── README.md                            # Project documentation
└── LICENSE                              # License file
```

---
## Model Training and Evaluation
### Data Preprocessing

- Dataset Loading: Images are loaded from the dataset/ directory, organized into subfolders for each class (Coccidiosis, Healthy, New Castle Disease, Salmonella).
- Class Imbalance: Addressed by oversampling minority classes (Healthy, New Castle Disease) to 1253 samples each, matching the majority class.
- Data Splitting: Split into training (70%), validation (15%), and test (15%) sets using stratified sampling.
- Data Augmentation: Applied transformations like rotation, flipping, and zooming to increase training data diversity.

### Model Architecture

- Base Model: Used EfficientNetB0 with pre-trained weights from ImageNet.
- Transfer Learning: Froze the first 30 layers of EfficientNetB0 and fine-tuned the remaining layers.
- Custom Layers: Added GlobalAveragePooling2D, a Dense layer with 128 units and ReLU activation (with L2 regularization), a Dropout layer (0.7), and a final Dense layer with 4 units and softmax activation.
- Optimizer: Adam with a learning rate of 1e-5.
- Loss Function: Sparse categorical crossentropy.
- Regularization: Used class weights to handle class imbalance, along with L2 regularization and dropout to prevent overfitting.

### Training

- Epochs: Trained for up to 30 epochs with early stopping (patience=5) based on validation loss.
- Batch Size: 32.
- Hardware: Utilized NVIDIA GPU for faster training (CUDA 11.8, cuDNN 8.6).

---

### Evaluation

- Test Accuracy: Achieved 94% on the test set.
- Classification Report:
```
                    precision    recall  f1-score   support

       Coccidiosis       0.99      0.95      0.97       266
           Healthy       0.82      0.89      0.85       118
New Castle Disease       0.90      0.95      0.92        58
        Salmonella       0.96      0.96      0.96       269

          accuracy                           0.94       711
         macro avg       0.92      0.94      0.93       711
      weighted avg       0.94      0.94      0.94       711
```
---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---
## Acknowledgments

- Thanks to the creators of EfficientNet and TensorFlow for providing powerful tools for deep learning.
- Inspired by the need for automated disease detection in poultry farming to improve animal health and farm productivity.

---