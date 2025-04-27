# Chicken Disease Identification by Fecal Images

## Project Description and Review

This project aims to classify chicken health status (e.g., diseases or healthy) using fecal images. The pipeline includes data preprocessing, augmentation, and training a deep learning model to achieve accurate classification. Below is a detailed review of the work done:

### Data Processing
- **Exploratory Data Analysis (EDA):**
  - Visualized class distribution and analyzed image quality (e.g., blurriness, RGB channel distribution).
  - Identified imbalances in class representation and image quality issues.

- **Preprocessing:**
  - Filtered blurry images using Laplacian variance and sharpened low-quality images.
  - Resized all images to a uniform size of 224x224 pixels.

- **Augmentation:**
  - Applied transformations like rotation, flipping, and contrast adjustments using Albumentations.

- **Balancing:**
  - Oversampled minority classes to address class imbalance, ensuring equal representation across all classes.

- **Data Splitting:**
  - Split the dataset into training, validation, and test sets with stratification to maintain class balance.

- **Saving Processed Data:**
  - Saved processed datasets in the `processedData/` directory for reuse.

### Model Training
- **Architecture:**
  - Utilized the EfficientNetB2 model for feature extraction and classification.
  - Fine-tuned the model on the processed dataset.

- **Optimization:**
  - Enabled mixed precision training for better performance and memory optimization.
  - Limited GPU memory growth to prevent crashes during training.

- **Batch Size Adjustment:**
  - Reduced batch size to 16 to avoid out-of-memory (OOM) errors.

- **Model Saving:**
  - Saved the trained model as `trained_model_efficientnetb2.h5` in the `model/` directory.

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
├── chick-env/              # Virtual environment folder
├── dataset/                # Raw dataset folder
│   ├── Train/              # Training images
│   └── train_data.csv      # CSV with image labels
├── processedData/          # Processed datasets
│   ├── X_train_v1.npy
│   ├── y_train_v1.npy
│   ├── X_val_v1.npy
│   ├── y_val_v1.npy
│   ├── X_test_v1.npy
│   ├── y_test_v1.npy
│   ├── X_train_v2.npy
│   ├── y_train_v2.npy
│   ├── X_val_v2.npy
│   ├── y_val_v2.npy
│   ├── X_test_v2.npy
│   └── y_test_v2.npy
├── model/                  # Saved models
│   ├── trained_model_efficientnetb2.h5
│   └── trained_model01.h5
├── dataProcessing.ipynb    # Data preprocessing notebook
├── modelTraining.py        # Model training script
├── testModel.py            # Model testing script
├── requirements.txt        # Dependency list
├── README.md               # Project documentation
└── LICENSE                 # License file
```

---

### Notes
- Ensure the `dataset/Train/` folder contains the raw images.
- Use the `processedData/` folder for preprocessed datasets to save time during training.
- Follow the setup instructions carefully to replicate the environment and results.