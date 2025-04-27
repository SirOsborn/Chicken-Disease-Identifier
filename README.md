# Chicken Disease Identification by Fecal Images

## Project Description and Review

This project aims to classify chicken health status (e.g., diseases or healthy) using fecal images. The pipeline includes data preprocessing, augmentation, and training a deep learning model to achieve accurate classification. Below is a detailed review of the work done:

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
├── dataset/                # Raw dataset folder
│   ├── Train/              # Training images
│   └── train_data.csv      # CSV with image labels
├── model/                  # Saved models
│   └── trained_model_efficientnetb0.h5
├── chick_disease_prediction.ipynb # working notebook
├── requirements.txt        # Dependency list
├── README.md               # Project documentation
└── LICENSE                 # License file
```