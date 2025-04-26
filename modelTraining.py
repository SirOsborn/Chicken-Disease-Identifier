import numpy as np
import os

# Define the folder where the processed data is stored
processed_data_folder = "processedData"

# Load each dataset from the folder
X_train = np.load(os.path.join(processed_data_folder, "X_train.npy"))
y_train = np.load(os.path.join(processed_data_folder, "y_train.npy"))
X_val = np.load(os.path.join(processed_data_folder, "X_val.npy"))
y_val = np.load(os.path.join(processed_data_folder, "y_val.npy"))
X_test = np.load(os.path.join(processed_data_folder, "X_test.npy"))
y_test = np.load(os.path.join(processed_data_folder, "y_test.npy"))

print("Datasets loaded successfully from the processedData folder!")

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")