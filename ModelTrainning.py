import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os

# Load processed data
processed_data_folder = "processedData"
X_train = np.load(os.path.join(processed_data_folder, "X_train_v2.npy"))
y_train = np.load(os.path.join(processed_data_folder, "y_train_v2.npy"))
X_val = np.load(os.path.join(processed_data_folder, "X_val_v2.npy"))
y_val = np.load(os.path.join(processed_data_folder, "y_val_v2.npy"))

# Define the EfficientNetB2 model
base_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Add custom layers
global_avg_pooling = GlobalAveragePooling2D()(base_model.output)
output_layer = Dense(4, activation='softmax')(global_avg_pooling)
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with frozen base
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Enable mixed precision training using the updated API
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Reduce batch size to avoid OOM errors
batch_size = 16

# Update the training process to use the reduced batch size
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=batch_size, callbacks=[early_stopping])

# Fine-tune the model with the reduced batch size
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_fine_tune = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=batch_size, callbacks=[early_stopping])

# Save the model
model.save("model/trained_model_efficientnetb2.h5")