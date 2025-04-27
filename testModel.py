import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import albumentations as A
from tensorflow.keras.utils import Sequence
import numpy as np
import os

# Define augmentation pipelines
augmentation_version_1 = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.Blur(blur_limit=3, p=0.2),
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

augmentation_version_2 = A.Compose([
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.Transpose(p=1.0),
    A.Affine(rotate=(-180, 180), translate_percent={'x': (-0.3, 0.3), 'y': (-0.3, 0.3)}, scale=(0.7, 1.3), p=1.0),
    A.Blur(blur_limit=7, p=1.0),
    A.Resize(224, 224, p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Custom Data Generator
class AlbumentationsDataGenerator(Sequence):
    def __init__(self, images, labels, transform=None, batch_size=32, shuffle=True):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        augmented_images = [self.transform(image=img.copy())['image'] if self.transform else img for img in batch_images]
        return np.array(augmented_images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.images, self.labels))
            np.random.shuffle(temp)
            self.images, self.labels = zip(*temp)
            self.images = np.array(self.images)
            self.labels = np.array(self.labels)

# Define a simple CNN model
def create_model(input_shape=(224, 224, 3), num_classes=4):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the model for each augmentation version
def train_and_evaluate(train_gen, val_gen, test_gen, version_name):
    model = create_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[early_stopping])
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Version {version_name} - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return history, test_loss, test_accuracy

# Load processed data for augmentation version 1
processed_data_folder = "processedData"
X_train_v1 = np.load(os.path.join(processed_data_folder, "X_train_v1.npy"))
y_train_v1 = np.load(os.path.join(processed_data_folder, "y_train_v1.npy"))
X_val_v1 = np.load(os.path.join(processed_data_folder, "X_val_v1.npy"))
y_val_v1 = np.load(os.path.join(processed_data_folder, "y_val_v1.npy"))
X_test_v1 = np.load(os.path.join(processed_data_folder, "X_test_v1.npy"))
y_test_v1 = np.load(os.path.join(processed_data_folder, "y_test_v1.npy"))

# Load processed data for augmentation version 2
X_train_v2 = np.load(os.path.join(processed_data_folder, "X_train_v2.npy"))
y_train_v2 = np.load(os.path.join(processed_data_folder, "y_train_v2.npy"))
X_val_v2 = np.load(os.path.join(processed_data_folder, "X_val_v2.npy"))
y_val_v2 = np.load(os.path.join(processed_data_folder, "y_val_v2.npy"))
X_test_v2 = np.load(os.path.join(processed_data_folder, "X_test_v2.npy"))
y_test_v2 = np.load(os.path.join(processed_data_folder, "y_test_v2.npy"))

# Create data generators for both augmentation versions
batch_size = 32
train_generator_v1 = AlbumentationsDataGenerator(X_train_v1, y_train_v1, transform=None, batch_size=batch_size)
val_generator_v1 = AlbumentationsDataGenerator(X_val_v1, y_val_v1, transform=None, batch_size=batch_size, shuffle=False)
test_generator_v1 = AlbumentationsDataGenerator(X_test_v1, y_test_v1, transform=None, batch_size=batch_size, shuffle=False)

train_generator_v2 = AlbumentationsDataGenerator(X_train_v2, y_train_v2, transform=None, batch_size=batch_size)
val_generator_v2 = AlbumentationsDataGenerator(X_val_v2, y_val_v2, transform=None, batch_size=batch_size, shuffle=False)
test_generator_v2 = AlbumentationsDataGenerator(X_test_v2, y_test_v2, transform=None, batch_size=batch_size, shuffle=False)

# Train with augmentation version 1
history_v1, test_loss_v1, test_accuracy_v1 = train_and_evaluate(train_generator_v1, val_generator_v1, test_generator_v1, "Version 1")

# Train with augmentation version 2
history_v2, test_loss_v2, test_accuracy_v2 = train_and_evaluate(train_generator_v2, val_generator_v2, test_generator_v2, "Version 2")

# Compare results
print("Comparison of Augmentation Versions:")
print(f"Version 1 - Test Accuracy: {test_accuracy_v1}")
print(f"Version 2 - Test Accuracy: {test_accuracy_v2}")