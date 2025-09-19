import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# === USER CONFIG ===
# Replace this with your actual dataset path
DATA_DIR = r'C:\Users\CDU\Desktop\vishnu\BrainTumor'
IMG_SIZE = 128
EPOCHS = 10

def load_data(data_dir, img_size=128):
    X = []
    y = []

    for label in ['yes', 'no']:
        path = os.path.join(data_dir, label)
        class_num = 1 if label == 'yes' else 0

        if not os.path.exists(path):
            print(f"❌ Folder not found: {path}")
            continue

        for img_file in os.listdir(path):
            try:
                img_path = os.path.join(path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size))
                X.append(np.array(img))
                y.append(class_num)
            except Exception as e:
                print(f"⚠️ Skipped file: {img_file} due to {e}")
    
    return np.array(X), np.array(y)

# === Load Data ===
print("🔄 Loading data...")
X, y = load_data(DATA_DIR, IMG_SIZE)
X = X / 255.0  # Normalize pixel values

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Data Augmentation ===
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

# === Build CNN Model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# === Train Model ===
print("🚀 Training model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=EPOCHS
)

# === Save Model ===
model.save("brain_tumor_model.h5")
print("✅ Model saved as 'brain_tumor_model.h5'")

# === Plot Training History ===
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
