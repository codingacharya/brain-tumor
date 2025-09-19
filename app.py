import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# === CONFIG ===
MODEL_PATH = "brain_tumor_model.h5"
DATASET_FOLDER = r"C:\Users\CDU\Desktop\vishnu\BrainTumor"  # 🔁 Update this!
OUTPUT_CSV = "predictions.csv"
IMG_SIZE = (128, 128)

# === Load the model ===
model = tf.keras.models.load_model(MODEL_PATH)

def load_image_from_mat(mat_path):
    try:
        with h5py.File(mat_path, 'r') as f:
            image = np.array(f['cjdata']['image']).T  # MATLAB fix
            img = Image.fromarray(image).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img, img_array
    except Exception as e:
        print(f"⚠️ Error reading {mat_path}: {e}")
        return None, None

# === Predict on all .mat files ===
results = []

for file in os.listdir(DATASET_FOLDER):
    if file.endswith(".mat"):
        full_path = os.path.join(DATASET_FOLDER, file)
        img_vis, img_tensor = load_image_from_mat(full_path)

        if img_tensor is None:
            continue

        prediction = model.predict(img_tensor)[0][0]
        label = "Tumor" if prediction > 0.5 else "No Tumor"
        confidence = round(prediction * 100, 2) if label == "Tumor" else round((1 - prediction) * 100, 2)

        results.append({
            "filename": file,
            "prediction": label,
            "confidence (%)": confidence
        })

# === Save predictions to CSV ===
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to: {OUTPUT_CSV}")
