import streamlit as st
import numpy as np
from PIL import Image
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === Load pre-trained Keras model ===
@st.cache_resource
def load_brain_model():
    return load_model("brain_tumor_model.h5")

model = load_brain_model()

# === Helper: Process image ===
def preprocess_image(img_pil):
    img_pil = img_pil.resize((128, 128))
    img = img_to_array(img_pil) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# === Helper: Handle .mat file ===
def mat_to_image(uploaded_mat):
    try:
        with h5py.File(uploaded_mat, 'r') as f:
            image_data = np.array(f['cjdata']['image']).T

            # Normalize
            image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
            image_data = (image_data * 255).astype(np.uint8)

            image = Image.fromarray(image_data).convert("RGB")
            return image
    except Exception as e:
        st.error(f"Failed to extract image from .mat: {e}")
        return None

# === Title & Info ===
st.title("🧠 Brain Tumor Detection")
st.write("Upload MRI images (`.jpg`, `.png`, `.mat`) to check for brain tumor and view analytics.")

# === Session storage for results ===
if "results" not in st.session_state:
    st.session_state.results = []

# === Upload multiple files ===
uploaded_files = st.file_uploader("Upload MRI images", type=["jpg", "jpeg", "png", "mat"], accept_multiple_files=True)

if uploaded_files:
    threshold = 0.6  # decision threshold

    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split('.')[-1].lower()

        # Extract image
        if file_type == "mat":
            image = mat_to_image(uploaded_file)
        else:
            image = Image.open(uploaded_file).convert("RGB")

        if image is not None:
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

            # Predict
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0][0]

            tumor_prob = float(prediction)
            no_tumor_prob = 1 - tumor_prob
            label = "🧠 Brain Tumor Detected" if tumor_prob >= threshold else "✅ No Tumor Detected"

            # Save result
            st.session_state.results.append({
                "File": uploaded_file.name,
                "Prediction": label,
                "Tumor Probability": tumor_prob * 100,
                "No Tumor Probability": no_tumor_prob * 100
            })

# === Show Statistics and Charts ===
if st.session_state.results:
    df = pd.DataFrame(st.session_state.results)

    st.subheader("📊 Prediction Results")
    st.dataframe(df)

    # Download results
    csv = df.to_csv(index=False).encode()
    st.download_button("📄 Download Full Report (CSV)", csv, "tumor_report.csv", "text/csv")

    # === Statistics ===
    st.subheader("📈 Statistics")
    tumor_count = (df["Prediction"] == "🧠 Brain Tumor Detected").sum()
    no_tumor_count = (df["Prediction"] == "✅ No Tumor Detected").sum()
    avg_conf = df["Tumor Probability"].mean()

    st.write(f"Total Images: {len(df)}")
    st.write(f"Tumor Cases: {tumor_count}")
    st.write(f"No Tumor Cases: {no_tumor_count}")
    st.write(f"Average Tumor Probability: {avg_conf:.2f}%")

    # === Charts ===
    st.subheader("📊 Visualizations")

    # Pie Chart
    fig1, ax1 = plt.subplots()
    ax1.pie([tumor_count, no_tumor_count],
            labels=["Tumor", "No Tumor"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#e74c3c", "#2ecc71"])
    ax1.axis("equal")
    st.pyplot(fig1)

    # Bar Chart - Predictions
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Prediction", data=df, palette="Set2", ax=ax2)
    st.pyplot(fig2)

    # Histogram - Tumor Probability
    fig3, ax3 = plt.subplots()
    sns.histplot(df["Tumor Probability"], bins=10, kde=True, color="blue", ax=ax3)
    ax3.set_title("Tumor Probability Distribution")
    ax3.set_xlabel("Tumor Probability (%)")
    st.pyplot(fig3)

    # Confidence Scatter Plot
    fig4, ax4 = plt.subplots()
    ax4.scatter(range(len(df)), df["Tumor Probability"], c="red", label="Tumor Probability")
    ax4.scatter(range(len(df)), df["No Tumor Probability"], c="green", label="No Tumor Probability")
    ax4.set_title("Confidence per Image")
    ax4.set_xlabel("Image Index")
    ax4.set_ylabel("Probability (%)")
    ax4.legend()
    st.pyplot(fig4)

# === Sidebar Info ===
st.sidebar.title("🧬 About")
st.sidebar.info("""
This app uses a deep learning CNN model trained on brain MRI images to detect tumors.  

**Features added:**  
- Upload multiple MRI images  
- Normalization for `.mat` files  
- Prediction statistics  
- Advanced visualizations: pie chart, bar chart, histogram, scatter plot  
- Downloadable CSV report  

**Model**: `brain_tumor_model.h5`  
**Input Size**: `128x128x3`  
""")
