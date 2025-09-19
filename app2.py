import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Brain Tumor Dataset Explorer")

# === Upload Dataset ===
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # === Basic Statistics ===
    st.subheader("📈 Dataset Statistics")
    st.write(df.describe(include="all"))

    # === Class Distribution ===
    if "Tumor" in df.columns:  # assumes "Tumor" column is target
        st.subheader("⚖️ Class Distribution")
        tumor_counts = df["Tumor"].value_counts()

        fig1, ax1 = plt.subplots()
        tumor_counts.plot(kind="bar", color=["#e74c3c", "#2ecc71"], ax=ax1)
        ax1.set_title("Tumor vs No Tumor Count")
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        # Pie Chart
        fig2, ax2 = plt.subplots()
        tumor_counts.plot.pie(autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"], ax=ax2)
        ax2.set_ylabel("")
        ax2.set_title("Class Distribution (Pie)")
        st.pyplot(fig2)

    # === Correlation Heatmap ===
    st.subheader("📊 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        fig3, ax3 = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("No numeric features available for correlation heatmap.")

    # === Feature Distributions ===
    st.subheader("📊 Feature Distributions")
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, bins=20, ax=ax, color="blue")
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # === Download processed dataset ===
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download Processed Dataset", csv, "processed_dataset.csv", "text/csv")

else:
    st.info("👆 Please upload a dataset (CSV) to explore statistics and visualizations.")
