import streamlit as st
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from PIL import Image
import os
import requests

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/chest_xray_best.keras"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1_JaiQe7nDQszzJb4XGJc1x0d-fbnoJE2"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Radiology AI", layout="centered")

# =========================
# DOWNLOAD MODEL
# =========================
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)

        st.info("📥 Downloading AI model (first run only)...")

        response = requests.get(MODEL_URL, stream=True)

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        st.success("✅ Model downloaded")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_ai_model():
    download_model()

    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model failed to load: {e}")
        st.stop()

model = load_ai_model()

# =========================
# PREPROCESS
# =========================
def preprocess_image(image):
    image = image.convert("L")
    image = image.resize(IMG_SIZE)

    img = np.array(image).astype("float32")
    img = img / 255.0

    # PACS-style normalization
    img = (img - np.mean(img)) / (np.std(img) + 1e-6)

    img = np.stack([img, img, img], axis=-1)
    img = np.expand_dims(img, axis=0)

    return img

# =========================
# PREDICT
# =========================
def predict(image):
    try:
        processed = preprocess_image(image)
        prob = model.predict(processed, verbose=0)[0][0]

        prob = float(prob)
        prob = min(max(prob, 0.01), 0.99)

        return prob
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return 0.0

# =========================
# UI
# =========================
st.title("🫁 Radiology AI Analysis System")
st.markdown("AI-assisted interpretation of chest X-ray images")

# =========================
# PATIENT INFO
# =========================
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name", "Anonymous")
    patient_id = st.text_input("Patient ID", "XR-001")

with col2:
    age = st.number_input("Age", min_value=0, value=40)
    gender = st.selectbox("Gender", ["Male", "Female"])

# =========================
# MODEL STATUS
# =========================
st.success("🟢 AI Model Loaded")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

# =========================
# ANALYSIS
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded X-ray", width=400)

    st.markdown("### Imaging Data")

    prob = predict(image)

    # Risk logic
    if prob > 0.75:
        risk = "HIGH"
        color = "🔴"
    elif prob > 0.5:
        risk = "MODERATE"
        color = "🟠"
    else:
        risk = "LOW"
        color = "🟢"

    # =========================
    # RESULT
    # =========================
    st.markdown("## AI Analysis Result")

    st.warning("Findings suggest radiographic features consistent with Pneumonia")

    st.markdown(f"### {color} {risk} PROBABILITY")

    st.markdown("**Confidence Score**")
    st.write(f"{prob*100:.2f}%")

    st.markdown("**Pneumonia Probability**")
    st.write(f"{prob*100:.2f}%")

    if prob < 0.5:
        st.success("No significant radiographic evidence of Pneumonia detected")

    # =========================
    # REPORT
    # =========================
    st.markdown("## Radiology Report")

    st.write(f"Patient Name: {name}")
    st.write(f"Patient ID: {patient_id}")
    st.write(f"Age / Gender: {age} / {gender}")

    st.write("Study: Chest X-ray")
    st.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.markdown("### Findings")
    st.write("AI-assisted analysis suggests features consistent with Pneumonia.")

    st.markdown("### Impression")
    st.write(f"Pneumonia Probability: {prob*100:.2f}%")
    st.write(f"Confidence Level: {prob*100:.2f}%")
    st.write(f"Risk Category: {risk}")

    st.markdown("### Recommendation")
    st.write(
        "Clinical correlation is recommended. "
        "Consider physician review and further diagnostic testing."
    )

    st.success("✔ Analysis Completed")

# =========================
# DISCLAIMER
# =========================
st.markdown("---")
st.warning(
    "⚠️ This system is NOT a medical device. "
    "It is for research and educational use only. "
    "Clinical decisions must be made by qualified professionals."
)