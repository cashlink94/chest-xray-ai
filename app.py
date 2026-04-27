import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime
import os
import gdown

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/chest_xray_best.keras"
MODEL_URL = "https://drive.google.com/uc?id=1_JaiQe7nDQszzJb4XGJc1x0d-fbnoJE2"
IMG_SIZE = (224, 224)

# =========================
# DOWNLOAD MODEL (IMPORTANT)
# =========================
def download_model():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# PREDICTION
# =========================
def predict(image, model):
    processed = preprocess_image(image)
    prob = model.predict(processed)[0][0]
    return float(prob)

# =========================
# UI
# =========================
st.title("🫁 Radiology AI Analysis System")
st.write("AI-assisted interpretation of chest X-ray images")

# =========================
# PATIENT INFO
# =========================
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name", "Anonymous")
    patient_id = st.text_input("Patient ID", "XR-001")

with col2:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    gender = st.selectbox("Gender", ["Male", "Female"])

# =========================
# LOAD MODEL
# =========================
model = load_model()
st.success("🟢 AI Model Loaded")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION FLOW
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded X-ray", width=300)

    probability = predict(image, model)
    confidence = probability * 100

    # =========================
    # RISK LEVEL
    # =========================
    if confidence >= 75:
        risk = "🔴 HIGH PROBABILITY"
    elif confidence >= 50:
        risk = "🟠 MODERATE PROBABILITY"
    else:
        risk = "🟢 LOW PROBABILITY"

    # =========================
    # RESULT
    # =========================
    st.subheader("AI Analysis Result")

    st.warning("Findings suggest radiographic features consistent with Pneumonia")

    st.markdown(f"### {risk}")
    st.write("Confidence Score")
    st.write(f"{confidence:.2f}%")

    st.write("Pneumonia Probability")
    st.write(f"{confidence:.2f}%")

    # =========================
    # REPORT
    # =========================
    st.subheader("Radiology Report")

    st.write(f"Patient Name: {name}")
    st.write(f"Patient ID: {patient_id}")
    st.write(f"Age / Gender: {age} / {gender}")
    st.write("Study: Chest X-ray")
    st.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.write("Findings")
    st.write("AI-assisted analysis suggests features consistent with Pneumonia.")

    st.write("Impression")
    st.write(f"Pneumonia Probability: {confidence:.2f}%")
    st.write(f"Confidence Level: {confidence:.2f}%")
    st.write(f"Risk Category: {risk.split()[1]}")

    st.write("Recommendation")
    st.write("Clinical correlation is recommended. Consider physician review and further diagnostic testing.")

    st.success("✔ Analysis Completed")

# =========================
# DISCLAIMER
# =========================
st.warning(
    "⚠️ This system is NOT a medical device. "
    "It is for research and educational use only. "
    "Clinical decisions must be made by qualified professionals."
)