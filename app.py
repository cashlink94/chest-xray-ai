import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import cv2
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/chest_xray_best.keras"
IMG_SIZE = (224, 224)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Chest X-Ray Analysis System",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------
# HEADER
# -----------------------------
st.title("🫁 Chest X-Ray Analysis System")
st.markdown(
    "AI-assisted analysis for detecting signs of **Pneumonia** from chest X-ray images."
)
st.caption(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# FIND CONV LAYER (FOR HEATMAP)
# -----------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None

# -----------------------------
# MAIN
# -----------------------------
if uploaded_file is not None:

    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Input X-ray", use_container_width=True)

    with st.spinner("Analyzing image..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)

    prob = float(prediction[0][0])
    confidence = max(prob, 1 - prob)

    # Risk logic
    if prob >= 0.75:
        risk = "High Risk"
    elif prob >= 0.5:
        risk = "Moderate Risk"
    else:
        risk = "Low Risk"

    with col2:
        st.subheader("Analysis Result")

        if prob >= 0.5:
            st.error("🩺 Pneumonia Indicators Detected")
        else:
            st.success("✅ No Significant Pneumonia Indicators")

        st.metric("Model Confidence", f"{confidence:.2%}")
        st.metric("Risk Level", risk)
        st.progress(int(confidence * 100))

        st.markdown("### Clinical Summary")
        if prob >= 0.5:
            st.write(
                f"The model detected patterns consistent with pneumonia "
                f"with a confidence of {prob:.2%}. Further clinical evaluation is recommended."
            )
        else:
            st.write(
                f"No strong indicators of pneumonia were detected "
                f"(confidence {(1 - prob):.2%}). Correlate with clinical findings."
            )

    # -----------------------------
    # PROBABILITY CHART
    # -----------------------------
    st.markdown("---")
    st.subheader("Prediction Probability Distribution")

    chart = pd.DataFrame({
        "Condition": ["Healthy", "Pneumonia"],
        "Probability": [1 - prob, prob]
    })

    st.bar_chart(chart.set_index("Condition"))

    st.write(f"Healthy Probability: {(1 - prob):.2%}")
    st.write(f"Pneumonia Probability: {prob:.2%}")

    # -----------------------------
    # HEATMAP (SAFE + OPTIONAL)
    # -----------------------------
    st.markdown("---")
    st.subheader("Model Attention (Grad-CAM)")

    last_conv_layer = find_last_conv_layer(model)

    if last_conv_layer is None:
        st.info("ℹ️ Heatmap not available for this model architecture.")
    else:
        try:
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[last_conv_layer.output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(processed)
                loss = predictions[:, 0]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = np.maximum(heatmap, 0)

            if np.max(heatmap) != 0:
                heatmap /= np.max(heatmap)

            heatmap = heatmap.numpy()

            # Overlay
            heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            original = np.array(image)
            overlay = heatmap * 0.4 + original

            col3, col4 = st.columns(2)

            with col3:
                st.image(image, caption="Original X-ray", use_container_width=True)

            with col4:
                st.image(overlay.astype("uint8"), caption="Model Attention", use_container_width=True)

        except Exception:
            st.warning("⚠️ Unable to generate heatmap for this model.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built with TensorFlow + Streamlit")

st.markdown(
    "**Notice:** This AI system is for research and educational purposes only. "
    "It is not a medical diagnostic tool. Always consult a qualified healthcare professional."
)