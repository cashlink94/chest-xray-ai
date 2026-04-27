import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = os.path.join("models", "chest_xray_model.keras")
IMG_SIZE = (128, 128)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["NORMAL", "PNEUMONIA"]

# =========================
# PREDICTION FUNCTION
# =========================
def predict_xray(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    print("\n🩻 Chest X-Ray Prediction Result")
    print("---------------------------------")
    print(f"Prediction : {class_names[predicted_index]}")
    print(f"Confidence : {confidence:.2f}")

    return class_names[predicted_index], confidence

# =========================
# RUN EXAMPLE
# =========================
if __name__ == "__main__":
    img_path = input("Enter path to X-ray image: ").strip()

    if not os.path.exists(img_path):
        print("❌ File not found. Check path and try again.")
    else:
        predict_xray(img_path)