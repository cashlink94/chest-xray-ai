import tensorflow as tf
from utils.preprocess import preprocess_image

def predict(model, image):
    processed = preprocess_image(image)
    prob = model.predict(processed)[0][0]

    if prob > 0.5:
        label = "Pneumonia"
        risk = "HIGH" if prob > 0.75 else "MODERATE"
    else:
        label = "Normal"
        risk = "LOW"

    return label, prob, risk