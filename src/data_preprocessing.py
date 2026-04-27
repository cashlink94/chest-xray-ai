import cv2
import numpy as np

IMG_SIZE = (224, 224)

def preprocess_image(image):
    img = np.array(image)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = img / 255.0
    img = cv2.resize(img, IMG_SIZE)

    img = np.stack([img, img, img], axis=-1)

    return np.expand_dims(img, axis=0)