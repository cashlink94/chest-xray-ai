import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ======================
# CONFIG
# ======================
MODEL_PATH = "models/chest_xray_model.keras"
DATA_DIR = "data/raw/val"

IMG_SIZE = (128, 128)   # MUST match training
BATCH_SIZE = 32

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH)

# ======================
# LOAD DATA
# ======================
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_ds.class_names

# Normalize (same as training style)
normalization = tf.keras.layers.Rescaling(1./255)
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

# ======================
# PREDICTIONS
# ======================
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ======================
# CLASSIFICATION REPORT
# ======================
print("\n📊 CLASSIFICATION REPORT\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")

os.makedirs("outputs/evaluation", exist_ok=True)
plt.title("Chest X-Ray Confusion Matrix")
plt.savefig("outputs/evaluation/confusion_matrix.png")
plt.close()

print("\n✅ Saved: outputs/evaluation/confusion_matrix.png")