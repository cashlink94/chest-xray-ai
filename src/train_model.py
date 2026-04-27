import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE_1 = 5
EPOCHS_STAGE_2 = 8

TRAIN_DIR = "data/raw/train"
VAL_DIR = "data/raw/val"
MODEL_PATH = "models/chest_xray_best.keras"

AUTOTUNE = tf.data.AUTOTUNE

# =========================
# DATA PIPELINE (FAST + AUGMENTED)
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    class_names = train_ds.class_names

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


# =========================
# MODEL
# =========================
def build_model():
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


# =========================
# CLASS WEIGHTS (CRITICAL FIX)
# =========================
def get_class_weights(train_ds):
    labels = []

    for _, y in train_ds:
        labels.extend(y.numpy())

    labels = np.array(labels).flatten()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels
    )

    return {0: weights[0], 1: weights[1]}


# =========================
# TRAINING
# =========================
def train():
    train_ds, val_ds, class_names = load_data()
    model, base_model = build_model()

    class_weights = get_class_weights(train_ds)

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True),
        ReduceLROnPlateau(patience=2, factor=0.5)
    ]

    print("\n🚀 STAGE 1: Training head only\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE_1,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print("\n🔥 STAGE 2: Fine-tuning backbone\n")

    base_model.trainable = True

    for layer in base_model.layers[:200]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE_2,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print("\n✅ TRAINING COMPLETE → MODEL SAVED")


if __name__ == "__main__":
    train()