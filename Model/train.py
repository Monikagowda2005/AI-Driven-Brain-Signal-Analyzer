import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from dnn_model import build_dnn_model


# ======================================================
# Configuration
# ======================================================

TRAIN_DIR = "model"
SAVE_DIR = "saved_model"

MODEL_PATH = os.path.join(
    SAVE_DIR,
    "brain_signal_dnn.keras"
)

FINAL_MODEL_PATH = os.path.join(
    SAVE_DIR,
    "final_brain_signal_model.keras"
)

os.makedirs(SAVE_DIR, exist_ok=True)


# ======================================================
# Labels
# ======================================================

LABELS = [
    "Rest",
    "Left Fist",
    "Right Fist",
    "Both Fists",
    "Both Feet"
]


# ======================================================
# Load Dataset
# ======================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

X_train = np.load(
    os.path.join(TRAIN_DIR, "X_train.npy")
)

X_test = np.load(
    os.path.join(TRAIN_DIR, "X_test.npy")
)

y_train = np.load(
    os.path.join(TRAIN_DIR, "y_train.npy")
)

y_test = np.load(
    os.path.join(TRAIN_DIR, "y_test.npy")
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print("Original X_train :", X_train.shape)


# ======================================================
# Flatten EEG
# ======================================================

X_train = X_train.reshape(
    X_train.shape[0],
    -1
)

X_test = X_test.reshape(
    X_test.shape[0],
    -1
)

print("\nFlattened EEG")

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)


# ======================================================
# Compute Class Weights
# ======================================================

classes = np.unique(
    y_train
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print("\nClass Weights")

for k, v in class_weights.items():

    label = (
        LABELS[int(k)]
        if int(k) < len(LABELS)
        else f"Class {k}"
    )

    print(
        f"{label:<15} : {v:.3f}"
    )


# ======================================================
# Build Model
# ======================================================

model = build_dnn_model(
    input_shape=X_train.shape[1]
)

model.summary()


# ======================================================
# Callbacks
# ======================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1,
    min_lr=1e-6
)


# ======================================================
# Train
# ======================================================

print("\n" + "=" * 60)
print("STARTING DNN TRAINING")
print("=" * 60)

history = model.fit(

    X_train,
    y_train,

    validation_split=0.20,

    epochs=50,

    batch_size=64,

    shuffle=True,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],

    verbose=1
)


# ======================================================
# Evaluate
# ======================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n" + "=" * 60)

print(
    f"Test Loss     : {loss:.4f}"
)

print(
    f"Test Accuracy : {accuracy * 100:.2f}%"
)

print("=" * 60)


# ======================================================
# Save Final Model
# ======================================================

model.save(
    FINAL_MODEL_PATH
)

print(
    "\nFinal model saved:"
)

print(
    FINAL_MODEL_PATH
)


# ======================================================
# Plot Training History
# ======================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "DNN Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "training_accuracy.png"
    )
)

plt.show()


# ======================================================
# Loss Graph
# ======================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "DNN Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "training_loss.png"
    )
)

plt.show()

print("\nTraining completed successfully.")