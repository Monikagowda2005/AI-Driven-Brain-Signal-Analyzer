import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ======================================================
# Configuration
# ======================================================

MODEL_PATH = (
    "saved_model/brain_signal_dnn.keras"
)

X_PATH = "model/X_test.npy"
Y_PATH = "model/y_test.npy"


LABELS = [
    "Rest",
    "Left Fist",
    "Right Fist",
    "Both Fists",
    "Both Feet"
]


# ======================================================
# Load Test Data
# ======================================================

print("=" * 60)
print("LOADING TEST DATASET")
print("=" * 60)

X_test = np.load(
    X_PATH
)

y_test = np.load(
    Y_PATH
)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ======================================================
# Flatten
# ======================================================

X_test = X_test.reshape(
    X_test.shape[0],
    -1
)

print(
    "Flattened X_test :",
    X_test.shape
)


# ======================================================
# Load Model
# ======================================================

print("\nLoading trained model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ======================================================
# Prediction
# ======================================================

print("\nRunning predictions...")

start_time = time.perf_counter()

probabilities = model.predict(
    X_test,
    verbose=0
)

end_time = time.perf_counter()

predictions = np.argmax(
    probabilities,
    axis=1
)

total_time = (
    end_time - start_time
)

average_time = (
    total_time / len(X_test)
)


# ======================================================
# Metrics
# ======================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


# ======================================================
# Display Metrics
# ======================================================

print("\n" + "=" * 60)
print("DNN PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy          : {accuracy * 100:.2f}%"
)

print(
    f"Precision         : {precision * 100:.2f}%"
)

print(
    f"Recall            : {recall * 100:.2f}%"
)

print(
    f"F1 Score          : {f1 * 100:.2f}%"
)

print(
    f"Total Inference   : {total_time:.4f} sec"
)

print(
    f"Per Sample        : {average_time * 1000:.4f} ms"
)

print("=" * 60)


# ======================================================
# Classification Report
# ======================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=LABELS,
        digits=4,
        zero_division=0
    )
)


# ======================================================
# Confidence
# ======================================================

confidence = np.max(
    probabilities,
    axis=1
)

print(
    f"Average Confidence : "
    f"{np.mean(confidence) * 100:.2f}%"
)


# ======================================================
# Prediction Distribution
# ======================================================

print("\nPrediction Distribution")

unique, counts = np.unique(
    predictions,
    return_counts=True
)

for u, c in zip(unique, counts):

    print(
        f"{LABELS[u]:<15} : {c}"
    )


# ======================================================
# Confusion Matrix
# ======================================================

cm = confusion_matrix(
    y_test,
    predictions
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS
)

disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

plt.title(
    "DNN Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "saved_model/confusion_matrix.png"
)

plt.show()


print("\nEvaluation completed successfully.")