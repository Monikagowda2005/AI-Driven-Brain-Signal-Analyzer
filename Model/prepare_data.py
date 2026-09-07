import os
import numpy as np
from sklearn.model_selection import train_test_split

# ======================================================
# Configuration
# ======================================================

DATASET_DIR = "dataset"
OUTPUT_DIR = "model"

X_PATH = os.path.join(DATASET_DIR, "X.npy")
Y_PATH = os.path.join(DATASET_DIR, "y.npy")

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

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Original Dataset")
print("X :", X.shape)
print("y :", y.shape)

# ======================================================
# Validate Shape
# ======================================================

if X.ndim != 3:
    raise ValueError(
        f"Expected X to have 3 dimensions "
        f"(samples, channels, samples), got {X.shape}"
    )

print("\nEEG Shape")
print("Channels :", X.shape[1])
print("Samples  :", X.shape[2])

# ======================================================
# Remove Invalid Samples
# ======================================================

mask = ~np.isnan(X).any(axis=(1, 2))

removed = np.sum(~mask)

X = X[mask]
y = y[mask]

print("\nInvalid Samples Removed :", removed)

print("Clean X :", X.shape)
print("Clean y :", y.shape)

# ======================================================
# Shuffle Dataset
# ======================================================

np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# ======================================================
# Train / Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ======================================================
# Create Output Directory
# ======================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# Save Dataset
# ======================================================

np.save(
    os.path.join(OUTPUT_DIR, "X_train.npy"),
    X_train
)

np.save(
    os.path.join(OUTPUT_DIR, "X_test.npy"),
    X_test
)

np.save(
    os.path.join(OUTPUT_DIR, "y_train.npy"),
    y_train
)

np.save(
    os.path.join(OUTPUT_DIR, "y_test.npy"),
    y_test
)

# ======================================================
# Display Information
# ======================================================

print("\n" + "=" * 60)
print("DATASET PREPARED SUCCESSFULLY")
print("=" * 60)

print("\nTraining Set")
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("\nTesting Set")
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# ======================================================
# Training Distribution
# ======================================================

print("\nTraining Distribution")

classes, counts = np.unique(
    y_train,
    return_counts=True
)

for c, n in zip(classes, counts):

    if int(c) < len(LABELS):
        label = LABELS[int(c)]
    else:
        label = f"Class {c}"

    print(f"{label:<15} : {n}")

# ======================================================
# Testing Distribution
# ======================================================

print("\nTesting Distribution")

classes, counts = np.unique(
    y_test,
    return_counts=True
)

for c, n in zip(classes, counts):

    if int(c) < len(LABELS):
        label = LABELS[int(c)]
    else:
        label = f"Class {c}"

    print(f"{label:<15} : {n}")

print("\nFiles Saved:")
print("model/X_train.npy")
print("model/X_test.npy")
print("model/y_train.npy")
print("model/y_test.npy")

print("\nDone!")