import os
import numpy as np
import mne

# Dataset path
DATASET_PATH = r"C:\Users\monik\Downloads\Brain_Signal_Analyzer\dataset\files"

# Read one EEG file
file_path = os.path.join(DATASET_PATH, "S001", "S001R01.edf")

print("Loading EEG File...")

raw = mne.io.read_raw_edf(file_path, preload=True)

# Filter EEG
raw.filter(1, 40)

print("Creating Epochs...")

# Create fixed-length epochs
epochs = mne.make_fixed_length_epochs(
    raw,
    duration=2.0,
    preload=True
)

# Convert to NumPy
epoch_data = epochs.get_data()

print("\nEpoch Shape:", epoch_data.shape)

# Save epochs
np.save("dataset/sample_epochs.npy", epoch_data)

print("Epochs saved successfully!")