import os
import mne

# Dataset path
DATASET_PATH = r"C:\Users\monik\Downloads\Brain_Signal_Analyzer\dataset\files"

subject = "S001"
recording = "S001R01.edf"

file_path = os.path.join(DATASET_PATH, subject, recording)

print("Loading EEG data...")

# Read EDF file
raw = mne.io.read_raw_edf(file_path, preload=True)

print("EEG Loaded Successfully!")

# Display basic information
print("\nOriginal Data")
print(raw)

# Plot first 10 channels
raw.plot(n_channels=10, duration=5, title="Raw EEG Signal")

# Apply Band-pass Filter
print("\nApplying Band-pass Filter (1-40 Hz)...")

raw.filter(l_freq=1.0, h_freq=40.0)

print("Filtering Completed!")

# Plot filtered EEG
raw.plot(n_channels=10, duration=5, title="Filtered EEG Signal")

import numpy as np

# Get EEG data as NumPy array
data = raw.get_data()

# Normalize the data
data = (data - np.mean(data)) / np.std(data)

print("\nData Normalized Successfully!")

print("Shape:", data.shape)

np.save("dataset/preprocessed_eeg.npy", data)

print("Preprocessed data saved successfully!")