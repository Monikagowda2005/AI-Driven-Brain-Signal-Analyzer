import os
import mne

# Dataset path
DATASET_PATH = r"C:\Users\monik\Downloads\Brain_Signal_Analyzer\dataset\files"

# Select one subject and one recording
subject = "S001"
recording = "S001R01.edf"

# Complete file path
file_path = os.path.join(DATASET_PATH, subject, recording)

print("=" * 50)
print("Loading EEG File...")
print("=" * 50)

# Read EDF file
raw = mne.io.read_raw_edf(file_path, preload=True)

print("\n✅ EEG File Loaded Successfully!\n")

# -------------------------
# Dataset Information
# -------------------------

print("Dataset Information")
print("-" * 30)

print("File Name :", recording)
print("Subject   :", subject)

print("Number of Channels :", len(raw.ch_names))
print("Channel Names       :", raw.ch_names)

print("Sampling Frequency  :", raw.info['sfreq'], "Hz")

print("Recording Duration  :", raw.times[-1], "seconds")

print("Total Samples       :", raw.n_times)

print("Data Shape          :", raw.get_data().shape)

print("=" * 50)