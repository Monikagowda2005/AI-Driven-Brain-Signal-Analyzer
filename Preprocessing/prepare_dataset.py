import os
import mne
import numpy as np

# Dataset location
DATASET_PATH = r"C:\Users\monik\Downloads\Brain_Signal_Analyzer\dataset\files"

# Lists to store data
X = []
y = []

# Labels for PhysioNet runs
label_map = {
    "R01": 0,   # Rest
    "R03": 1,   # Left Fist
    "R04": 2,   # Right Fist
    "R07": 3,   # Both Fists
    "R08": 4,   # Both Feet
}

print("Reading dataset...\n")

for subject in sorted(os.listdir(DATASET_PATH)):

    subject_path = os.path.join(DATASET_PATH, subject)

    if not os.path.isdir(subject_path):
        continue

    print(f"Processing {subject}")

    for file in os.listdir(subject_path):

        if not file.endswith(".edf"):
            continue

        run = file[4:7]

        if run not in label_map:
            continue

        file_path = os.path.join(subject_path, file)

        raw = mne.io.read_raw_edf(file_path,
                                  preload=True,
                                  verbose=False)

        raw.filter(1,40,verbose=False)

        data = raw.get_data()

        data = (data - np.mean(data)) / np.std(data)

        X.append(data)

        y.append(label_map[run])

print("\nDataset Loaded Successfully!")

X = np.array(X,dtype=object)
y = np.array(y)

print("Total Samples :",len(X))
print("Labels :",len(y))

np.save("dataset/X.npy",X)
np.save("dataset/y.npy",y)

print("\nDataset Saved Successfully!")