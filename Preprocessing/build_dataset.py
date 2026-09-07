import os
import numpy as np
import mne
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\monik\Downloads\Brain_Signal_Analyzer\dataset\files"

SAVE_PATH = r"C:\Users\monik\Downloads\Brain_Signal_Analyzer\dataset"

SFREQ = 160

EPOCH_DURATION = 2.0
EXPECTED_SAMPLES = int(SFREQ * EPOCH_DURATION)

LOW_FREQ = 1.0
HIGH_FREQ = 40.0

# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "Rest",
    "Left Fist",
    "Right Fist",
    "Both Fists",
    "Both Feet"
]

REST = 0
LEFT_FIST = 1
RIGHT_FIST = 2
BOTH_FISTS = 3
BOTH_FEET = 4

# ============================================================
# RECORDING TASK MAPPING
# ============================================================

# PhysioNet EEGMMIDB task recordings used by this project.
#
# R01/R02 -> Rest
# R03/R04 -> Left Fist / Right Fist using T1/T2
# R07/R08 -> Both Fists / Both Feet using T1/T2

VALID_RUNS = {
    "R01",
    "R02",
    "R03",
    "R04",
    "R07",
    "R08"
}

# ============================================================
# STORAGE
# ============================================================

X = []
y = []

metadata = []

print("=" * 70)
print("AI BRAIN SIGNAL ANALYZER - DATASET BUILDER")
print("=" * 70)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_epoch(epoch):
    """
    Channel-wise z-score normalization.
    epoch shape = (channels, samples)
    """

    mean = np.mean(epoch, axis=1, keepdims=True)
    std = np.std(epoch, axis=1, keepdims=True)

    normalized = (epoch - mean) / (std + 1e-8)

    return normalized


# ============================================================
# QUALITY CHECK
# ============================================================

def is_good_epoch(epoch):

    # NaN / Inf check
    if not np.isfinite(epoch).all():
        return False

    # Shape check
    if epoch.shape != (64, EXPECTED_SAMPLES):
        return False

    # Completely flat channel check
    channel_std = np.std(epoch, axis=1)

    if np.any(channel_std < 1e-8):
        return False

    return True


# ============================================================
# EXTRACT EVENT EPOCH
# ============================================================

def extract_epoch(raw, event_sample):

    start_sample = int(event_sample)

    end_sample = start_sample + EXPECTED_SAMPLES

    if end_sample > raw.n_times:
        return None

    epoch = raw.get_data(
        start=start_sample,
        stop=end_sample
    )

    return epoch


# ============================================================
# GET LABEL FROM RECORDING + EVENT
# ============================================================

def get_label(run, event_name):

    # T0 = Rest
    if event_name == "T0":
        return REST

    # Runs R03/R04:
    # T1 = Left Fist
    # T2 = Right Fist
    if run in ["R03", "R04"]:

        if event_name == "T1":
            return LEFT_FIST

        if event_name == "T2":
            return RIGHT_FIST

    # Runs R07/R08:
    # T1 = Both Fists
    # T2 = Both Feet
    if run in ["R07", "R08"]:

        if event_name == "T1":
            return BOTH_FISTS

        if event_name == "T2":
            return BOTH_FEET

    return None


# ============================================================
# SUBJECT LOOP
# ============================================================

subjects = sorted(os.listdir(DATASET_PATH))

total_files = 0
total_epochs = 0
rejected_epochs = 0

for subject in subjects:

    subject_folder = os.path.join(DATASET_PATH, subject)

    if not os.path.isdir(subject_folder):
        continue

    print("\n" + "=" * 70)
    print(f"SUBJECT: {subject}")
    print("=" * 70)

    files = sorted(os.listdir(subject_folder))

    for file in files:

        if not file.lower().endswith(".edf"):
            continue

        # ----------------------------------------------------
        # Extract run
        # Example:
        # S001R03.edf -> R03
        # ----------------------------------------------------

        run = file[4:7]

        if run not in VALID_RUNS:
            continue

        path = os.path.join(subject_folder, file)

        total_files += 1

        print(f"\nProcessing: {file}")

        try:

            # =================================================
            # LOAD EDF
            # =================================================

            raw = mne.io.read_raw_edf(
                path,
                preload=True,
                verbose=False
            )

            print(
                f"Channels: {len(raw.ch_names)} | "
                f"Sampling: {raw.info['sfreq']} Hz | "
                f"Duration: {raw.times[-1]:.2f}s"
            )

            # =================================================
            # SELECT EEG CHANNELS
            # =================================================

            eeg_picks = mne.pick_types(
                raw.info,
                eeg=True,
                exclude=[]
            )

            if len(eeg_picks) < 64:

                print(
                    f"SKIPPED: only {len(eeg_picks)} EEG channels found."
                )

                continue

            # Keep first 64 EEG channels
            eeg_picks = eeg_picks[:64]

            raw.pick(eeg_picks)

            # =================================================
            # RESAMPLE IF REQUIRED
            # =================================================

            current_sfreq = raw.info["sfreq"]

            if abs(current_sfreq - SFREQ) > 0.1:

                print(
                    f"Resampling {current_sfreq} Hz -> {SFREQ} Hz"
                )

                raw.resample(
                    SFREQ,
                    npad="auto",
                    verbose=False
                )

            # =================================================
            # BAND-PASS FILTER
            # =================================================

            raw.filter(
                l_freq=LOW_FREQ,
                h_freq=HIGH_FREQ,
                verbose=False
            )

            # =================================================
            # READ EEG ANNOTATIONS
            # =================================================

            events, event_id = mne.events_from_annotations(
                raw,
                verbose=False
            )

            # Reverse event dictionary
            id_to_event = {
                value: key
                for key, value in event_id.items()
            }

            print("Events:", event_id)

            file_count = 0
            file_class_counts = Counter()

            # =================================================
            # EVENT-BASED EPOCH EXTRACTION
            # =================================================

            for event in events:

                event_sample = event[0]
                event_code = event[2]

                event_name = id_to_event.get(
                    event_code,
                    None
                )

                if event_name not in ["T0", "T1", "T2"]:
                    continue

                label = get_label(
                    run,
                    event_name
                )

                if label is None:
                    continue

                # ---------------------------------------------
                # Extract 2-second window after event
                # ---------------------------------------------

                epoch = extract_epoch(
                    raw,
                    event_sample
                )

                if epoch is None:
                    rejected_epochs += 1
                    continue

                # ---------------------------------------------
                # Check shape
                # ---------------------------------------------

                if epoch.shape != (
                    64,
                    EXPECTED_SAMPLES
                ):

                    rejected_epochs += 1
                    continue

                # ---------------------------------------------
                # Quality check
                # ---------------------------------------------

                if not is_good_epoch(epoch):

                    rejected_epochs += 1
                    continue

                # ---------------------------------------------
                # Normalize
                # ---------------------------------------------

                epoch = normalize_epoch(epoch)

                if not np.isfinite(epoch).all():

                    rejected_epochs += 1
                    continue

                # ---------------------------------------------
                # Store
                # ---------------------------------------------

                X.append(
                    epoch.astype(np.float32)
                )

                y.append(label)

                metadata.append({
                    "subject": subject,
                    "file": file,
                    "run": run,
                    "event": event_name,
                    "label": label
                })

                file_count += 1
                file_class_counts[label] += 1
                total_epochs += 1

            # =================================================
            # FILE SUMMARY
            # =================================================

            print(
                f"Valid epochs: {file_count}"
            )

            if file_class_counts:

                for class_id, count in sorted(
                    file_class_counts.items()
                ):

                    print(
                        f"  {CLASS_NAMES[class_id]}: {count}"
                    )

            else:

                print(
                    "  No valid labeled epochs found."
                )

        except Exception as e:

            print(
                f"ERROR processing {file}"
            )

            print(
                str(e)
            )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

print("\n" + "=" * 70)
print("CONVERTING DATASET")
print("=" * 70)

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int32
)

# ============================================================
# FINAL VALIDATION
# ============================================================

print("\nFinal dataset information")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

if len(X) == 0:

    raise RuntimeError(
        "\nNo valid EEG epochs were generated.\n"
        "Check EDF annotations and dataset path."
    )

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

unique, counts = np.unique(
    y,
    return_counts=True
)

distribution = dict(
    zip(unique, counts)
)

for class_id, class_name in enumerate(CLASS_NAMES):

    count = distribution.get(
        class_id,
        0
    )

    print(
        f"{class_name:<15} : {count}"
    )

# ============================================================
# SAVE DATASET
# ============================================================

os.makedirs(
    SAVE_PATH,
    exist_ok=True
)

X_path = os.path.join(
    SAVE_PATH,
    "X.npy"
)

y_path = os.path.join(
    SAVE_PATH,
    "y.npy"
)

metadata_path = os.path.join(
    SAVE_PATH,
    "metadata.npy"
)

np.save(
    X_path,
    X
)

np.save(
    y_path,
    y
)

np.save(
    metadata_path,
    np.array(
        metadata,
        dtype=object
    )
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("DATASET BUILD COMPLETE")
print("=" * 70)

print(
    f"Total EDF files processed : {total_files}"
)

print(
    f"Total valid epochs        : {total_epochs}"
)

print(
    f"Rejected epochs           : {rejected_epochs}"
)

print(
    f"X saved to                : {X_path}"
)

print(
    f"y saved to                : {y_path}"
)

print(
    f"Metadata saved to         : {metadata_path}"
)

print("\nShape expected by models:")

print(
    "(samples, 64 channels, 320 time samples)"
)

print("\nNext step:")

print(
    "Run prepare_data.py"
)

print(
    "Then train ANN, DNN and SNN."
)

print("=" * 70)