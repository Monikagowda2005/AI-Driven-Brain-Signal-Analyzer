import os
import numpy as np
import mne

from tensorflow.keras.models import load_model


# ==========================================================
# Project Paths
# ==========================================================

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "saved_model",
    "brain_signal_dnn.keras"
)


# ==========================================================
# Load Model
# ==========================================================

model = load_model(
    MODEL_PATH
)


# ==========================================================
# Labels
# ==========================================================

CLASS_NAMES = [
    "Rest",
    "Left Fist",
    "Right Fist",
    "Both Fists",
    "Both Feet"
]


# ==========================================================
# Signal Quality
# ==========================================================

def calculate_signal_quality(raw):

    data = raw.get_data()

    std = float(
        np.std(data)
    )

    peak = float(
        np.max(np.abs(data))
    )

    # Basic quality score
    if std < 20:

        quality = "Excellent"
        score = 95
        noise = "Very Low"

    elif std < 50:

        quality = "Good"
        score = 82
        noise = "Low"

    elif std < 100:

        quality = "Moderate"
        score = 65
        noise = "Medium"

    else:

        quality = "Poor"
        score = 45
        noise = "High"

    return {
        "score": score,
        "quality": quality,
        "noise": noise,
        "standard_deviation": std,
        "peak_amplitude": peak
    }


# ==========================================================
# Normalize Epoch
# ==========================================================

def normalize_epoch(epoch):

    mean = epoch.mean(
        axis=1,
        keepdims=True
    )

    std = epoch.std(
        axis=1,
        keepdims=True
    )

    return (
        epoch -
        mean
    ) / (
        std + 1e-8
    )


# ==========================================================
# Prediction Stability
# ==========================================================

def calculate_stability(predictions):

    if len(predictions) == 0:

        return 0.0

    predictions = np.asarray(
        predictions
    )

    counts = np.bincount(
        predictions,
        minlength=len(CLASS_NAMES)
    )

    majority = np.max(
        counts
    )

    stability = (
        majority /
        len(predictions)
    )

    return float(
        stability * 100
    )


# ==========================================================
# Main Processing Function
# ==========================================================

def process_edf(edf_path):

    try:

        # --------------------------------------------------
        # Read EDF
        # --------------------------------------------------

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            verbose=False
        )

        # --------------------------------------------------
        # Signal Quality
        # --------------------------------------------------

        quality_info = (
            calculate_signal_quality(raw)
        )

        # --------------------------------------------------
        # Filtering
        # --------------------------------------------------

        raw.filter(
            l_freq=1,
            h_freq=40,
            verbose=False
        )

        # --------------------------------------------------
        # Create 2-second epochs
        # --------------------------------------------------

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=2,
            overlap=0,
            preload=True,
            reject_by_annotation=True,
            verbose=False
        )

        data = epochs.get_data()

        predictions = []

        probabilities_all = []

        confidences = []

        processing_times = []

        valid_epoch_indices = []

        # --------------------------------------------------
        # Predict Each Epoch
        # --------------------------------------------------

        for index, epoch in enumerate(data):

            # Current model expects:
            # 64 channels × 320 samples

            if epoch.shape != (
                64,
                320
            ):

                continue

            start = (
                __import__(
                    "time"
                ).perf_counter()
            )

            # Normalize
            epoch = normalize_epoch(
                epoch
            )

            # Flatten
            sample = epoch.reshape(
                1,
                -1
            )

            # Predict
            prob = model.predict(
                sample,
                verbose=0
            )[0]

            elapsed = (
                __import__(
                    "time"
                ).perf_counter()
                - start
            )

            cls = int(
                np.argmax(prob)
            )

            confidence = float(
                np.max(prob)
            )

            predictions.append(
                cls
            )

            probabilities_all.append(
                prob
            )

            confidences.append(
                confidence
            )

            processing_times.append(
                elapsed
            )

            valid_epoch_indices.append(
                index
            )

        # --------------------------------------------------
        # No Valid Epochs
        # --------------------------------------------------

        if len(predictions) == 0:

            return None

        # --------------------------------------------------
        # Convert Arrays
        # --------------------------------------------------

        predictions = np.asarray(
            predictions
        )

        probabilities_all = np.asarray(
            probabilities_all
        )

        # --------------------------------------------------
        # Majority Voting
        # --------------------------------------------------

        counts = np.bincount(
            predictions,
            minlength=len(CLASS_NAMES)
        )

        final_class = int(
            np.argmax(counts)
        )

        # --------------------------------------------------
        # Average Probability
        # --------------------------------------------------

        average_probability = (
            np.mean(
                probabilities_all,
                axis=0
            )
        )

        confidence = float(
            average_probability[
                final_class
            ] * 100
        )

        # --------------------------------------------------
        # Stability
        # --------------------------------------------------

        stability = (
            calculate_stability(
                predictions
            )
        )

        # --------------------------------------------------
        # Prediction Names
        # --------------------------------------------------

        epoch_names = [
            CLASS_NAMES[i]
            for i in predictions
        ]

        # --------------------------------------------------
        # Processing Latency
        # --------------------------------------------------

        average_latency = (
            np.mean(
                processing_times
            ) * 1000
        )

        # --------------------------------------------------
        # Assistive Intent
        # --------------------------------------------------

        assistive_commands = {

            "Rest": "NO ACTION",

            "Left Fist": "LEFT",

            "Right Fist": "RIGHT",

            "Both Fists": "SELECT",

            "Both Feet": "CONFIRM"

        }

        intent = assistive_commands[
            CLASS_NAMES[final_class]
        ]

        # --------------------------------------------------
        # Return Results
        # --------------------------------------------------

        return {

            "prediction":
                CLASS_NAMES[final_class],

            "confidence":
                confidence,

            "probabilities":
                average_probability,

            "raw":
                raw,

            "epochs":
                epochs,

            "epoch_names":
                epoch_names,

            "epoch_predictions":
                predictions,

            "epoch_confidences":
                np.asarray(confidences),

            "prediction_counts":
                counts,

            "stability":
                stability,

            "signal_quality":
                quality_info,

            "average_latency_ms":
                average_latency,

            "intent":
                intent,

            "valid_epochs":
                len(predictions)

        }

    except Exception as e:

        print(
            "EEG Processing Error:",
            e
        )

        return None