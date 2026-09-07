import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# ==========================================================
# PROJECT PATH
# ==========================================================

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


# ==========================================================
# IMPORT PROJECT MODULES
# ==========================================================

from eeg_processing import process_edf

from visualization import (
    plot_eeg_signal,
    probability_chart,
    confidence_gauge,
    prediction_distribution
)

from report_generator import generate_report

from signal_analysis import (
    frequency_band_analysis,
    plot_psd,
    eeg_heatmap
)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Brain Signal Analyzer",
    page_icon="🧠",
    layout="wide"
)


# ==========================================================
# LOAD CSS
# ==========================================================

css_path = os.path.join(
    os.path.dirname(__file__),
    "style.css"
)

if os.path.exists(css_path):

    with open(
        css_path,
        "r",
        encoding="utf-8"
    ) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# ==========================================================
# CONSTANTS
# ==========================================================

CLASS_NAMES = [
    "Rest",
    "Left Fist",
    "Right Fist",
    "Both Fists",
    "Both Feet"
]


CLINICAL_INTERPRETATION = {

    "Rest":
        "Resting EEG activity detected.",

    "Left Fist":
        "Motor imagery associated with left-hand movement detected.",

    "Right Fist":
        "Motor imagery associated with right-hand movement detected.",

    "Both Fists":
        "Motor imagery associated with bilateral upper-limb activity detected.",

    "Both Feet":
        "Motor imagery associated with bilateral lower-limb activity detected."
}


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        "<h1 style='text-align:center;'>🧠</h1>",
        unsafe_allow_html=True
    )

    st.title(
        "AI Brain Signal Analyzer"
    )

    st.markdown("---")

    st.success(
        "✔ DNN Model Ready"
    )

    st.info(
        """
### Dataset
PhysioNet EEG Motor Movement Dataset

### Model
Deep Neural Network

### Framework
TensorFlow + MNE + Streamlit

### Input
EDF EEG Recording

### Version
v2.0
"""
    )

    st.markdown("---")

    st.subheader(
        "👨‍💻 Project Team"
    )

    st.markdown(
        """
**Team Members**

- 👩 B N Monika
- 👩 Prarthana P Nayak
- 👩 Mariam

**Department:** CSE (Data Science)

**Institution:**  
Dayananda Sagar Academy of Technology and Management
"""
    )


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "🧠 AI Driven Brain Signal Analyzer"
)

st.markdown(
    """
### Assistive Human–Computer Interaction using Deep Neural Networks

Analyze EEG recordings, classify motor imagery activity,
measure prediction confidence and stability, and generate
assistive interaction intents.
"""
)


# ==========================================================
# HEADER METRICS
# ==========================================================

header1, header2, header3, header4 = st.columns(4)


with header1:

    st.metric(
        label="🧠 AI Model",
        value="DNN"
    )


with header2:

    st.metric(
        label="📚 Dataset",
        value="PhysioNet"
    )


with header3:

    st.metric(
        label="📂 Input",
        value="EDF"
    )


with header4:

    st.metric(
        label="🟢 System",
        value="READY"
    )


st.markdown("---")


# ==========================================================
# FILE UPLOAD
# ==========================================================

st.subheader(
    "📂 Upload EEG Recording"
)

uploaded_file = st.file_uploader(
    "Upload an EDF EEG file",
    type=["edf"],
    help="Upload an EEG recording in EDF format."
)


# ==========================================================
# PROCESS FILE
# ==========================================================

if uploaded_file is not None:

    temp_file = None

    try:

        # --------------------------------------------------
        # Save uploaded file
        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".edf"
        ) as tmp:

            tmp.write(
                uploaded_file.getbuffer()
            )

            temp_file = tmp.name


        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        progress = st.progress(0)

        status = st.empty()


        status.info(
            "📂 Loading EEG recording..."
        )

        progress.progress(10)

        time.sleep(0.2)


        status.info(
            "🧹 Preprocessing EEG signal..."
        )

        progress.progress(30)

        time.sleep(0.2)


        status.info(
            "📊 Filtering EEG signal..."
        )

        progress.progress(50)

        time.sleep(0.2)


        status.info(
            "🧠 Running DNN prediction..."
        )

        progress.progress(75)


        # --------------------------------------------------
        # MAIN PROCESSING
        # --------------------------------------------------

        result = process_edf(
            temp_file
        )


        progress.progress(100)


        if result is None:

            status.error(
                "❌ Unable to process the EEG file."
            )

            st.error(
                """
The uploaded EDF could not be processed.

Possible reasons:
- Unsupported channel configuration
- Invalid EEG data
- Insufficient samples
- Model input mismatch
"""
            )

        else:

            status.success(
                "✅ EEG analysis completed"
            )


            # ==================================================
            # EXTRACT RESULT
            # ==================================================

            prediction = result[
                "prediction"
            ]

            confidence = result[
                "confidence"
            ]

            probabilities = result[
                "probabilities"
            ]

            raw = result[
                "raw"
            ]

            epochs = result[
                "epochs"
            ]

            epoch_names = result[
                "epoch_names"
            ]

            epoch_predictions = result[
                "epoch_predictions"
            ]

            stability = result[
                "stability"
            ]

            quality_info = result[
                "signal_quality"
            ]

            average_latency = result[
                "average_latency_ms"
            ]

            intent = result[
                "intent"
            ]

            valid_epochs = result[
                "valid_epochs"
            ]


            # ==================================================
            # SIGNAL QUALITY VALUES
            # ==================================================

            quality_score = quality_info[
                "score"
            ]

            quality = quality_info[
                "quality"
            ]

            noise = quality_info[
                "noise"
            ]

            std_value = quality_info[
                "standard_deviation"
            ]

            peak_value = quality_info[
                "peak_amplitude"
            ]


            # ==================================================
            # RECORDING INFORMATION
            # ==================================================

            subject_name = uploaded_file.name

            recording_length = (
                raw.times[-1]
                if len(raw.times) > 0
                else 0
            )

            sampling_rate = float(
                raw.info["sfreq"]
            )

            channels = len(
                raw.ch_names
            )

            epochs_count = len(
                epoch_predictions
            )


            # ==================================================
            # TOP RESULT CARDS
            # ==================================================

            st.markdown("---")

            st.subheader(
                "🎯 AI Prediction"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Prediction",
                    prediction
                )


            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


            with col3:

                st.metric(
                    "Stability",
                    f"{stability:.2f}%"
                )


            with col4:

                st.metric(
                    "Valid Epochs",
                    valid_epochs
                )


            # ==================================================
            # ASSISTIVE INTENT
            # ==================================================

            st.markdown("---")

            st.subheader(
                "🧩 Assistive Interaction"
            )


            intent_col1, intent_col2 = st.columns(2)


            with intent_col1:

                st.info(
                    f"""
### Detected Brain Activity

**{prediction}**
"""
                )


            with intent_col2:

                st.success(
                    f"""
### Suggested Assistive Intent

**{intent}**
"""
                )


            st.caption(
                "Assistive intent is a prototype mapping for the current research system and should not be interpreted as a clinical command."
            )


            # ==================================================
            # SIGNAL QUALITY
            # ==================================================

            st.markdown("---")

            st.subheader(
                "🧠 EEG Signal Quality Assessment"
            )


            q1, q2, q3, q4 = st.columns(4)


            with q1:

                st.metric(
                    "Quality Score",
                    f"{quality_score}%"
                )


            with q2:

                st.metric(
                    "Signal Quality",
                    quality
                )


            with q3:

                st.metric(
                    "Noise Level",
                    noise
                )


            with q4:

                st.metric(
                    "Prediction Latency",
                    f"{average_latency:.2f} ms"
                )


            # ==================================================
            # SIGNAL QUALITY DETAILS
            # ==================================================

            with st.expander(
                "🔎 Signal Quality Details"
            ):

                quality_df = pd.DataFrame({

                    "Parameter": [
                        "Quality Score",
                        "Quality Level",
                        "Noise Level",
                        "Signal Standard Deviation",
                        "Peak Amplitude"
                    ],

                    "Value": [
                        f"{quality_score}%",
                        quality,
                        noise,
                        f"{std_value:.6f}",
                        f"{peak_value:.6f}"
                    ]
                })


                st.dataframe(
                    quality_df,
                    use_container_width=True,
                    hide_index=True
                )


            # ==================================================
            # RECORDING INFORMATION
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📄 Recording Information"
            )


            r1, r2, r3, r4 = st.columns(4)


            with r1:

                st.info(
                    f"📂 **File**\n\n{subject_name}"
                )


            with r2:

                st.info(
                    f"⏱ **Duration**\n\n{recording_length:.2f} sec"
                )


            with r3:

                st.info(
                    f"📡 **Sampling Rate**\n\n{sampling_rate:.0f} Hz"
                )


            with r4:

                st.info(
                    f"🔢 **Channels**\n\n{channels}"
                )


            # ==================================================
            # EEG SIGNAL VIEWER
            # ==================================================

            st.markdown("---")

            st.subheader(
                "🧠 EEG Signal Viewer"
            )


            channel = st.selectbox(
                "Select EEG Channel",
                raw.ch_names
            )


            plot_eeg_signal(
                raw,
                channel
            )


            # ==================================================
            # SIGNAL STATISTICS
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📊 Signal Statistics"
            )


            signal = raw.get_data()


            stats = pd.DataFrame({

                "Statistic": [

                    "Minimum",

                    "Maximum",

                    "Mean",

                    "Standard Deviation"

                ],

                "Value": [

                    round(
                        float(signal.min()),
                        6
                    ),

                    round(
                        float(signal.max()),
                        6
                    ),

                    round(
                        float(signal.mean()),
                        6
                    ),

                    round(
                        float(signal.std()),
                        6
                    )
                ]
            })


            st.dataframe(
                stats,
                use_container_width=True,
                hide_index=True
            )


            # ==================================================
            # FREQUENCY BAND ANALYSIS
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📈 EEG Frequency Band Analysis"
            )


            band_power = (
                frequency_band_analysis(
                    raw
                )
            )


            band_df = pd.DataFrame({

                "Frequency Band":
                    list(
                        band_power.keys()
                    ),

                "Power":
                    list(
                        band_power.values()
                    )
            })


            st.dataframe(
                band_df,
                use_container_width=True,
                hide_index=True
            )


            st.bar_chart(
                band_df.set_index(
                    "Frequency Band"
                )
            )


            # ==================================================
            # PSD
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📊 Power Spectral Density"
            )


            plot_psd(
                raw
            )


            # ==================================================
            # EEG HEATMAP
            # ==================================================

            st.markdown("---")

            st.subheader(
                "🧠 Brain Activity Heatmap"
            )


            eeg_heatmap(
                raw
            )


            # ==================================================
            # PROBABILITY + CONFIDENCE
            # ==================================================

            st.markdown("---")

            left, right = st.columns(2)


            # ==================================================
            # LEFT COLUMN
            # ==================================================

            with left:

                st.subheader(
                    "📈 Class Probability"
                )


                probability_chart(
                    probabilities
                )


                prob_df = pd.DataFrame({

                    "Brain Activity":
                        CLASS_NAMES,

                    "Probability (%)":
                        np.round(
                            probabilities * 100,
                            2
                        )
                })


                st.dataframe(
                    prob_df,
                    use_container_width=True,
                    hide_index=True
                )


                # ----------------------------------------------
                # Prediction Ranking
                # ----------------------------------------------

                st.subheader(
                    "🏆 AI Prediction Ranking"
                )


                ranking = pd.DataFrame({

                    "Brain Activity":
                        CLASS_NAMES,

                    "Probability (%)":
                        np.round(
                            probabilities * 100,
                            2
                        )
                })


                ranking = ranking.sort_values(
                    by="Probability (%)",
                    ascending=False
                )


                st.dataframe(
                    ranking,
                    use_container_width=True,
                    hide_index=True
                )


            # ==================================================
            # RIGHT COLUMN
            # ==================================================

            with right:

                st.subheader(
                    "🎯 Confidence Meter"
                )


                confidence_gauge(
                    confidence
                )


                st.subheader(
                    "🤖 AI Confidence Interpretation"
                )


                if confidence >= 90:

                    st.success(
                        "🟢 Very High Confidence Prediction"
                    )

                elif confidence >= 75:

                    st.success(
                        "🟢 High Confidence Prediction"
                    )

                elif confidence >= 50:

                    st.warning(
                        "🟡 Moderate Confidence Prediction"
                    )

                else:

                    st.error(
                        "🔴 Low Confidence Prediction"
                    )


                # ----------------------------------------------
                # Stability
                # ----------------------------------------------

                st.subheader(
                    "📊 Prediction Stability"
                )


                st.progress(
                    min(
                        max(
                            stability / 100,
                            0.0
                        ),
                        1.0
                    )
                )


                if stability >= 80:

                    st.success(
                        f"Stable prediction: {stability:.2f}%"
                    )

                elif stability >= 60:

                    st.warning(
                        f"Moderately stable: {stability:.2f}%"
                    )

                else:

                    st.error(
                        f"Unstable prediction: {stability:.2f}%"
                    )


            # ==================================================
            # CLINICAL / SIGNAL INTERPRETATION
            # ==================================================

            st.markdown("---")

            st.subheader(
                "🧠 Signal Interpretation"
            )


            interpretation = (
                CLINICAL_INTERPRETATION.get(
                    prediction,
                    "Brain activity pattern detected."
                )
            )


            st.info(
                interpretation
            )


            st.caption(
                "This interpretation describes the model's predicted EEG class and is not a medical diagnosis."
            )


            # ==================================================
            # EPOCH PREDICTIONS
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📊 Epoch Prediction Distribution"
            )


            prediction_distribution(
                epoch_predictions
            )


            # ==================================================
            # EPOCH DETAILS
            # ==================================================

            with st.expander(
                "🔬 View Epoch-Level Predictions"
            ):

                epoch_df = pd.DataFrame({

                    "Epoch":
                        np.arange(
                            1,
                            len(epoch_predictions) + 1
                        ),

                    "Prediction":
                        epoch_names,

                    "Confidence (%)":
                        np.round(
                            result[
                                "epoch_confidences"
                            ] * 100,
                            2
                        )
                })


                st.dataframe(
                    epoch_df,
                    use_container_width=True,
                    hide_index=True
                )


            # ==================================================
            # RECORDING SUMMARY
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📋 Recording Summary"
            )


            summary = pd.DataFrame({

                "Property": [

                    "File Name",

                    "Channels",

                    "Sampling Frequency",

                    "Recording Length",

                    "Total Epochs",

                    "Valid Epochs",

                    "Final Prediction",

                    "Confidence",

                    "Stability",

                    "Assistive Intent"

                ],

                "Value": [

                    subject_name,

                    channels,

                    f"{sampling_rate:.0f} Hz",

                    f"{recording_length:.2f} sec",

                    epochs_count,

                    valid_epochs,

                    prediction,

                    f"{confidence:.2f}%",

                    f"{stability:.2f}%",

                    intent

                ]
            })


            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )


            # ==================================================
            # PROCESSING LOG
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📝 Processing Log"
            )


            current_time = (
                datetime.now().strftime(
                    "%H:%M:%S"
                )
            )


            logs = [

                (
                    "EDF File Loaded",
                    "✅"
                ),

                (
                    "Signal Quality Assessment",
                    "✅"
                ),

                (
                    "EEG Filtering Completed",
                    "✅"
                ),

                (
                    "EEG Normalization Completed",
                    "✅"
                ),

                (
                    "Epoch Generation Completed",
                    "✅"
                ),

                (
                    "DNN Prediction Completed",
                    "✅"
                ),

                (
                    "Confidence Analysis Completed",
                    "✅"
                ),

                (
                    "Prediction Stability Calculated",
                    "✅"
                ),

                (
                    "Frequency Analysis Completed",
                    "✅"
                ),

                (
                    "Report Ready",
                    "✅"
                )
            ]


            for event, status_icon in logs:

                st.write(
                    f"{status_icon} "
                    f"{current_time} - "
                    f"{event}"
                )


            # ==================================================
            # PDF REPORT
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📄 Analysis Report"
            )


            report_path = os.path.join(
                ROOT_DIR,
                "Brain_Signal_Report.pdf"
            )


            try:

                generate_report(

                    report_path,

                    prediction,

                    confidence,

                    channels,

                    sampling_rate,

                    recording_length,

                    probabilities

                )


                if os.path.exists(
                    report_path
                ):

                    with open(
                        report_path,
                        "rb"
                    ) as pdf:

                        st.download_button(

                            label=
                            "📄 Download Analysis Report",

                            data=pdf,

                            file_name=
                            "Brain_Signal_Report.pdf",

                            mime=
                            "application/pdf"

                        )

            except Exception as report_error:

                st.warning(
                    f"Report generation failed: {report_error}"
                )


            # ==================================================
            # SUCCESS
            # ==================================================

            st.success(
                "✅ EEG analysis completed successfully!"
            )


    except Exception as error:

        st.error(
            f"❌ Application Error: {error}"
        )

        with st.expander(
            "🔎 Technical Details"
        ):

            st.exception(
                error
            )


    finally:

        # --------------------------------------------------
        # Remove Temporary EDF
        # --------------------------------------------------

        if (
            temp_file is not None
            and os.path.exists(temp_file)
        ):

            try:

                os.remove(
                    temp_file
                )

            except Exception:

                pass


# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.markdown("---")


with st.expander(
    "📘 About this Project",
    expanded=False
):

    st.markdown(
        """
## 🧠 AI Driven Brain Signal Analyzer

This project analyzes EEG brain signals and classifies
motor imagery activity using a Deep Neural Network.

### 🎯 Supported Classes

- Rest
- Left Fist
- Right Fist
- Both Fists
- Both Feet

### 📂 Dataset

PhysioNet EEG Motor Movement / Imagery Dataset

### 🛠 Technologies

- Python
- TensorFlow / Keras
- MNE-Python
- Streamlit
- Plotly
- NumPy
- Pandas
- Scikit-learn
- SciPy

### 🚀 Current Features

- EDF EEG upload
- EEG preprocessing
- EEG filtering
- Channel normalization
- DNN classification
- Confidence analysis
- Prediction stability
- Signal quality assessment
- EEG waveform visualization
- Frequency band analysis
- Power Spectral Density
- Brain activity heatmap
- Epoch-level predictions
- Prediction ranking
- Assistive intent mapping
- Processing latency measurement
- PDF report generation

### 🔬 Planned Advanced Features

- ANN model
- SNN model
- ANN vs DNN vs SNN comparison
- Hybrid model fusion
- Explainable AI
- EEG scalp topography
- Personalized BCI
- Real-time EEG streaming
- Assistive communication interface
- SOS interaction
"""
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")


footer_col1, footer_col2 = st.columns(
    [3, 1]
)


with footer_col1:

    st.caption(
        "© 2026 AI Driven Brain Signal Analyzer | "
        "TensorFlow • MNE-Python • Streamlit"
    )


with footer_col2:

    st.caption(
        "Version 2.0"
    )