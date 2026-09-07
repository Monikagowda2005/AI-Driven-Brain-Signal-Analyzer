import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================================================
# Class Names
# ==========================================================

CLASS_NAMES = [
    "Rest",
    "Left Fist",
    "Right Fist",
    "Both Fists",
    "Both Feet"
]

# ==========================================================
# EEG Waveform
# ==========================================================

def plot_eeg_signal(raw, channel):

    idx = raw.ch_names.index(channel)

    signal = raw.get_data()[idx]

    time = raw.times

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=signal,
            mode="lines",
            name=channel
        )
    )

    fig.update_layout(
        title=f"EEG Signal - {channel}",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Probability Chart
# ==========================================================

def probability_chart(probabilities):

    probs = np.array(probabilities) * 100

    fig = px.bar(
        x=CLASS_NAMES,
        y=probs,
        labels={
            "x": "Class",
            "y": "Probability (%)"
        },
        text=np.round(probs, 2)
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        title="Prediction Probability",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Confidence Gauge
# ==========================================================

def confidence_gauge(confidence):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text": "Confidence (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.35},
                "steps": [
                    {"range": [0, 40], "color": "#ff4d4d"},
                    {"range": [40, 70], "color": "#ffcc00"},
                    {"range": [70, 100], "color": "#33cc33"}
                ]
            }
        )
    )

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Prediction Distribution
# ==========================================================

def prediction_distribution(predictions):

    counts = {}

    for item in predictions:

        counts[item] = counts.get(item, 0) + 1

    df = pd.DataFrame({

        "Prediction": list(counts.keys()),
        "Epochs": list(counts.values())

    })

    st.subheader("Epoch-wise Prediction Distribution")

    fig = px.pie(
        df,
        names="Prediction",
        values="Epochs",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Epoch-wise Prediction Table")

    table = pd.DataFrame({

        "Epoch": np.arange(1, len(predictions)+1),
        "Prediction": predictions

    })

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )