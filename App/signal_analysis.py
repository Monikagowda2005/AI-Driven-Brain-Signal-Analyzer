import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import welch


# -------------------------------------------------------
# Frequency Band Power
# -------------------------------------------------------

def frequency_band_analysis(raw):

    data = raw.get_data()

    sfreq = raw.info["sfreq"]

    freqs, psd = welch(
        data,
        sfreq,
        nperseg=1024,
        axis=1
    )

    bands = {
        "Delta (0.5–4 Hz)": (0.5, 4),
        "Theta (4–8 Hz)": (4, 8),
        "Alpha (8–13 Hz)": (8, 13),
        "Beta (13–30 Hz)": (13, 30),
        "Gamma (30–45 Hz)": (30, 45),
    }

    band_power = {}

    for name, (low, high) in bands.items():

        idx = np.logical_and(freqs >= low, freqs <= high)

        power = psd[:, idx].mean()

        band_power[name] = float(power)

    return band_power
# -------------------------------------------------------
# Power Spectral Density Plot
# -------------------------------------------------------

def plot_psd(raw):

    data = raw.get_data()

    sfreq = raw.info["sfreq"]

    freqs, psd = welch(
        data,
        sfreq,
        nperseg=1024,
        axis=1
    )

    mean_psd = psd.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(freqs, mean_psd)

    ax.set_xlabel("Frequency (Hz)")

    ax.set_ylabel("Power")

    ax.set_title("Power Spectral Density")

    ax.grid(True)

    st.pyplot(fig)
    # -------------------------------------------------------
# EEG Heatmap
# -------------------------------------------------------

def eeg_heatmap(raw):

    signal = raw.get_data()

    fig, ax = plt.subplots(figsize=(12,6))

    im = ax.imshow(
        signal,
        aspect="auto",
        cmap="viridis"
    )

    ax.set_xlabel("Samples")

    ax.set_ylabel("Channels")

    ax.set_title("Brain Activity Heatmap")

    plt.colorbar(im)

    st.pyplot(fig)