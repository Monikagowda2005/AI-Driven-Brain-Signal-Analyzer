import numpy as np

def signal_quality(raw):
    signal = raw.get_data()

    std = np.std(signal)

    if std < 20:
        score = 95
        quality = "Excellent"
        noise = "Very Low"

    elif std < 50:
        score = 82
        quality = "Good"
        noise = "Low"

    elif std < 100:
        score = 65
        quality = "Moderate"
        noise = "Medium"

    else:
        score = 45
        quality = "Poor"
        noise = "High"

    return score, quality, noise