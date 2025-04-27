import numpy as np
from scipy.signal import spectrogram

def process_meg_signal(meg_data):
    f, t, Sxx = spectrogram(meg_data)
    attention_score = np.mean(Sxx)
    return attention_score > np.median(Sxx)