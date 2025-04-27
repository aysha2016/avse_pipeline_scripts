import librosa
import numpy as np

def denoise_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    stft = librosa.stft(audio)
    magnitude, phase = librosa.magphase(stft)
    denoised = librosa.istft(magnitude * phase)
    return denoised, sr