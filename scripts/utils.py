# utils.py – Feature extraction for .wav and .mp3 using librosa
import numpy as np
import librosa

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1)
        ])

        return features
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None
