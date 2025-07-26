import os
import librosa
import numpy as np

base_path = '../audio'

def load_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    valid_exts = ['.wav', '.mp3']
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in valid_exts:
        raise ValueError(f"Unsupported audio format: {ext}. Supported: {valid_exts}")

    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {file_path}. Error: {e}")

    return y, sr

def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    energy = np.mean(y**2)
    features = {'energy': energy, 'rolloff': rolloff}
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i}'] = mfcc
    return features

def get_features_for_audio(member, sample):
    y, sr = load_audio(member, sample)
    feats = extract_features(y, sr)
    return feats

# Example usage:
if __name__ == '__main__':
    member = 'david'
    sample = 'approve'
    features = get_features_for_audio(member, sample)
    print(features)
