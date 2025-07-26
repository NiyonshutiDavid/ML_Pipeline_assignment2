import os
import librosa
import numpy as np

base_path = '../audio'

def load_audio(member, sample):
    for ext in ['wav', 'mp3']:
        path = os.path.join(base_path, member, f'{sample}.{ext}')
        if os.path.exists(path):
            y, sr = librosa.load(path, sr=None)
            return y, sr
    raise FileNotFoundError(f'No audio found for {member} - {sample}')

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
