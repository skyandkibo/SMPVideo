import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyAudioAnalysis import audioSegmentation as aS


def extract_audio_features(audio_path, sr=22050):
    """
    Extracts 12 key audio features for TikTok popularity prediction:
    1. RMS mean & std
    2. ZCR mean
    3. Tempo & beat stability
    4. Spectral centroid & rolloff mean
    5. Spectral contrast mean
    6. Chroma mean
    7. MFCC 1-3 mean
    8. Delta MFCC 1-3 mean
    9. Predicted emotion label
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Frame-level features
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
    delta_mfcc = librosa.feature.delta(mfcc)

    # Rhythm
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_intervals = np.diff(beats)

    # Emotion label
    emotion = "unknown"

    # Aggregate statistics with .item() for future NumPy compatibility
    feats = {
        'rms_mean': np.mean(rms).item(),
        'rms_std': np.std(rms).item(),
        'zcr_mean': np.mean(zcr).item(),
        'tempo': float(tempo) if not isinstance(tempo, np.ndarray) else tempo.item(),
        'beat_std': np.std(beat_intervals).item() if beat_intervals.size > 1 else 0.0,
        'spectral_centroid_mean': np.mean(spec_cent).item(),
        'spectral_rolloff_mean': np.mean(spec_roll).item(),
        'spectral_contrast_mean': np.mean(spec_contrast).item(),
        'chroma_mean': np.mean(chroma).item(),
        'emotion_label': emotion
    }

    for i in range(3):
        feats[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i]).item()
        feats[f'delta_mfcc_{i+1}_mean'] = np.mean(delta_mfcc[i]).item()

    return feats


def process_folder(audio_folder, output_csv, sr=22050):
    """Process all audio files in a folder and save features to CSV."""
    rows = []
    files = [f for f in os.listdir(audio_folder) if f.lower().endswith(('.wav', '.mp3'))]
    for fname in tqdm(files, desc="Extracting audio features"):
        path = os.path.join(audio_folder, fname)
        base = os.path.splitext(fname)[0]
        feats = extract_audio_features(path, sr)
        feats['vid'] = base
        rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Extract 12 key audio features for TikTok XGBoost model")
    parser.add_argument('--audio_folder', type=str, required=True, help='Path to audio folder')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--sr', type=int, default=22050, help='Sampling rate')
    args = parser.parse_args()
    process_folder(args.audio_folder, args.output_csv, sr=args.sr)
