"""Evaluate every .wav file in a folder and print predictions + confidence.

Usage:
    python evaluate.py test
"""
import os
import sys

import numpy as np
from scipy.io import wavfile
import joblib

import config
from features import extract_features


def load_model():
    data = joblib.load(config.MODEL_PATH)
    return data["model"], data["scaler"], data["label_encoder"]


def classify(audio, sr, model, scaler, le):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        new_len = int(len(audio) * config.SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                          np.arange(len(audio)), audio)
    feats = extract_features(audio, config.SAMPLE_RATE).reshape(1, -1)
    feats = scaler.transform(feats)
    proba = model.predict_proba(feats)[0]
    pred = le.inverse_transform([int(np.argmax(proba))])[0]
    return pred, proba


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "test"
    model, scaler, le = load_model()
    classes = list(le.classes_)
    drone_idx = classes.index("drone")
    no_idx = classes.index("no_drone")

    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav"))
    print(f"{'file':28s} {'drone%':>7s} {'no_drone%':>10s}  prediction")
    print("-" * 66)
    for f in files:
        sr, audio = wavfile.read(os.path.join(folder, f))
        pred, proba = classify(audio, sr, model, scaler, le)
        print(f"{f:28s} {proba[drone_idx] * 100:7.1f} "
              f"{proba[no_idx] * 100:10.1f}  {pred}")


if __name__ == "__main__":
    main()
