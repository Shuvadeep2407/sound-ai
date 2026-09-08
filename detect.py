"""Classify a sound clip (or live microphone) as 'drone' vs 'no_drone'.

Usage:
    python detect.py --file path/to/sound.wav
    python detect.py --mic               # live microphone (needs sounddevice)

Run 'python train.py' first so a model exists.
"""
import argparse
import sys

import numpy as np
from scipy.io import wavfile
import joblib

import config
from features import extract_features


def load_model():
    data = joblib.load(config.MODEL_PATH)
    return data["model"], data["scaler"], data["label_encoder"]


def classify_audio(audio, sr, model, scaler, le):
    # Match training: mono + resample to 16 kHz if needed.
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        new_len = int(len(audio) * config.SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                          np.arange(len(audio)), audio)

    feats = extract_features(audio, config.SAMPLE_RATE).reshape(1, -1)
    feats = scaler.transform(feats)
    proba = model.predict_proba(feats)[0]
    pred_idx = int(np.argmax(proba))
    label = le.inverse_transform([pred_idx])[0]
    return label, proba


def main():
    parser = argparse.ArgumentParser(description="Detect drone sounds")
    parser.add_argument("--file", help="path to a .wav file to classify")
    parser.add_argument("--mic", action="store_true",
                        help="classify live microphone input")
    args = parser.parse_args()

    if not args.file and not args.mic:
        parser.print_help()
        sys.exit(1)

    model, scaler, le = load_model()

    if args.file:
        sr, audio = wavfile.read(args.file)
        label, proba = classify_audio(audio, sr, model, scaler, le)
        print(f"File: {args.file}")
        for name, p in zip(le.classes_, proba):
            print(f"  {name:10s} : {p:.3f}")
        print(f"Prediction  : {label}  (confidence {proba.max():.3f})")

    if args.mic:
        try:
            import sounddevice as sd
        except ImportError:
            print("Please install sounddevice for mic input: pip install sounddevice")
            sys.exit(1)

        duration = 1.0  # classify 1-second windows
        frames = int(duration * config.SAMPLE_RATE)
        print("Listening... press Ctrl+C to stop.")
        try:
            while True:
                audio = sd.rec(frames, samplerate=config.SAMPLE_RATE,
                               channels=1, dtype="float32")
                sd.wait()
                label, proba = classify_audio(audio[:, 0], config.SAMPLE_RATE,
                                              model, scaler, le)
                print(f"\r{label}  confidence={proba.max():.3f}    ",
                      end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
