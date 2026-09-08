"""Recognize the object class of a sound clip (or live microphone).

Usage:
    python recognize_sound.py --file path/to/sound.wav
    python recognize_sound.py --mic

Run 'python train_sound_model.py' first so a model exists.
"""
import argparse
import sys

import numpy as np
from scipy.io import wavfile
import joblib

import sound_config as cfg
from sound_features import extract_features


def load_model():
    data = joblib.load(cfg.MODEL_PATH)
    return data["model"], data["scaler"], data["label_encoder"]


def classify_audio(audio, sr, model, scaler, le):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != cfg.SAMPLE_RATE:
        new_len = int(len(audio) * cfg.SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                          np.arange(len(audio)), audio)

    feats = extract_features(audio, cfg.SAMPLE_RATE).reshape(1, -1)
    feats = scaler.transform(feats)
    proba = model.predict_proba(feats)[0]
    order = np.argsort(proba)[::-1]
    label = le.inverse_transform([order[0]])[0]
    return label, proba, order


def main():
    parser = argparse.ArgumentParser(description="Recognize object sounds")
    parser.add_argument("--file", help="path to a .wav file to classify")
    parser.add_argument("--mic", action="store_true", help="live microphone input")
    args = parser.parse_args()

    if not args.file and not args.mic:
        parser.print_help()
        sys.exit(1)

    model, scaler, le = load_model()

    if args.file:
        sr, audio = wavfile.read(args.file)
        label, proba, order = classify_audio(audio, sr, model, scaler, le)
        print(f"File: {args.file}")
        for idx in order:
            name = le.inverse_transform([idx])[0]
            print(f"  {name:16s} : {proba[idx]:.3f}")
        print(f"Prediction  : {label}  (confidence {proba.max():.3f})")

    if args.mic:
        try:
            import sounddevice as sd
        except ImportError:
            print("Install sounddevice for mic input: pip install sounddevice")
            sys.exit(1)

        frames = int(cfg.CLIP_SECONDS * cfg.SAMPLE_RATE)
        print("Listening... press Ctrl+C to stop.")
        try:
            while True:
                audio = sd.rec(frames, samplerate=cfg.SAMPLE_RATE,
                               channels=1, dtype="float32")
                sd.wait()
                label, proba, order = classify_audio(
                    audio[:, 0], cfg.SAMPLE_RATE, model, scaler, le)
                print(f"\r{label}  confidence={proba.max():.3f}    ",
                      end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
