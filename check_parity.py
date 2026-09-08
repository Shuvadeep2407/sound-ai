"""Verify that the streaming C feature extractor gives identical predictions
to the Python reference (features.extract_features) on every file in a folder.

Run:  python check_parity.py test
"""
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
import joblib

import config
import export_c as ec
from features import extract_features, mel_filterbank


def load_audio(path):
    sr, a = wavfile.read(path)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        a = np.interp(np.linspace(0, len(a) - 1,
                                   int(len(a) * config.SAMPLE_RATE / sr)),
                      np.arange(len(a)), a)
    return a


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "test"
    data = joblib.load(config.MODEL_PATH)
    model, scaler = data["model"], data["scaler"]

    window = get_window("hamming", ec.FRAME_SAMPLES, fftbins=False)
    mel = mel_filterbank(ec.NFFT, config.SAMPLE_RATE, config.N_MELS)
    dct = ec.build_dct_matrix(config.N_MFCC, config.N_MELS)

    agree = total = 0
    max_dprob = 0.0
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".wav"):
            continue
        a = load_audio(os.path.join(folder, f))
        f_py = extract_features(a, config.SAMPLE_RATE)
        f_c = ec.c_reference_features(a, config.SAMPLE_RATE, window, mel, dct)

        p_py = model.predict_proba(scaler.transform(f_py.reshape(1, -1)))[0]
        p_c = model.predict_proba(scaler.transform(f_c.reshape(1, -1)))[0]

        same = int(np.argmax(p_py)) == int(np.argmax(p_c))
        total += 1
        agree += same
        max_dprob = max(max_dprob, abs(float(p_py[0] - p_c[0])))
        print(f"{f:24s} py={p_py[0]:.4f}  C-stream={p_c[0]:.4f}  "
              f"match={'OK' if same else 'MISMATCH'}")

    print(f"\nPredictions agree: {agree}/{total}")
    print(f"Max |delta drone-prob|: {max_dprob:.2e}")


if __name__ == "__main__":
    main()
