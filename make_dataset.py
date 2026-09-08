"""Generate synthetic training data so you can try the whole pipeline
without recording real drone audio.

Creates:
    dataset/drone/*.wav     -> synthetic "propeller buzz" sounds
    dataset/no_drone/*.wav  -> synthetic background/noise sounds

Run:  python make_dataset.py
"""
import os
import argparse

import numpy as np
from scipy.io import wavfile

import config


def make_drone(seconds, sr):
    """Synthetic drone: a blade-pass tone plus harmonics with a slow 'buzz' AM."""
    t = np.arange(int(seconds * sr)) / sr
    blade_pass_freq = np.random.uniform(120, 350)  # Hz (rotor blade pass)
    signal = np.zeros_like(t)
    for h in range(1, 7):  # harmonics fall off as 1/h
        amp = (1.0 / h) * np.random.uniform(0.5, 1.0)
        signal += amp * np.sin(2 * np.pi * blade_pass_freq * h * t)

    # Slow amplitude modulation = the characteristic "buzz".
    buzz_freq = np.random.uniform(8, 30)
    signal *= 1.0 + 0.4 * np.sin(2 * np.pi * buzz_freq * t)

    # A little broadband noise.
    signal += 0.02 * np.random.randn(len(t))
    return signal


def make_no_drone(seconds, sr):
    """Synthetic non-drone background: noise or a wandering tone."""
    t = np.arange(int(seconds * sr)) / sr
    if np.random.rand() < 0.5:
        # Coloured-ish noise (low-pass filtered white noise).
        white = np.random.randn(len(t))
        b = np.array([0.2, 0.5, 0.8, 0.5, 0.2])  # simple smoothing kernel
        b /= b.sum()
        signal = np.convolve(white, b, mode="same")
    else:
        # A tone that slowly drifts (sounds more like speech/music than a buzz).
        f0 = np.random.uniform(100, 800)
        phase = np.cumsum(np.random.uniform(0.0, 0.02, len(t)))
        signal = np.sin(2 * np.pi * f0 * t + phase)
    return 0.5 * signal


def write_clip(path, samples, sr):
    samples = samples - samples.mean()
    peak = np.max(np.abs(samples))
    if peak > 1e-6:
        samples = samples / peak
    samples = np.clip(samples, -1.0, 1.0)
    wavfile.write(path, sr, (samples * 32767).astype(np.int16))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic drone dataset")
    parser.add_argument("--per-class", type=int, default=40,
                        help="number of clips per class")
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="clip length in seconds")
    args = parser.parse_args()

    for cls in config.CLASSES:
        folder = os.path.join(config.DATASET_DIR, cls)
        os.makedirs(folder, exist_ok=True)

    sr = config.SAMPLE_RATE
    for i in range(args.per_class):
        drone = make_drone(args.seconds, sr)
        write_clip(os.path.join(config.DATASET_DIR, "drone", f"drone_{i:04d}.wav"),
                   drone, sr)

        no_drone = make_no_drone(args.seconds, sr)
        write_clip(os.path.join(config.DATASET_DIR, "no_drone", f"no_drone_{i:04d}.wav"),
                   no_drone, sr)

    print(f"Done. Generated {args.per_class} clips per class in {config.DATASET_DIR}")


if __name__ == "__main__":
    main()
