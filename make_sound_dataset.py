"""Generate synthetic training data for the multi-class sound recognizer.

Creates sound_dataset/<class>/*.wav clips for each object class:
car, train, bus, airplane, helicopter, mobile_ringtone.

The sounds are simplified but distinct synthetic versions of each object so
the whole pipeline can be exercised without recording real audio. Replace or
add real recordings later and re-run train_sound_model.py.

Run:  python make_sound_dataset.py
"""
import os
import argparse

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

import sound_config as cfg


def _lowpass(sig, sr, cutoff, order=4):
    b, a = butter(order, cutoff / (sr / 2.0), btype="low")
    return lfilter(b, a, sig)


def _highpass(sig, sr, cutoff, order=4):
    b, a = butter(order, cutoff / (sr / 2.0), btype="high")
    return lfilter(b, a, sig)


def _bandpass(sig, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2.0), hi / (sr / 2.0)], btype="band")
    return lfilter(b, a, sig)


# --------------------------------------------------------------------------- #
# Per-class synthesizers (each returns a float signal roughly in [-1, 1]).
# --------------------------------------------------------------------------- #
def make_car(seconds, sr):
    """Car: low-frequency engine rumble (fundamental + harmonics) + road noise."""
    t = np.arange(int(seconds * sr)) / sr
    base = np.random.uniform(40, 90)
    sig = np.zeros_like(t)
    for h in range(1, 5):
        sig += (1.0 / h) * np.sin(2 * np.pi * base * h * t + np.random.uniform(0, 2 * np.pi))
    rumble = _lowpass(np.random.randn(len(t)), sr, 300)
    sig = 0.7 * sig + 0.4 * rumble
    env = 1.0 + 0.25 * np.sin(2 * np.pi * np.random.uniform(0.5, 2.0) * t)
    return sig * env


def make_train(seconds, sr):
    """Train: rhythmic low thumps (chug) + low rumble + high rail hiss."""
    t = np.arange(int(seconds * sr)) / sr
    rate = np.random.uniform(2.0, 4.5)  # chugs per second
    pulse = (np.sin(2 * np.pi * rate * t) > 0).astype(float)
    thump = _lowpass(pulse, sr, 80)
    rumble = _lowpass(np.random.randn(len(t)), sr, 150)
    rail = _bandpass(np.random.randn(len(t)), sr, 1200, 4500)
    return 2.0 * thump + 0.5 * rumble + 0.15 * rail


def make_bus(seconds, sr):
    """Bus: deep diesel engine + occasional air-brake hiss."""
    t = np.arange(int(seconds * sr)) / sr
    base = np.random.uniform(30, 60)
    sig = np.zeros_like(t)
    for h in range(1, 6):
        sig += (1.0 / h) * np.sin(2 * np.pi * base * h * t)
    sig += 0.4 * _lowpass(np.random.randn(len(t)), sr, 200)
    if np.random.rand() < 0.5:
        seg = int(0.3 * sr)
        start = np.random.randint(0, max(1, len(t) - seg))
        sig[start:start + seg] += 1.5 * _highpass(np.random.randn(seg), sr, 1500)
    return sig


def make_airplane(seconds, sr):
    """Airplane: broadband jet noise + a high turbine whine."""
    t = np.arange(int(seconds * sr)) / sr
    jet = _lowpass(np.random.randn(len(t)), sr, 600)
    whine = np.sin(2 * np.pi * np.random.uniform(1800, 4000) * t)
    sig = 1.2 * jet + 0.25 * whine
    sig *= 1.0 + 0.15 * np.sin(2 * np.pi * np.random.uniform(0.5, 1.5) * t)
    return sig


def make_helicopter(seconds, sr):
    """Helicopter: blade-pass 'chop' AM on rotor noise + engine + hiss."""
    t = np.arange(int(seconds * sr)) / sr
    blade = np.random.uniform(10, 25)  # blade-pass rate in Hz
    chop = 0.5 + 0.5 * np.sin(2 * np.pi * blade * t)
    rotor = _lowpass(np.random.randn(len(t)), sr, 400)
    engine = np.sin(2 * np.pi * np.random.uniform(150, 300) * t)
    hiss = _highpass(np.random.randn(len(t)), sr, 2000)
    return 1.5 * rotor * chop + 0.3 * engine + 0.1 * hiss


def make_mobile_ringtone(seconds, sr):
    """Mobile ringtone: a two-tone ringing melody with harmonics."""
    t = np.arange(int(seconds * sr)) / sr
    f1 = np.random.uniform(700, 900)
    f2 = np.random.uniform(900, 1300)
    ring = np.where((t % 1.0) < 0.5, f1, f2)
    sig = (np.sin(2 * np.pi * ring * t)
           + 0.3 * np.sin(2 * np.pi * 2 * ring * t)
           + 0.15 * np.sin(2 * np.pi * 3 * ring * t))
    env = 0.7 + 0.3 * np.sign(np.sin(2 * np.pi * 2.0 * t))
    return sig * env


GENERATORS = {
    "car": make_car,
    "train": make_train,
    "bus": make_bus,
    "airplane": make_airplane,
    "helicopter": make_helicopter,
    "mobile_ringtone": make_mobile_ringtone,
}


def write_clip(path, samples, sr):
    samples = samples - samples.mean()
    peak = np.max(np.abs(samples))
    if peak > 1e-6:
        samples = samples / peak
    samples = np.clip(samples, -1.0, 1.0)
    wavfile.write(path, sr, (samples * 32767).astype(np.int16))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sound dataset")
    parser.add_argument("--per-class", type=int, default=60,
                        help="number of clips per class")
    parser.add_argument("--seconds", type=float, default=cfg.CLIP_SECONDS,
                        help="clip length in seconds")
    args = parser.parse_args()

    for cls in cfg.CLASSES:
        os.makedirs(os.path.join(cfg.DATASET_DIR, cls), exist_ok=True)

    sr = cfg.SAMPLE_RATE
    for cls in cfg.CLASSES:
        gen = GENERATORS[cls]
        for i in range(args.per_class):
            sig = gen(args.seconds, sr)
            write_clip(os.path.join(cfg.DATASET_DIR, cls, f"{cls}_{i:04d}.wav"), sig, sr)
        print(f"Generated {args.per_class} clips for '{cls}'")

    print(f"Done. Dataset in {cfg.DATASET_DIR}")


if __name__ == "__main__":
    main()
