"""Feature extraction for the multi-class sound-to-object recognizer.

Reuses the MFCC extractor from `features.py` and adds classic spectral /
time-domain descriptors (spectral centroid, bandwidth, roll-off, flatness,
zero-crossing rate and RMS energy). Every clip is summarised into one
fixed-length vector, which keeps training and inference fast on a PC or
Raspberry Pi.
"""
import numpy as np
from scipy.signal import get_window

import sound_config as cfg
from features import extract_mfcc

# 6 * N_MFCC (mfcc + delta + delta2, each mean + std) + 12 spectral stats.
FEAT_DIM = 6 * cfg.N_MFCC + 12


def _normalise(audio):
    audio = np.asarray(audio, dtype=np.float32)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak
    return audio


def _frames(audio, sr):
    frame_samples = int(round(cfg.FRAME_LEN * sr))
    hop_samples = int(round(cfg.HOP_LEN * sr))
    if len(audio) < frame_samples:
        audio = np.pad(audio, (0, frame_samples - len(audio)))
    n_frames = 1 + int((len(audio) - frame_samples) / hop_samples)
    window = get_window("hamming", frame_samples, fftbins=False)
    out = np.zeros((n_frames, frame_samples), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop_samples
        out[i] = audio[s:s + frame_samples] * window
    return out


def _spectral_stats(audio, sr):
    frames = _frames(audio, sr)
    nfft = 512
    power = np.abs(np.fft.rfft(frames, n=nfft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    total = power.sum(axis=1) + 1e-12

    centroid = (power * freqs).sum(axis=1) / total
    bandwidth = np.sqrt(
        ((freqs[None, :] - centroid[:, None]) ** 2 * power).sum(axis=1) / total)

    cum = np.cumsum(power, axis=1)
    rolloff = np.empty(len(frames))
    for i in range(len(frames)):
        rolloff[i] = freqs[int(np.searchsorted(cum[i], 0.85 * total[i]))]

    flatness = np.exp(np.log(power + 1e-12).mean(axis=1)) / (power.mean(axis=1) + 1e-12)
    zcr = ((frames[:, 1:] * frames[:, :-1]) < 0).sum(axis=1) / (frames.shape[1] - 1)
    rms = np.sqrt((frames ** 2).mean(axis=1))

    feats = np.vstack([centroid, bandwidth, rolloff, flatness, zcr, rms])
    return np.hstack([feats.mean(axis=1), feats.std(axis=1)])


def extract_features(audio, sr):
    """Return one fixed-length feature vector describing the whole clip."""
    audio = _normalise(audio)
    if len(audio) == 0:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    mfcc = extract_mfcc(audio, sr, n_mfcc=cfg.N_MFCC, n_mels=cfg.N_MELS,
                        frame_len=cfg.FRAME_LEN, hop_len=cfg.HOP_LEN)
    delta = np.diff(mfcc, axis=0, prepend=mfcc[:1])
    delta2 = np.diff(delta, axis=0, prepend=delta[:1])
    mfcc_stats = np.hstack([
        mfcc.mean(axis=0), mfcc.std(axis=0),
        delta.mean(axis=0), delta.std(axis=0),
        delta2.mean(axis=0), delta2.std(axis=0),
    ])

    return np.hstack([mfcc_stats, _spectral_stats(audio, sr)]).astype(np.float32)
