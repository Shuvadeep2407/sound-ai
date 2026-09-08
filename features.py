"""Audio feature extraction using only NumPy + SciPy (no librosa needed).

We convert a raw audio clip into MFCC features, add their first and second
derivatives (deltas), then summarise each clip with the mean + standard
deviation over time. This produces one small fixed-length vector per clip,
which makes training and inference very fast on CPU.
"""
import numpy as np
from scipy.fft import dct
from scipy.signal import get_window

import config


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=float) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(nfft, sr, n_mels, fmin=50.0, fmax=None):
    """Build a triangular mel filterbank (n_mels x (nfft//2 + 1))."""
    if fmax is None:
        fmax = sr / 2.0

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((nfft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, nfft // 2 + 1))
    for i in range(1, n_mels + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        for j in range(left, center):
            filters[i - 1, j] = (j - left) / (center - left + 1e-8)
        for j in range(center, right):
            filters[i - 1, j] = (right - j) / (right - center + 1e-8)
    return filters


def extract_mfcc(audio, sr, n_mfcc=config.N_MFCC, n_mels=config.N_MELS,
                 frame_len=config.FRAME_LEN, hop_len=config.HOP_LEN, nfft=None):
    """Return MFCC matrix of shape (num_frames, n_mfcc)."""
    audio = np.asarray(audio, dtype=np.float32)

    # Normalise amplitude so loudness does not matter.
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak

    # Pre-emphasis (boosts high frequencies, standard for speech/audio).
    pre_emph = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emph * audio[:-1])

    if nfft is None:
        nfft = 1
        while nfft < frame_len * sr:
            nfft *= 2

    frame_samples = int(round(frame_len * sr))
    hop_samples = int(round(hop_len * sr))

    if len(audio) < frame_samples:
        audio = np.pad(audio, (0, frame_samples - len(audio)))

    num_frames = 1 + int((len(audio) - frame_samples) / hop_samples)

    window = get_window("hamming", frame_samples, fftbins=False)
    filters = mel_filterbank(nfft, sr, n_mels)

    mfcc_frames = []
    for i in range(num_frames):
        start = i * hop_samples
        frame = audio[start:start + frame_samples] * window
        power = (np.abs(np.fft.rfft(frame, n=nfft)) ** 2) / nfft
        mel_energy = filters @ power
        mel_energy = np.where(mel_energy <= 0, np.finfo(float).eps, mel_energy)
        log_mel = np.log(mel_energy)
        mfcc = dct(log_mel, type=2, norm="ortho")[:n_mfcc]
        mfcc_frames.append(mfcc)

    return np.array(mfcc_frames)


def extract_features(audio, sr):
    """Return one fixed-length feature vector (mean + std of MFCC + deltas)."""
    mfcc = extract_mfcc(audio, sr)                      # (T, n_mfcc)
    delta = np.diff(mfcc, axis=0, prepend=mfcc[:1])     # 1st derivative
    delta2 = np.diff(delta, axis=0, prepend=delta[:1])  # 2nd derivative
    feats = np.hstack([mfcc, delta, delta2])            # (T, 3*n_mfcc)

    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    return np.hstack([mean, std]).astype(np.float32)    # (6*n_mfcc,)
