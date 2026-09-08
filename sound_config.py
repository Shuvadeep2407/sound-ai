"""Configuration for the multi-class sound-to-object recognizer.

This is a separate, larger model from the original drone detector. It learns
to tell apart several everyday sounds (car, train, bus, airplane, helicopter,
mobile ringtone) and runs on a PC or Raspberry Pi using only NumPy / SciPy /
scikit-learn.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders used by this model (kept separate from the drone-detector folders).
DATASET_DIR = os.path.join(BASE_DIR, "sound_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "sound_recognizer.joblib")

# Audio settings. 16 kHz keeps everything fast and matches embedded audio.
SAMPLE_RATE = 16000

# Object classes we recognise. Folder names under DATASET_DIR must match these.
CLASSES = [
    "car",
    "train",
    "bus",
    "airplane",
    "helicopter",
    "mobile_ringtone",
]

# MFCC / feature settings (kept in sync with features.py).
N_MFCC = 13
N_MELS = 40
FRAME_LEN = 0.025
HOP_LEN = 0.010

# Training clip length (seconds). The model works best on clips >= 1 s.
CLIP_SECONDS = 2.0
