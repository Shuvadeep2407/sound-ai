"""Shared configuration for the drone-sound detector project.

This project is a beginner-friendly "fast" audio classifier:
it learns to tell apart "drone buzzing" sounds from everything else,
and it runs entirely on the CPU (your Ryzen 5 5600G).

STM32 deployment can be added later; the model is intentionally small
and fast so it is a good candidate for an embedded port.
"""
import os

# Root folder of this project (where this file lives).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders used by the project.
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Audio settings. 16 kHz is enough for drone/noise classification and keeps
# everything fast (also the standard sample rate for embedded/TinyML audio).
SAMPLE_RATE = 16000

# Classes. Folder names under DATASET_DIR must match these.
CLASSES = ["no_drone", "drone"]

# MFCC / feature settings.
N_MFCC = 13        # number of MFCC coefficients per frame
N_MELS = 40        # number of mel filterbank bands
FRAME_LEN = 0.025  # seconds per frame
HOP_LEN = 0.010    # seconds between frames

# Where the trained model is saved.
MODEL_PATH = os.path.join(MODEL_DIR, "drone_detector.joblib")
