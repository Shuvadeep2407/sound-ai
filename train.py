"""Train a fast drone-sound classifier and save the best model.

Run:  python train.py

It loads every .wav file in dataset/<class>/, extracts features, then
trains BOTH a small neural network (MLP) and a Random Forest, prints the
accuracy of each, and saves the better one to model/drone_detector.joblib.
"""
import os
import sys

import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

import config
from features import extract_features


def load_wav(path):
    sr, audio = wavfile.read(path)
    if audio.ndim > 1:                # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != config.SAMPLE_RATE:      # quick linear resample if needed
        new_len = int(len(audio) * config.SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                          np.arange(len(audio)), audio)
    return audio


def load_dataset():
    X, y = [], []
    for cls in config.CLASSES:
        folder = os.path.join(config.DATASET_DIR, cls)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
        for f in files:
            audio = load_wav(os.path.join(folder, f))
            X.append(extract_features(audio, config.SAMPLE_RATE))
            y.append(cls)
        print(f"Loaded {len(files)} clips for class '{cls}'")

    if not X:
        print("No .wav files found! Run 'python make_dataset.py' first.")
        sys.exit(1)
    return np.array(X), np.array(y)


def main():
    X, y = load_dataset()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, stratify=y_enc, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Fast small neural network ---
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        early_stopping=True,
        random_state=42,
    )
    mlp.fit(X_train_s, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test_s))

    # --- Fast Random Forest (great baseline) ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_s))

    print("\n=== Results (test set) ===")
    print(f"MLP (neural net) accuracy : {mlp_acc:.3f}")
    print(f"Random Forest accuracy    : {rf_acc:.3f}")

    best_model = mlp if mlp_acc >= rf_acc else rf
    best_name = "MLP" if mlp_acc >= rf_acc else "RandomForest"
    print(f"\nSaving best model: {best_name}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(
        {"model": best_model, "scaler": scaler, "label_encoder": le,
         "classes": list(le.classes_), "mlp": mlp},
        config.MODEL_PATH,
    )
    print(f"Saved to {config.MODEL_PATH}")

    # Detailed per-class report for the chosen model.
    print("\n" + classification_report(
        y_test, best_model.predict(X_test_s), target_names=le.classes_))


if __name__ == "__main__":
    main()
