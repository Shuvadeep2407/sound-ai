"""Train the multi-class sound-to-object recognizer.

Run:  python train_sound_model.py

Loads every .wav in sound_dataset/<class>/, extracts features, then trains a
"big" ensemble (a deep MLP + Random Forest + Gradient Boosting combined with
soft voting), reports cross-validated accuracy, and saves the best model.
"""
import os
import sys

import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

import sound_config as cfg
from sound_features import extract_features


def load_wav(path):
    sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != cfg.SAMPLE_RATE:
        new_len = int(len(audio) * cfg.SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                          np.arange(len(audio)), audio)
    return audio


def load_dataset():
    X, y = [], []
    for cls in cfg.CLASSES:
        folder = os.path.join(cfg.DATASET_DIR, cls)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
        for f in files:
            audio = load_wav(os.path.join(folder, f))
            X.append(extract_features(audio, cfg.SAMPLE_RATE))
            y.append(cls)
        print(f"Loaded {len(files)} clips for '{cls}'")

    if not X:
        print("No .wav files found! Run 'python make_sound_dataset.py' first.")
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

    # --- Big models (PC / Raspberry Pi friendly, CPU only) ---
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=300, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("mlp", mlp), ("rf", rf), ("gb", gb)],
        voting="soft",
    )

    print("Training ensemble (MLP + RandomForest + GradientBoosting)...")
    ensemble.fit(X_train_s, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_train_s, y_train, cv=cv, n_jobs=1)
    test_acc = accuracy_score(y_test, ensemble.predict(X_test_s))

    print("\n=== Results ===")
    print(f"5-fold CV accuracy : {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"Test accuracy      : {test_acc:.3f}")

    print("\nPer-model test accuracy:")
    for name, model in [("MLP", mlp), ("RandomForest", rf),
                        ("GradientBoosting", gb), ("Ensemble", ensemble)]:
        print(f"  {name:16s} : {accuracy_score(y_test, model.predict(X_test_s)):.3f}")

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    joblib.dump(
        {"model": ensemble, "scaler": scaler, "label_encoder": le,
         "classes": list(le.classes_)},
        cfg.MODEL_PATH,
    )
    print(f"\nSaved model to {cfg.MODEL_PATH}")

    print("\n" + classification_report(
        y_test, ensemble.predict(X_test_s), target_names=le.classes_))


if __name__ == "__main__":
    main()
