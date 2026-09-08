"""Copy the real recordings from the 'test' folder into the training set,
using the filename to infer the label.

    drone / dron / combet drone  ->  dataset/drone
    bird / car                   ->  dataset/no_drone

Run:  python add_real_data.py [source_folder]
"""
import os
import shutil
import sys

import config

DRONE_KEYWORDS = ["drone", "dron", "combet"]
NO_DRONE_KEYWORDS = ["bird", "car"]


def infer_label(name):
    lower = name.lower()
    for kw in DRONE_KEYWORDS:
        if kw in lower:
            return "drone"
    for kw in NO_DRONE_KEYWORDS:
        if kw in lower:
            return "no_drone"
    return None


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "test"
    counts = {"drone": 0, "no_drone": 0}

    for f in sorted(os.listdir(src)):
        if not f.lower().endswith(".wav"):
            continue
        label = infer_label(f)
        if label is None:
            print(f"skip (unknown label): {f}")
            continue
        dst_dir = os.path.join(config.DATASET_DIR, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"real_{counts[label]:02d}_{f}")
        shutil.copyfile(os.path.join(src, f), dst)
        counts[label] += 1
        print(f"{label:9s} <- {f}")

    print(f"\nDone: {counts['drone']} drone, {counts['no_drone']} no_drone clips "
          f"copied into {config.DATASET_DIR}")


if __name__ == "__main__":
    main()
