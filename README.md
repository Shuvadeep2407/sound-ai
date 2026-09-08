# Fast Drone-Sound Detector (beginner practice)

A small, **fast** machine-learning model that tells apart **drone buzzing**
sounds from other sounds. It runs entirely on your CPU (Ryzen 5 5600G), and
can be exported to **STM32** C code when you are ready.

## How it works (simple version)

1. **Audio in** → a `.wav` clip (16 kHz).
2. **Features** → MFCC (a way to describe the "shape" of the sound), plus
   their changes over time. Each clip becomes one small vector.
3. **Model** → a small neural network (or Random Forest) learned to map that
   vector to `drone` or `no_drone`.
4. **Output** → the class + a confidence score.

## Project files

| File | What it does |
|------|--------------|
| `config.py` | Shared settings (sample rate, class names, paths). |
| `features.py` | Audio → MFCC feature extraction (NumPy/SciPy only). |
| `make_dataset.py` | Generates synthetic drone/no-drone audio for testing. |
| `train.py` | Trains the model and saves the best one (+ the MLP). |
| `detect.py` | Classifies a `.wav` file or live microphone. |
| `export_c.py` | Exports the MLP + MFCC pipeline to `stm32/drone_model.h`. |
| `stm32/` | Ready-to-use C header + example for the STM32 port. |

## Quick start

```powershell
# 1. Generate a practice dataset (40 clips per class)
python make_dataset.py

# 2. Train the model (prints accuracy, saves model/drone_detector.joblib)
python train.py

# 3. Test it on a generated clip
python detect.py --file dataset\drone\drone_0000.wav
python detect.py --file dataset\no_drone\no_drone_0000.wav

# 4. (optional) export a C header for STM32
python export_c.py
```

### Live microphone (optional)

```powershell
pip install sounddevice
python detect.py --mic
```

## Using your own real data

Put your own recordings into the folders (any `.wav` files):

- `dataset/drone/`    → clips that *are* a drone
- `dataset/no_drone/` → clips that are *not* a drone (background, speech, ...)

Then re-run `python train.py`.

## Tuning for speed (and the STM32 port)

- In `train.py` the neural net is only `(64, 32)` — very fast on CPU.
- `features.py` summarises each clip with mean/std, so inference is one tiny
  matrix multiply — ideal for embedded use.
- `export_c.py` converts the same MFCC + MLP into a single self-contained C
  header (`stm32/drone_model.h`) that runs the full pipeline on the MCU.
  See `stm32/README.md` for wiring it into STM32CubeIDE/HAL.

