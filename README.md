# Fast Drone-Sound Detector for STM32

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

## Adding the model to STM32 (step by step)

This is the full guide to deploy the drone detector on an STM32 using
STM32CubeIDE + HAL.

### What you get in `stm32/drone_model.h`

The header is **self-contained** (everything is `static` / `static inline`, no
`.c` file needed). It defines these constants:

| Constant | Value | Meaning |
|----------|-------|---------|
| `SAMPLE_RATE` | 16000 | Expected input sample rate (Hz) |
| `FRAME_SAMPLES` | 400 | One analysis frame = 25 ms |
| `HOP_SAMPLES` | 160 | Frame hop = 10 ms |
| `NFFT` | 512 | FFT size |
| `N_MFCC` | 13 | MFCC coefficients per frame |
| `FEAT_DIM` | 78 | Final feature-vector length |

And three functions:

```c
static inline void  extract_feature_vector(const float *audio, int len, float feat[FEAT_DIM]);
static inline void  drone_predict(const float *feat, float *drone_prob);
static inline float drone_detect(const float *audio, int len);   // returns prob in [0,1]
```

`drone_detect()` is the one you normally call: it takes a `float` audio buffer
and returns the probability that a drone is present (`> 0.5` = drone,
`<= 0.5` = no drone).

### Prerequisites

- **STM32CubeIDE** with the GCC toolchain for your MCU.
- An STM32 board with at least **256 KB flash** (the header is ~73 KB). The
  guide targets the **STM32C092KCT6** (256 KB flash / 30 KB RAM), but any
  similar part works.
- A **16 kHz audio source** — an I2S/PDM MEMS microphone is ideal.
- (Optional) **STM32CubeMX** to configure clocks and peripherals.

### Step 1 — Get the C header

Either use the **pre-generated** header already in this repo:

```
stm32/drone_model.h
```

or regenerate it after (re)training your model:

```powershell
python train.py
python export_c.py
```

`export_c.py` writes/overwrites `stm32/drone_model.h` and runs a self-test to
confirm the exported C math matches the Python model.

### Step 2 — Create the STM32 project

1. Open STM32CubeIDE → **File → New → STM32 Project**.
2. Choose your MCU/board (e.g. `STM32C092KCT6`) and name the project.
3. Use the default configuration for now (we will tune peripherals next).

### Step 3 — Add the header to the project

1. Copy `drone_model.h` into the include folder of your project, for example
   `Core/Inc/drone_model.h`.
2. In `main.c` (or wherever you use it), add:

```c
#include "drone_model.h"
```

There is nothing else to add to the build — it is a header-only library.

### Step 4 — Link the math library

The header uses `cosf/sinf/logf/expf/sqrtf`, so add `-lm` to the linker:

1. Right-click the project → **Properties**.
2. Go to **C/C++ Build → Settings → Tool Settings → MCU GCC Linker → Libraries**.
3. Add `m` to the **Libraries (-l)** list.
4. Apply and close.

### Step 5 — Configure the microphone (16 kHz)

Using STM32CubeMX (or manually in CubeIDE), set up an **I2S** or **PDM**
microphone so it produces samples at exactly **16000 Hz**. The model was
trained at 16 kHz, so the sample rate must match (or re-train at another rate).

- **I2S mic:** set the I2S peripheral to 16 kHz / 16-bit / mono.
- **PDM mic:** set the decimation filter to output 16 kHz.
- Use **DMA** so audio is collected in the background with little CPU load.

The model expects samples as `float` in `[-1.0, 1.0]`. If your driver gives
`int16`, divide by `32768.0f` when filling the buffer.

### Step 6 — Add the detection code

This minimal example buffers 0.25 s of audio and runs the detector:

```c
#include "drone_model.h"

/* 0.25 s of 16 kHz float audio = 4000 samples (16 KB). */
static float g_audio[SAMPLE_RATE / 4];
static int   g_audio_len = 0;

void on_mic_sample(float sample)      /* call from your mic/DMA callback */
{
    if (g_audio_len < (int)(SAMPLE_RATE / 4)) {
        g_audio[g_audio_len++] = sample;
    }
}

void drone_detect_run(void)
{
    if (g_audio_len < FRAME_SAMPLES) {
        return;                       /* not enough audio yet */
    }

    float prob = drone_detect(g_audio, g_audio_len);

    if (prob > 0.5f) {
        /* drone detected: turn on an LED, raise a flag, etc. */
    } else {
        /* no drone */
    }

    g_audio_len = 0;                  /* reset for the next window */
}
```

For a complete HAL sketch, see `stm32/main_example.c`.

### Step 7 — Build, flash, run

1. **Project → Build Project** (make sure it compiles without errors).
2. Flash to your board (**Run → Run**, or **Debug**).
3. Verify the LED / output toggles when you play a drone sound near the mic.

### Step 8 — Tune and integrate

- **Threshold:** `drone_detect()` returns a probability. Start with `0.5` and
  raise it to reduce false alarms, or lower it to be more sensitive.
- **Window length:** the model was trained on ~2 s clips. The 0.25 s example is
  enough to demonstrate it, but a full second or more gives more reliable
  results (if your part has enough RAM).
- **Continuous real-time:** for streaming (never buffering a full clip),
  replace the global peak-normalisation in `extract_feature_vector()` with a
  fixed gain / AGC. See `stm32/README.md`.

### Memory footprint (STM32C092KCT6: 256 KB flash / 30 KB RAM)

| Item | Size (approx.) |
|------|----------------|
| Hamming window + mel + DCT + scaler + MLP weights | ~73 KB flash |
| Library RAM (FFT scratch + accumulators) | ~5 KB |
| Caller audio buffer (0.25 s float) | 16 KB |

> The library itself is **streaming**: it keeps only a per-frame FFT scratch
> and running accumulators, not a full clip. The caller still supplies the
> audio buffer.

## Tuning for speed (and the STM32 port)

- In `train.py` the neural net is only `(64, 32)` — very fast on CPU.
- `features.py` summarises each clip with mean/std, so inference is one tiny
  matrix multiply — ideal for embedded use.
- `export_c.py` converts the same MFCC + MLP into a single self-contained C
  header (`stm32/drone_model.h`) that runs the full pipeline on the MCU.
  See `stm32/README.md` for wiring it into STM32CubeIDE/HAL.

## Support this project

If this project helped you, please consider supporting its development.

👉 **Donate here:** <https://lsty.modelofengineering.com/>

Thank you for your support!

