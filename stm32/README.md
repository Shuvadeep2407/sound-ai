# STM32 port (drone-sound detector)

This folder contains a **self-contained C header** (`drone_model.h`) generated
from your trained Python model. It runs the full pipeline on the MCU:

```
raw audio (16 kHz) -> MFCC (+deltas) -> mean/std features -> small MLP -> drone probability
```

## What is in `drone_model.h`

| Item | Size (approx.) |
|------|----------------|
| Hamming window (400) | 1.6 KB |
| Mel filterbank (40 x 257) | 41 KB |
| DCT matrix (13 x 40) | 2 KB |
| Scaler mean/scale (78 x 2) | 0.6 KB |
| MLP weights + biases (78→64→32→1) | 28 KB |
| **Total tables** | **~73 KB flash** |

It also provides two functions:

```c
static inline void  extract_feature_vector(const float *audio, int len, float feat[FEAT_DIM]);
static inline void  drone_predict(const float *feat, float *drone_prob);
static inline float drone_detect(const float *audio, int len);   // returns prob in [0,1]
```

`drone_detect()` returns the probability that the clip is a **drone**:
`> 0.5` = drone, `<= 0.5` = no drone.

## How to regenerate

After re-training (or changing classes/hyper-parameters), run:

```powershell
python train.py
python export_c.py
```

## Memory footprint (target: STM32C092KCT6 — 256 KB flash / 30 KB RAM)

- **Flash**: ~73 KB of tables + weights — fits easily in 256 KB.
- **RAM (this library)**: **~5 KB**. `extract_feature_vector()` is **streaming**:
  it makes two passes over the caller's buffer (peak, then frame-by-frame) and
  keeps only a per-frame FFT scratch (`re[512]`/`im[512]`, 4 KB) + small
  accumulators. There is **no full-clip buffer** anymore.
- **The caller still supplies the audio buffer.** `float32` audio costs
  64 KB/second, so a full 2 s clip (128 KB) will not fit in 30 KB SRAM. For a
  demo, feed a shorter window (0.25 s = 16 KB) or store samples as `int16`.
- For **continuous real-time** detection, replace the global peak-normalisation
  with a fixed gain / AGC so you never need the whole clip at once.

## Linking

Add `-lm` (the math library) to your linker flags — the header uses
`cosf/sinf/logf/expf/sqrtf`.

## Gotchas

- Sample rate **must** be 16 kHz (set `SAMPLE_RATE` in `config.py` + regenerate if you change it).
- Input samples are assumed to be `float` in `[-1.0, 1.0]`.
- The model was trained on ~2-second clips; feed at least a full second for reliable results.
