"""Export the trained MLP + MFCC feature pipeline to a self-contained C header.

The header can be #included directly in an STM32 (or any C) project. It contains:

  * the Hamming window, mel filterbank and DCT tables (precomputed here),
  * the scaler mean/scale and the MLP weights/biases,
  * a radix-2 FFT,
  * extract_feature_vector()  -> raw audio becomes the 78-dim feature vector,
  * drone_predict()           -> features become a 'drone probability' in [0,1].

Run:  python export_c.py            (also runs a self-test vs. the Python model)
"""
import os
import math

import numpy as np
import joblib
from scipy.signal import get_window
from scipy.fft import dct as scipy_dct
from scipy.io import wavfile

import config
from features import mel_filterbank, extract_features

# Fixed DSP sizes derived from config.py (must stay in sync with training).
NFFT = 512
FRAME_SAMPLES = int(round(config.FRAME_LEN * config.SAMPLE_RATE))  # 400
HOP_SAMPLES = int(round(config.HOP_LEN * config.SAMPLE_RATE))      # 160
NFFT_HALF = NFFT // 2 + 1                                          # 257
FEAT_DIM = 6 * config.N_MFCC                                       # 78

STM_DIR = os.path.join(config.BASE_DIR, "stm32")
HEADER_PATH = os.path.join(STM_DIR, "drone_model.h")


# --------------------------------------------------------------------------- #
# C code helpers
# --------------------------------------------------------------------------- #
def c_arr(name, arr, per_line=8):
    """Render a numpy array as a C 'static const float name[N]' definition."""
    flat = np.asarray(arr, dtype=np.float32).ravel()
    lines = [f"static const float {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        chunk = ", ".join(f"{v:.9g}f" for v in flat[i:i + per_line])
        tail = "," if i + per_line < len(flat) else ""
        lines.append("    " + chunk + tail)
    lines.append("};")
    return "\n".join(lines)


def build_dct_matrix(n_mfcc, n_mels):
    """Orthonormal DCT-II matrix (first n_mfcc rows), matching scipy.fft.dct."""
    basis = np.eye(n_mels)
    full = scipy_dct(basis, type=2, norm="ortho", axis=0)
    return full[:n_mfcc].astype(np.float32)


# --------------------------------------------------------------------------- #
# Python reference of the exact algorithm the C code implements (for testing)
# --------------------------------------------------------------------------- #
def radix2_fft(re, im):
    """Iterative in-place radix-2 FFT (mirrors the C code exactly)."""
    n = len(re)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    length = 2
    while length <= n:
        ang = -2.0 * math.pi / length
        wlen_re, wlen_im = math.cos(ang), math.sin(ang)
        for i in range(0, n, length):
            w_re, w_im = 1.0, 0.0
            half = length // 2
            for jj in range(half):
                u_re, u_im = re[i + jj], im[i + jj]
                v_re = re[i + jj + half] * w_re - im[i + jj + half] * w_im
                v_im = re[i + jj + half] * w_im + im[i + jj + half] * w_re
                re[i + jj] = u_re + v_re
                im[i + jj] = u_im + v_im
                re[i + jj + half] = u_re - v_re
                im[i + jj + half] = u_im - v_im
                nw_re = w_re * wlen_re - w_im * wlen_im
                w_im = w_re * wlen_im + w_im * wlen_re
                w_re = nw_re
        length <<= 1


def c_reference_features(audio, sr, window, mel, dct_mat):
    """Streaming re-implementation of the C feature extraction (same tables).

    Mirrors the two-pass C code: (1) find the peak, (2) stream frame-by-frame
    accumulating running sum/sumsq. Produces identical features to
    features.extract_features without buffering the whole clip.
    """
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    if n == 0:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    peak = float(np.max(np.abs(audio)))
    inv = 1.0 / peak if peak > 1e-6 else 1.0

    n_eff = n if n >= FRAME_SAMPLES else FRAME_SAMPLES
    n_frames = 1 + (n_eff - FRAME_SAMPLES) // HOP_SAMPLES

    total = np.zeros(3 * config.N_MFCC, dtype=np.float64)
    total_sq = np.zeros(3 * config.N_MFCC, dtype=np.float64)
    prev_m = np.zeros(config.N_MFCC, dtype=np.float64)
    prev_d = np.zeros(config.N_MFCC, dtype=np.float64)
    have_prev = False

    for f in range(n_frames):
        re = np.zeros(NFFT, dtype=np.float64)
        im = np.zeros(NFFT, dtype=np.float64)
        start = f * HOP_SAMPLES
        for i in range(FRAME_SAMPLES):
            idx = start + i
            if idx < n:
                norm = float(audio[idx]) * inv
                pre = norm if idx == 0 else norm - 0.97 * float(audio[idx - 1]) * inv
            else:
                pre = 0.0
            re[i] = pre * window[i]
        radix2_fft(re, im)
        power = (re[:NFFT_HALF] ** 2 + im[:NFFT_HALF] ** 2) / NFFT
        mel_energy = mel @ power
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
        logmel = np.log(mel_energy)
        m = dct_mat @ logmel

        if not have_prev:
            prev_m = m.copy()
            prev_d = np.zeros(config.N_MFCC)
            have_prev = True

        d = m - prev_m
        d2 = d - prev_d
        vals = np.concatenate([m, d, d2])
        total += vals
        total_sq += vals * vals
        prev_m = m
        prev_d = d

    mean = total / n_frames
    var = total_sq / n_frames - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return np.hstack([mean, std]).astype(np.float32)


def self_test():
    """Verify the exported tables + forward pass against the Python model."""
    data = joblib.load(config.MODEL_PATH)
    mlp = data["mlp"]
    scaler = data["scaler"]
    le = data["label_encoder"]

    window = get_window("hamming", FRAME_SAMPLES, fftbins=False)
    mel = mel_filterbank(NFFT, config.SAMPLE_RATE, config.N_MELS)
    dct_mat = build_dct_matrix(config.N_MFCC, config.N_MELS)

    # 1) Validate our radix-2 FFT vs numpy on a random signal.
    rng = np.random.default_rng(0)
    re = rng.standard_normal(NFFT).astype(np.float64)
    im = rng.standard_normal(NFFT).astype(np.float64)
    re0, im0 = re.copy(), im.copy()
    radix2_fft(re, im)
    ref = np.fft.fft(re0 + 1j * im0)
    fft_ok = np.allclose(re, ref.real, atol=1e-6) and np.allclose(im, ref.imag, atol=1e-6)
    print(f"[self-test] radix-2 FFT matches numpy : {fft_ok}")

    # 2) Validate C feature extraction matches features.extract_features.
    drone_idx = int(np.argmax(le.classes_ == "drone"))
    max_err = 0.0
    for cls in config.CLASSES:
        folder = os.path.join(config.DATASET_DIR, cls)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav"))
        # Synthetic clips + a few real (silence-containing) clips.
        files = files[:5] + files[-3:]
        for f in files:
            sr, a = wavfile.read(os.path.join(folder, f))
            if a.ndim > 1:
                a = a.mean(axis=1)
            if sr != config.SAMPLE_RATE:
                a = np.interp(np.linspace(0, len(a) - 1,
                                           int(len(a) * config.SAMPLE_RATE / sr)),
                              np.arange(len(a)), a)
            f_py = extract_features(a, config.SAMPLE_RATE)
            f_c = c_reference_features(a, config.SAMPLE_RATE, window, mel, dct_mat)
            max_err = max(max_err, float(np.max(np.abs(f_py - f_c))))
    print(f"[self-test] feature extraction max error  : {max_err:.3e}")

    # 3) Validate z-score + MLP forward matches sklearn predict_proba.
    sr, a = wavfile.read(os.path.join(config.DATASET_DIR, "drone", "drone_0000.wav"))
    if a.ndim > 1:
        a = a.mean(axis=1)
    f = extract_features(a, config.SAMPLE_RATE)
    x = (f - scaler.mean_) / scaler.scale_
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        x = np.maximum(0, x @ W + b)
    z = x @ mlp.coefs_[-1] + mlp.intercepts_[-1]
    if mlp.out_activation_ == "logistic":
        p1 = float(1.0 / (1.0 + np.exp(-z))[0])       # prob of class index 1
        drone_p = p1 if drone_idx == 1 else 1.0 - p1
    else:
        e = np.exp(z - z.max())
        drone_p = float((e / e.sum())[drone_idx])
    sk_proba = mlp.predict_proba(((f - scaler.mean_) / scaler.scale_).reshape(1, -1))[0]
    print(f"[self-test] sklearn drone prob            : {sk_proba[drone_idx]:.4f}")
    print(f"[self-test] C-logic drone prob            : {drone_p:.4f}")
    print(f"[self-test] forward pass matches          : "
          f"{np.allclose(drone_p, sk_proba[drone_idx], atol=1e-4)}")


# --------------------------------------------------------------------------- #
# C code generation
# --------------------------------------------------------------------------- #
def gen_fft():
    return r"""
/* Iterative in-place radix-2 FFT. Only NFFT-point arrays are ever passed. */
static inline void fft512(float *re, float *im) {
    const int n = NFFT;
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i], ti = im[i];
            re[i] = re[j]; im[i] = im[j];
            re[j] = tr; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -6.2831853071795864769f / (float)len;
        float wlen_re = cosf(ang), wlen_im = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            int half = len >> 1;
            for (int k = 0; k < half; k++) {
                int a = i + k, b = i + k + half;
                float u_re = re[a], u_im = im[a];
                float v_re = re[b] * w_re - im[b] * w_im;
                float v_im = re[b] * w_im + im[b] * w_re;
                re[a] = u_re + v_re; im[a] = u_im + v_im;
                re[b] = u_re - v_re; im[b] = u_im - v_im;
                float nw_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
            }
        }
    }
}
"""


def gen_predict(mlp, drone_idx):
    sizes = [FEAT_DIM] + list(mlp.hidden_layer_sizes)
    n_out = mlp.n_outputs_
    out_act = mlp.out_activation_
    sizes_out = sizes + ([n_out] if n_out > 1 else [1])

    lines = []
    lines.append("")
    lines.append("/* MLP forward pass: z-score -> ReLU hidden layers -> output. */")
    lines.append("static inline void drone_predict(const float *feat, float *drone_prob) {")
    lines.append("    float x[FEAT_DIM];")
    lines.append("    for (int i = 0; i < FEAT_DIM; i++)")
    lines.append("        x[i] = (feat[i] - MEAN[i]) / SCALE[i];")

    prev = "x"
    n_layers = len(sizes_out) - 1
    for l in range(n_layers):
        in_size = sizes_out[l]
        out_size = sizes_out[l + 1]
        is_last = (l == n_layers - 1)
        if not is_last:
            cur = f"h{l}"
            lines.append(f"    float {cur}[{out_size}];")
            lines.append(f"    for (int j = 0; j < {out_size}; j++) {{")
            lines.append(f"        float s = B{l}[j];")
            lines.append(f"        for (int i = 0; i < {in_size}; i++)")
            lines.append(f"            s += {prev}[i] * W{l}[i * {out_size} + j];")
            lines.append(f"        {cur}[j] = s > 0.0f ? s : 0.0f;")
            lines.append("    }")
            prev = cur
        else:
            if out_act == "logistic":
                lines.append("    float z = B%d[0];" % l)
                lines.append(f"    for (int i = 0; i < {in_size}; i++) z += {prev}[i] * W{l}[i];")
                lines.append("    float p1 = 1.0f / (1.0f + expf(-z));  /* prob of class index 1 */")
                lines.append(f"    *drone_prob = (DRONE_INDEX == 1) ? p1 : (1.0f - p1);")
            else:
                lines.append(f"    float z[{out_size}];")
                lines.append(f"    for (int j = 0; j < {out_size}; j++) {{")
                lines.append(f"        float s = B{l}[j];")
                lines.append(f"        for (int i = 0; i < {in_size}; i++)")
                lines.append(f"            s += {prev}[i] * W{l}[i * {out_size} + j];")
                lines.append("        z[j] = s;")
                lines.append("    }")
                lines.append("    float mx = z[0];")
                lines.append(f"    for (int j = 1; j < {out_size}; j++) if (z[j] > mx) mx = z[j];")
                lines.append("    float sum = 0.0f;")
                lines.append(f"    for (int j = 0; j < {out_size}; j++) {{ z[j] = expf(z[j] - mx); sum += z[j]; }}")
                lines.append("    *drone_prob = z[DRONE_INDEX] / sum;")
    lines.append("}")
    lines.append("")
    lines.append("/* Convenience: raw audio -> drone probability in [0,1]. */")
    lines.append("static inline float drone_detect(const float *audio, int len) {")
    lines.append("    float feat[FEAT_DIM];")
    lines.append("    float prob = 0.0f;")
    lines.append("    extract_feature_vector(audio, len, feat);")
    lines.append("    drone_predict(feat, &prob);")
    lines.append("    return prob;")
    lines.append("}")
    return "\n".join(lines)


def generate_header(data):
    mlp = data["mlp"]
    scaler = data["scaler"]
    le = data["label_encoder"]
    classes = [str(c) for c in le.classes_]
    drone_idx = classes.index("drone")

    window = get_window("hamming", FRAME_SAMPLES, fftbins=False).astype(np.float32)
    mel = mel_filterbank(NFFT, config.SAMPLE_RATE, config.N_MELS).astype(np.float32)
    dct_mat = build_dct_matrix(config.N_MFCC, config.N_MELS).astype(np.float32)

    parts = []
    parts.append("/* Auto-generated by export_c.py. Do not edit by hand. */")
    parts.append("/* Drone-sound detector for STM32 / embedded C.            */")
    parts.append("")
    parts.append("#ifndef DRONE_MODEL_H")
    parts.append("#define DRONE_MODEL_H")
    parts.append("")
    parts.append("#include <math.h>")
    parts.append("")
    parts.append(f"#define SAMPLE_RATE {config.SAMPLE_RATE}")
    parts.append(f"#define N_MFCC {config.N_MFCC}")
    parts.append(f"#define N_MELS {config.N_MELS}")
    parts.append(f"#define FRAME_SAMPLES {FRAME_SAMPLES}")
    parts.append(f"#define HOP_SAMPLES {HOP_SAMPLES}")
    parts.append(f"#define NFFT {NFFT}")
    parts.append(f"#define NFFT_HALF {NFFT_HALF}")
    parts.append(f"#define FEAT_DIM {FEAT_DIM}")
    parts.append(f"#define DRONE_INDEX {drone_idx}")
    parts.append(f"/* class order (label encoder): {classes} */")
    parts.append("")
    parts.append(c_arr("HAMMING", window))
    parts.append("")
    parts.append(c_arr("MEL", mel))
    parts.append("")
    parts.append(c_arr("DCT", dct_mat))
    parts.append("")
    parts.append(c_arr("MEAN", scaler.mean_))
    parts.append("")
    parts.append(c_arr("SCALE", scaler.scale_))
    parts.append("")

    for l in range(len(mlp.coefs_)):
        parts.append(c_arr(f"W{l}", mlp.coefs_[l].astype(np.float32)))
        parts.append("")
        parts.append(c_arr(f"B{l}", mlp.intercepts_[l].astype(np.float32)))
        parts.append("")

    parts.append(gen_fft())
    parts.append(gen_features())
    parts.append(gen_predict(mlp, drone_idx))
    parts.append("")
    parts.append("#endif /* DRONE_MODEL_H */")
    parts.append("")
    return "\n".join(parts)


def main():
    if not os.path.exists(config.MODEL_PATH):
        print("No model found. Run 'python train.py' first.")
        return
    data = joblib.load(config.MODEL_PATH)
    if "mlp" not in data:
        print("Saved model has no MLP. Re-run 'python train.py' (it now saves the MLP).")
        return

    os.makedirs(STM_DIR, exist_ok=True)
    header = generate_header(data)
    with open(HEADER_PATH, "w", newline="\n") as fh:
        fh.write(header)
    print(f"Wrote {HEADER_PATH} ({len(header)} bytes)")

    # Verify the exported math matches the Python model.
    self_test()


def gen_features():
    return r"""
/*
 * Streaming feature extractor: raw audio -> the 78-dim feature vector the
 * Python model was trained on (MFCC + deltas, mean + std over time).
 *
 * Unlike a naive implementation this does NOT buffer the whole clip. It makes
 * two passes over the caller's buffer (peak, then frame-by-frame) and keeps
 * only a per-frame FFT scratch + running sum/sumsq accumulators.
 *
 * RAM used here: re/im[512] (4 KB) + a few small stack arrays (~1 KB).
 * The caller supplies the audio buffer itself.
 */
static inline void extract_feature_vector(const float *audio, int len,
                                          float feat[FEAT_DIM]) {
    static float re[NFFT], im[NFFT];

    if (len <= 0) {
        for (int i = 0; i < FEAT_DIM; i++) feat[i] = 0.0f;
        return;
    }

    /* 1) amplitude normalisation: find the global peak first */
    float peak = 0.0f;
    for (int i = 0; i < len; i++) {
        float a = audio[i] < 0.0f ? -audio[i] : audio[i];
        if (a > peak) peak = a;
    }
    float inv = (peak > 1e-6f) ? 1.0f / peak : 1.0f;

    int n_eff = (len < FRAME_SAMPLES) ? FRAME_SAMPLES : len;
    int n_frames = 1 + (n_eff - FRAME_SAMPLES) / HOP_SAMPLES;

    float sum[3 * N_MFCC] = {0};
    float sumsq[3 * N_MFCC] = {0};
    float prev_m[N_MFCC], prev_d[N_MFCC];
    int have_prev = 0;

    /* 2) frame-by-frame streaming (window -> FFT -> mel -> DCT -> stats) */
    for (int f = 0; f < n_frames; f++) {
        for (int i = 0; i < NFFT; i++) { re[i] = 0.0f; im[i] = 0.0f; }
        int start = f * HOP_SAMPLES;
        for (int i = 0; i < FRAME_SAMPLES; i++) {
            int idx = start + i;
            float s = 0.0f;
            if (idx < len) {
                float norm = audio[idx] * inv;
                float pre = (idx == 0) ? norm
                                       : norm - 0.97f * (audio[idx - 1] * inv);
                s = pre;
            }
            re[i] = s * HAMMING[i];
        }
        fft512(re, im);

        float logmel[N_MELS];
        for (int m = 0; m < N_MELS; m++) {
            float e = 0.0f;
            for (int k = 0; k < NFFT_HALF; k++)
                e += MEL[m * NFFT_HALF + k] * (re[k] * re[k] + im[k] * im[k]) / NFFT;
            if (e <= 0.0f) e = 2.220446049250313e-16f; /* match np.finfo(float).eps */
            logmel[m] = logf(e);
        }
        float mfcc[N_MFCC];
        for (int c = 0; c < N_MFCC; c++) {
            float s = 0.0f;
            for (int m = 0; m < N_MELS; m++)
                s += DCT[c * N_MELS + m] * logmel[m];
            mfcc[c] = s;
        }

        if (!have_prev) {
            for (int c = 0; c < N_MFCC; c++) { prev_m[c] = mfcc[c]; prev_d[c] = 0.0f; }
            have_prev = 1;
        }

        /* deltas + running statistics */
        for (int c = 0; c < N_MFCC; c++) {
            float m = mfcc[c];
            float d = m - prev_m[c];
            float d2 = d - prev_d[c];
            float vals[3] = {m, d, d2};
            for (int k = 0; k < 3; k++) {
                int idx = k * N_MFCC + c;
                sum[idx] += vals[k];
                sumsq[idx] += vals[k] * vals[k];
            }
            prev_m[c] = m;
            prev_d[c] = d;
        }
    }

    /* 3) mean / std -> feature vector */
    for (int i = 0; i < 3 * N_MFCC; i++) {
        float mean = sum[i] / n_frames;
        float var = sumsq[i] / n_frames - mean * mean;
        if (var < 0.0f) var = 0.0f;
        feat[i] = mean;
        feat[3 * N_MFCC + i] = sqrtf(var);
    }
}
"""


if __name__ == "__main__":
    main()



