# Deepfake Audio Detector

> Binary speech authentication: classifying an audio clip as **Genuine (Human)** or **Deepfake (AI-Generated)** from log-mel spectrograms, trained across two datasets for cross-corpus robustness.

---

## Problem

Modern text-to-speech and voice-conversion systems produce speech that is hard to distinguish from real recordings by ear. The task is to decide, from a short audio clip alone, whether the voice is genuinely human or machine-generated. The core difficulty is generalisation: a detector that memorises one corpus's synthesis artefacts often collapses on unseen generators, codecs, or recording conditions. The goal is a model that holds up across datasets, not just on an in-distribution test split.

---

## Approach

A single pipeline from raw waveform to verdict, trained once on a merged corpus rather than per-dataset.

| Stage | What it does |
|-------|--------------|
| Audio I/O | Load → mono → resample to 16 kHz |
| Length fix | Center-crop or tile to exactly 3 s (random crop during training) |
| Loudness norm | Scale to a target RMS so volume isn't a shortcut feature |
| Feature | 64-band log-mel spectrogram, power-to-dB, per-clip standardised |
| Model | 4-block CNN → global average pool → MLP head → 1 logit |
| Decision | Sigmoid → tuned threshold → Genuine / Deepfake |

### Key design choices
- **Cross-corpus training:** FoR and ASVspoof 2019 (LA) are merged into one training set so the model can't lean on a single corpus's artefacts. ASVspoof's official eval split is held out as a fully unseen test.
- **Class imbalance handled in the loss:** `BCEWithLogitsLoss` with `pos_weight = n_real / n_fake` instead of resampling.
- **Augmentation only on train:** Gaussian noise, gain, time-stretch, plus a μ-law "cheap codec" degradation to mimic compression artefacts.
- **Threshold tuned for balanced accuracy:** the deployment threshold is swept over a held-out set to maximise balanced accuracy, not fixed at 0.5.

---

## Feature & Augmentation Pipeline

Audio is reduced to a fixed-shape spectrogram so a 2D CNN can be applied directly.

| Group | Detail |
|-------|--------|
| Sampling | 16 kHz mono, 3.0 s (`N_SAMPLES = 48000`) |
| Spectrogram | `n_fft=1024`, `hop_length=256`, `n_mels=64` → ~188 frames |
| Normalisation | loudness RMS norm + per-clip mean/std standardisation |
| Train-only aug | `AddGaussianNoise`, `Gain(±6 dB)`, `TimeStretch(0.9–1.1)`, μ-law codec degrade |

---

## Model

A compact convolutional classifier (`DeepfakeCNN`).

| Component | Detail |
|-----------|--------|
| Features | 4 × `Conv3×3 → BatchNorm → ReLU → MaxPool`, channels 16 → 32 → 64 → 128 |
| Pooling | `AdaptiveAvgPool2d(1)` (handles variable time length) |
| Head | `Dropout → Linear(128→64) → ReLU → Dropout → Linear(64→1)` |
| Loss | `BCEWithLogitsLoss` with positive-class weighting |
| Optimiser | Adam, `lr=1e-3`, `weight_decay=1e-4`, `ReduceLROnPlateau` |

---

## Results

Reported on the in-distribution FoR test split and the unseen ASVspoof eval split.

| Metric | Value |
|--------|-------|
| Accuracy | 87.4% |
| F1-score | 86.9% |
| EER | 8.1% |

The deployment threshold is selected by sweeping `t` over `[0.05, 0.6]` and picking the value that maximises balanced accuracy across the combined held-out set, rather than defaulting to 0.5.

---

## Inference Pipeline

The deployed path mirrors training preprocessing exactly to avoid train/serve skew:

1. **Preprocess** — load, fix to 3 s, loudness-normalise, log-mel, standardise.
2. **Forward pass** — single-logit CNN → sigmoid → `P(deepfake)`.
3. **Verdict** — compare against the tuned threshold stored in the model bundle.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Audio | `librosa`, `audiomentations` |
| Deep learning | `torch` |
| Data | `numpy` |
| Metrics | `scikit-learn` |
| App / deploy | `streamlit`, `huggingface` |

---

## Notes & Limitations

- Only the middle 3 s of long clips is analysed.
- Strongest on audio resembling the FoR / ASVspoof training distribution; novel generators, codecs, or languages may degrade accuracy.
- Research/educational project — not a forensic-grade tool.