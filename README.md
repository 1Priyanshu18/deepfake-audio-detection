# Deepfake Audio Detection

A custom CNN that classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)**, built from scratch (no pretrained weights) with a strong focus on cross-dataset generalization.

**Live demo:** {{ https://huggingface.co/spaces/1Priyanshu18/deepfake-audio-detector }}

---

## Problem

Generative AI can produce highly realistic synthetic speech, usable for impersonation, fraud, and misinformation. The goal is a detector that flags AI-generated audio — and, critically, one that generalizes to generators it has never seen, rather than memorizing the fingerprints of one dataset.

## Why generalization was the central challenge

A naive model trained on a single deepfake dataset tends to learn dataset-specific artifacts (a particular vocoder's signature, recording conditions, even clip-length differences between classes) instead of the general concept of "synthetic speech." Such a model scores near-perfectly on its own validation split but collapses on real-world AI audio from a different generator. This project was designed specifically to avoid that failure mode.

## Methodology

### 1. Data leakage audit
Before training, the Fake-or-Real (`for-norm`) training set was audited for shortcuts. The key finding: real clips were much longer than fake clips (real median ~3.9 s vs fake ~1.6 s). A model could trivially classify by clip length. Sample rate (16 kHz) and channels (mono) were already uniform; loudness differed mildly between classes.

### 2. Preprocessing (leak-neutralizing, shared between training and inference)
The identical pipeline runs in training and in the deployed app:
- Load → mono → resample to 16 kHz
- Force every clip to exactly 3 seconds (repeat-pad short clips, center/random-crop long clips) — this removes clip length as a usable cue
- Loudness normalization to a fixed RMS — removes loudness as a cue
- Log-mel spectrogram (64 mels, n_fft 1024, hop 256), per-spectrogram standardized

Using the same preprocessing function in both places is what guarantees the app sees exactly what the model was trained on.

### 3. Data augmentation (the generalization engine)
Applied only during training, to force the model off dataset-specific shortcuts:
- Codec-style degradation (mu-law companding/quantization, simulating compression)
- Additive Gaussian noise
- Random gain
- Mild time-stretch
- SpecAugment (random time/frequency masking on the spectrogram)

The principle is "train hard, test easy": the model trains on deliberately degraded audio so clean or messy real-world audio is, if anything, easier at inference.

### 4. Model architecture (custom, from scratch)
A small CNN treating the log-mel spectrogram as a single-channel image:
- 4 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool), channels 16 → 32 → 64 → 128
- Global Average Pooling (instead of a large flatten) — fewer parameters, robust to where in the clip an artifact appears
- Dropout-regularized classifier head → single logit; `sigmoid(logit) = P(deepfake)`
- ~106k trainable parameters (deliberately small to discourage memorization)
- No pretrained weights.

### 5. Training
- Loss: `BCEWithLogitsLoss` (with class-balance weighting)
- Optimizer: Adam, `ReduceLROnPlateau`, early stopping on validation loss
- Validation: FoR's **official** `validation` split (no random split, to avoid speaker leakage)

### 6. Cross-dataset fine-tuning
To improve generalization to unseen generators, a small balanced slice of ASVspoof 2019 LA train was merged into the FoR training set and the model fine-tuned (warm restart). The ASVspoof eval split was kept fully unseen as an honest cross-dataset test.

### 7. Threshold calibration
Instead of a blind 0.5 cutoff, the decision threshold was calibrated for balanced performance across both datasets. Deployed threshold: {{ 0.30 }}.

## Results

### FoR test set (in-distribution)
| Metric | Result
|---|---|
| Accuracy | 88.5%
| F1 (fake) | 87.6%
| EER | 7.5%
| Per-class (real / fake) | 98.8% / 78.8%

All required verification thresholds are met.

### ASVspoof 2019 LA eval (cross-dataset, fully unseen — honest stress test)
The model was evaluated on a generator family it never trained on. Performance degrades on this truly-unseen distribution (per-class fake ~50%, higher EER) — this is expected and reflects the open research problem of unseen-attack generalization. It is reported transparently rather than hidden. The fine-tuning step measurably improved cross-dataset fake detection over a FoR-only model.

## Pipeline summary
audio → 16 kHz mono → fixed 3 s → loudness norm → log-mel (64×188)

→ custom CNN → P(deepfake) → threshold → Genuine / Deepfake

## Datasets
- **Fake-or-Real (FoR), `for-norm`** — training, validation, testing
- **ASVspoof 2019 LA** — supplementary cross-dataset training + unseen evaluation

## Tech stack
PyTorch · librosa · audiomentations · Streamlit · scikit-learn