import numpy as np
import librosa
import torch
import torch.nn as nn


class DeepfakeCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), 
                                nn.BatchNorm2d(o),
                                nn.ReLU(inplace=True), 
                                nn.MaxPool2d(2))
        
        self.features = nn.Sequential(block(1, 16), 
                                      block(16, 32),
                                      block(32, 64), 
                                      block(64, 128))
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(nn.Flatten(), 
                                        nn.Dropout(dropout), 
                                        nn.Linear(128, 64),
                                        nn.ReLU(inplace=True), 
                                        nn.Dropout(dropout), 
                                        nn.Linear(64, 1))

    def forward(self, x):
        return self.classifier(self.gap(self.features(x))).squeeze(1)


# preprocessing
def _fix_length(y, n_samples):
    n = len(y)
    if n < n_samples:
        y = np.tile(y, int(np.ceil(n_samples / n)))[:n_samples]
    elif n > n_samples:
        start = (n - n_samples) // 2
        y = y[start:start + n_samples]
    return y


def preprocess(path, cfg):
    # Load -> mono/resample -> fixed length -> loudness norm -> log-mel -> standardize.
    y, _ = librosa.load(path, sr=cfg['SR'], mono=True)
    y = _fix_length(y, cfg['N_SAMPLES'])
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    y = y * (cfg['TARGET_RMS'] / rms)
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg['SR'], n_fft=cfg['N_FFT'],
        hop_length=cfg['HOP_LENGTH'], n_mels=cfg['N_MELS'])
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = (lm - lm.mean()) / (lm.std() + 1e-9)
    return lm.astype(np.float32)


def load_model(path="deploy_model.pt", device="cpu"):
    bundle = torch.load(path, map_location=device)
    cfg = bundle['config']
    model = DeepfakeCNN().to(device)
    model.load_state_dict(bundle['model_state'])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_file(path, model, cfg, device="cpu"):
    feat = preprocess(path, cfg)
    x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,64,T)
    p_fake = torch.sigmoid(model(x)).item()
    verdict = "Deepfake (AI-Generated)" if p_fake > cfg['threshold'] else "Genuine (Human)"
    return verdict, p_fake