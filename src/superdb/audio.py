"""Audio: loader WAV, fitur MFCC/log-mel, embedding pretrained (wav2vec2, Whisper), transkripsi Whisper, CNN spektrogram.
Pendaftaran notebook ada di audio_reg.py. Bobot dari pretrained/<nama>/ (offline)."""
import hashlib
import time

import numpy as np

from .paths import ROOT
from .cachekeys import array_key, file_key, weights_signature

SR = 16000


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def pretrained_dir(name):
    d = ROOT / "pretrained" / name
    if not (d / "config.json").exists():
        raise FileNotFoundError(f"Bobot '{name}' belum ada di {d}. Saat persiapan (ada internet) pasang ulang lewat tools/download_pretrained_audio.py {name}")
    return d


# ================================================================== data
def load_wavs(paths, root, length=SR, cache=True):
    """Daftar path WAV -> array float32 (N, length) mono 16 kHz, dipotong/di-pad nol ke `length` sampel. Di-cache."""
    import soundfile as sf
    paths = [str(p) for p in paths]
    key = file_key(paths, root, length, SR)
    f = ROOT / "outputs" / "_cache" / f"wav_{key}.npy"
    if cache and f.exists():
        return np.load(f)
    out = []
    for p in paths:
        x, sr = sf.read(f"{root}/{p}", dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        x = x[:length] if len(x) >= length else np.pad(x, (0, length - len(x)))
        out.append(x.astype(np.float32))
    X = np.stack(out)
    if cache:
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, X)
    return X


def plot_wave_mel(x, title="", path=None):
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.6))
    ax[0].plot(np.arange(len(x)) / SR, x, lw=.6); ax[0].set(title=f"{title} gelombang", xlabel="detik")
    M = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=SR, n_fft=512, hop_length=160, n_mels=64), ref=np.max)
    librosa.display.specshow(M, sr=SR, hop_length=160, x_axis="time", y_axis="mel", ax=ax[1]); ax[1].set_title("mel-spektrogram (dB)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


# ================================================================== fitur klasik
def mfcc_stats(X, n_mfcc=20):
    """Ringkasan MFCC per klip: mean/std/min/max tiap koefisien + mean/std delta + ZCR + RMS. (N, d)"""
    import librosa
    feats = []
    for x in X:
        m = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=n_mfcc, n_fft=512, hop_length=160)
        d = librosa.feature.delta(m)
        zcr = librosa.feature.zero_crossing_rate(x, frame_length=512, hop_length=160); rms = librosa.feature.rms(y=x, frame_length=512, hop_length=160)
        feats.append(np.concatenate([m.mean(1), m.std(1), m.min(1), m.max(1), d.mean(1), d.std(1), [zcr.mean(), zcr.std(), rms.mean(), rms.std()]]))
    return np.nan_to_num(np.asarray(feats, np.float32))


# ================================================================== embedding pretrained
class Wav2VecEmbedder:
    """Embedding wav2vec2-base: rata-rata semua lapisan tersembunyi, dirata-ratakan terhadap waktu (768 dim). Di-cache."""

    def __init__(self, name="wav2vec2_base", batch_size=32):
        self.name, self.bs = name, batch_size

    def encode(self, X):
        import torch
        from transformers import Wav2Vec2Model
        key = array_key(X, self.name, weights_signature(pretrained_dir(self.name)))
        f = ROOT / "outputs" / "_cache" / f"emb_{self.name}_{key}.npy"
        if f.exists():
            return np.load(f)
        dev = device()
        m = Wav2Vec2Model.from_pretrained(str(pretrained_dir(self.name)), local_files_only=True).to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.bs):
                xb = torch.from_numpy(X[i:i + self.bs]).to(dev)
                xb = (xb - xb.mean(1, keepdim=True)) / (xb.std(1, keepdim=True) + 1e-7)           # normalisasi seperti feature extractor wav2vec2
                hs = m(xb, output_hidden_states=True).hidden_states
                out.append(torch.stack(hs).mean(0).mean(1).float().cpu().numpy())
        E = np.vstack(out)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, E)
        return E


class WhisperEmbedder:
    """Embedding dari encoder Whisper-tiny: rata-rata 50 frame pertama (≈1 detik) -> 384 dim. Di-cache."""

    def __init__(self, name="whisper_tiny", batch_size=32, frames=50):
        self.name, self.bs, self.frames = name, batch_size, frames

    def encode(self, X):
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel
        key = array_key(X, self.name, self.frames, weights_signature(pretrained_dir(self.name)))
        f = ROOT / "outputs" / "_cache" / f"emb_{self.name}_{key}.npy"
        if f.exists():
            return np.load(f)
        d = str(pretrained_dir(self.name)); dev = device()
        fe = WhisperFeatureExtractor.from_pretrained(d, local_files_only=True); enc = WhisperModel.from_pretrained(d, local_files_only=True).encoder.to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.bs):
                feats = fe(list(X[i:i + self.bs]), sampling_rate=SR, return_tensors="pt").input_features.to(dev)
                h = enc(feats).last_hidden_state[:, :self.frames]
                out.append(h.mean(1).float().cpu().numpy())
        E = np.vstack(out)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, E)
        return E


def whisper_transcribe(X, name="whisper_tiny", language="en", batch_size=16, max_new_tokens=10):
    """Transkripsi Whisper offline. Kembalikan list teks."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    d = str(pretrained_dir(name)); dev = device()
    proc = WhisperProcessor.from_pretrained(d, local_files_only=True); model = WhisperForConditionalGeneration.from_pretrained(d, local_files_only=True).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            feats = proc(list(X[i:i + batch_size]), sampling_rate=SR, return_tensors="pt").input_features.to(dev)
            ids = model.generate(feats, language=language, task="transcribe", max_new_tokens=max_new_tokens)
            out += proc.batch_decode(ids, skip_special_tokens=True)
    return [t.strip() for t in out]


# ================================================================== CNN log-mel (GPU)
def make_logmel_cnn(n_classes, n_mels=64):
    import torch.nn as nn
    import torchaudio

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=160, n_mels=n_mels)
            self.db = torchaudio.transforms.AmplitudeToDB()
            blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
            self.cnn = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

        def forward(self, wav):                                   # wav: (B, T)
            m = self.db(self.mel(wav)).unsqueeze(1)
            m = (m - m.mean((1, 2, 3), keepdim=True)) / (m.std((1, 2, 3), keepdim=True) + 1e-5)
            return self.cnn(m)
    return Net()


def augment_wave(x, max_shift=1600, noise_snr_db=(15, 30)):
    """Augmentasi gelombang (GPU): geser waktu acak, tambah derau pada SNR acak, gain acak. x: (B, T)."""
    import torch
    B, T = x.shape
    sh = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
    x = torch.stack([torch.roll(x[i], int(sh[i])) for i in range(B)])
    snr = torch.empty(B, 1, device=x.device).uniform_(*noise_snr_db)
    p = x.pow(2).mean(1, keepdim=True) + 1e-9
    x = x + torch.randn_like(x) * torch.sqrt(p / (10 ** (snr / 10)))
    return x * torch.empty(B, 1, device=x.device).uniform_(0.7, 1.3)


def train_audio_cnn(net, Xtr, ytr, Xva, yva, epochs=20, lr=2e-3, batch_size=64, aug=False, seed=42, log=print):
    import torch
    torch.manual_seed(seed)
    dev = device(); net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(Xtr) / batch_size)), pct_start=0.2)
    Xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(np.asarray(ytr)).to(dev)
    rng = np.random.default_rng(seed); hist, t0 = [], time.time()
    for ep in range(epochs):
        net.train(); perm = rng.permutation(len(Xt)); tl = 0.0
        for i in range(0, len(perm), batch_size):
            idx = torch.from_numpy(perm[i:i + batch_size]).to(dev)
            xb = augment_wave(Xt[idx]) if aug else Xt[idx]
            loss = torch.nn.functional.cross_entropy(net(xb), yt[idx]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
        acc = float((predict_proba_cnn(net, Xva).argmax(1) == np.asarray(yva)).mean())
        hist.append({"epoch": ep + 1, "train_loss": tl / len(Xt), "valid_acc": acc, "detik": round(time.time() - t0)}); log(hist[-1])
    return hist


def predict_proba_cnn(net, X, batch_size=256):
    import torch
    dev = next(net.parameters()).device; net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(torch.softmax(net(torch.from_numpy(X[i:i + batch_size]).to(dev)), -1).cpu().numpy())
    return np.vstack(out)
