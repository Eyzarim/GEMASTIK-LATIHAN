"""Gambar: loader, fitur klasik, jaringan (scratch/transfer), pelatihan di GPU, ekstraksi fitur dengan cache, CLIP.
Pendaftaran notebook ada di vision_reg.py. Bobot dibaca dari pretrained/<nama>/ (offline)."""
import hashlib
import time

import numpy as np
import pandas as pd

from .paths import ROOT
from .cachekeys import array_key, file_key, weights_signature

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================== data
def load_images(paths, root, size=None, cache=True):
    """Daftar path -> array uint8 (N, H, W, 3). Di-cache ke outputs/_cache agar pemanggilan berikutnya instan."""
    from PIL import Image
    paths = [str(p) for p in paths]
    key = file_key(paths, root, size)
    f = ROOT / "outputs" / "_cache" / f"img_{key}.npy"
    if cache and f.exists():
        return np.load(f)
    out = []
    for p in paths:
        im = Image.open(f"{root}/{p}").convert("RGB")
        if size:
            im = im.resize((size, size), Image.BILINEAR)
        out.append(np.asarray(im))
    X = np.stack(out)
    if cache:
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, X)
    return X


def pretrained_path(name, fname="model.pth"):
    p = ROOT / "pretrained" / name / fname
    if not p.exists():
        raise FileNotFoundError(f"Bobot '{name}' belum ada di {p}. Saat persiapan (ada internet): python tools/download_pretrained_vision.py {name}")
    return p


def show_grid(X, y=None, n=12, cols=6, path=None):
    import matplotlib.pyplot as plt
    n = min(n, len(X))
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.7 * rows), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for i in range(n):
        ax.ravel()[i].imshow(X[i]); ax.ravel()[i].set_title("" if y is None else str(y[i]), fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


# ================================================================== fitur klasik
def hog_features(X, orientations=9, ppc=8, cpb=2):
    from skimage.color import rgb2gray
    from skimage.feature import hog
    return np.stack([hog(rgb2gray(x), orientations=orientations, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb)) for x in X])


def color_hist_features(X, bins=8):
    """Histogram warna per kanal (RGB) + rata-rata/std per kanal."""
    f = []
    for x in X:
        h = [np.histogram(x[..., c], bins=bins, range=(0, 255))[0] / x[..., c].size for c in range(3)]
        f.append(np.concatenate(h + [x.reshape(-1, 3).mean(0) / 255, x.reshape(-1, 3).std(0) / 255]))
    return np.asarray(f)


def lbp_features(X, P=8, R=1):
    from skimage.color import rgb2gray
    from skimage.feature import local_binary_pattern
    out = []
    for x in X:
        g = (rgb2gray(x) * 255).astype(np.uint8)
        lbp = local_binary_pattern(g, P, R, method="uniform")
        out.append(np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)[0])
    return np.asarray(out)


def pixel_features(X, size=16):
    """Piksel mentah RGB diperkecil ke size x size lalu dihamparkan (untuk baseline PCA+SVM)."""
    import torch
    t = torch.tensor(X).permute(0, 3, 1, 2).float() / 255
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="area")
    return t.reshape(len(X), -1).numpy()


# ================================================================== tensor, augmentasi
def to_batch(Xb, size, dev, normalize=True):
    """uint8 (B,H,W,3) -> float (B,3,size,size) di device, ternormalisasi ImageNet."""
    import torch
    t = torch.from_numpy(Xb).to(dev).permute(0, 3, 1, 2).float() / 255
    if t.shape[-1] != size:
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    if normalize:
        t = (t - torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)) / torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return t


def augment(x, max_shift=0.12, max_rot=10, flip=True):
    """Augmentasi per sampel di GPU: flip horizontal + geser + rotasi kecil (lewat affine grid)."""
    import math
    import torch
    B = x.shape[0]
    ang = (torch.rand(B, device=x.device) * 2 - 1) * math.radians(max_rot)
    sx = (torch.rand(B, device=x.device) * 2 - 1) * max_shift * 2
    sy = (torch.rand(B, device=x.device) * 2 - 1) * max_shift * 2
    fl = torch.where(torch.rand(B, device=x.device) < 0.5, -1.0, 1.0) if flip else torch.ones(B, device=x.device)
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = torch.cos(ang) * fl; theta[:, 0, 1] = -torch.sin(ang); theta[:, 0, 2] = sx
    theta[:, 1, 0] = torch.sin(ang) * fl; theta[:, 1, 1] = torch.cos(ang); theta[:, 1, 2] = sy
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)


# ================================================================== jaringan
def small_cnn(num_classes, width=32):
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(3, width), blk(width, width * 2), blk(width * 2, width * 4), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(width * 4, num_classes))


_HEAD = {"resnet18": "fc", "resnet50": "fc", "mobilenet_v3_small": "classifier.3", "efficientnet_b0": "classifier.1",
         "convnext_tiny": "classifier.2", "vit_b_16": "heads.head"}


def _get_set(model, dotted, new=None):
    parts = dotted.split(".")
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if new is None:
        return obj[int(last)] if last.isdigit() else getattr(obj, last)
    if last.isdigit():
        obj[int(last)] = new
    else:
        setattr(obj, last, new)


def tv_model(name, num_classes=None, pretrained=True, freeze=False):
    """Backbone torchvision dengan bobot lokal. num_classes=None -> kepala dibuang (ekstraktor fitur); selain itu kepala diganti."""
    import torch
    import torch.nn as nn
    import torchvision.models as M
    model = getattr(M, name)(weights=None)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path(name), map_location="cpu"))
    head = _HEAD[name]
    old = _get_set(model, head)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    _get_set(model, head, nn.Identity() if num_classes is None else nn.Linear(old.in_features, num_classes))
    return model


def train_classifier(model, Xtr, ytr, Xva, yva, size=64, epochs=10, lr=1e-3, batch_size=64, weight_decay=1e-4,
                     aug=False, seed=42, log=print, normalize=True):
    """Pelatihan standar: AdamW + cosine LR + mixed precision (GPU). Kembalikan riwayat per epoch."""
    import torch
    torch.manual_seed(seed)
    dev = device()
    model.to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    steps = epochs * int(np.ceil(len(Xtr) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    ytr_t = torch.tensor(np.asarray(ytr)).to(dev)
    rng = np.random.default_rng(seed)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tl, n = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb = to_batch(Xtr[idx], size, dev, normalize)
            if aug:
                xb = augment(xb)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(xb), ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(idx); n += len(idx)
        acc = float((predict_proba(model, Xva, size, normalize=normalize).argmax(1) == np.asarray(yva)).mean())
        hist.append({"epoch": ep + 1, "train_loss": tl / n, "valid_acc": acc, "detik": round(time.time() - t0)})
        log(hist[-1])
    return hist


def predict_proba(model, X, size=64, batch_size=256, normalize=True):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        for i in range(0, len(X), batch_size):
            out.append(torch.softmax(model(to_batch(X[i:i + batch_size], size, dev, normalize)).float(), -1).cpu().numpy())
    return np.vstack(out)


def extract_features(name, X, size=64, batch_size=128, tag=""):
    """Fitur penultimate dari backbone beku; di-cache. Bobot beku => aman dihitung sebelum split."""
    import torch
    key = array_key(X, name, size, tag, weights_signature(ROOT / "pretrained" / name))
    f = ROOT / "outputs" / "_cache" / f"feat_{name}_{key}.npy"
    if f.exists():
        return np.load(f)
    dev = device()
    m = tv_model(name, None, True).to(dev).eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
        for i in range(0, len(X), batch_size):
            out.append(m(to_batch(X[i:i + batch_size], size, dev)).float().flatten(1).cpu().numpy())
    F = np.vstack(out)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, F)
    return F


# ================================================================== CLIP
class ClipEmbedder:
    """CLIP (gambar-teks) dari pretrained/clip_b32. Embedding ternormalisasi."""

    def __init__(self, name="clip_b32"):
        from transformers import CLIPModel, CLIPProcessor
        d = ROOT / "pretrained" / name
        if not (d / "config.json").exists():
            raise FileNotFoundError(f"Bobot CLIP belum ada di {d}. Jalankan: python tools/download_pretrained_vision.py {name}")
        self.dev = device()
        self.model = CLIPModel.from_pretrained(str(d), local_files_only=True).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(str(d), local_files_only=True)

    @staticmethod
    def _tensor(o):
        """Versi transformers berbeda mengembalikan Tensor atau objek output; ambil Tensor-nya."""
        import torch
        if isinstance(o, torch.Tensor):
            return o
        return getattr(o, "pooler_output", None) if getattr(o, "pooler_output", None) is not None else o[0]

    def images(self, X, batch_size=128):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                inp = self.proc(images=list(X[i:i + batch_size]), return_tensors="pt").to(self.dev)
                e = self._tensor(self.model.get_image_features(**inp))
                out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.vstack(out)

    def texts(self, prompts, batch_size=32):
        import torch
        prompts = list(prompts)
        out = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                inp = self.proc(text=prompts[i:i + batch_size], return_tensors="pt", padding=True,
                                truncation=True, max_length=77).to(self.dev)
                e = self._tensor(self.model.get_text_features(**inp))
                out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.vstack(out)
