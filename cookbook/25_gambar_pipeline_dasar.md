# 25 · Gambar: memuat, mengecilkan, transfer learning cepat

**Kapan:** data berupa gambar. **Kata kunci:** gambar, image, CNN, transfer learning, resnet, PIL, resize, augmentasi, CLIP, folder, kelas.

```python
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image

ROOT = Path("data_latihan/vision/classification/cifar10_small")
tr = pd.read_csv(ROOT / "train.csv")                            # kolom: path, label  (format umum: folder per kelas, atau CSV)

# 1. Muat gambar -> array (N, H, W, 3) uint8. Selalu .convert("RGB") (ada yang grayscale/RGBA/palet)
def muat(paths, ukuran=None):
    out = []
    for p in paths:
        im = Image.open(ROOT / p).convert("RGB")
        out.append(np.asarray(im.resize((ukuran, ukuran)) if ukuran else im))
    return np.stack(out)
tr = tr.sample(400, random_state=0)                        # JANGAN head(): data sering terurut menurut kelas -> head() hanya berisi satu kelas
X = muat(tr["path"]); y = tr["label"].to_numpy()
print(X.shape, X.dtype, "| kelas:", sorted(set(y)))

# 2. Periksa data dulu: ukuran seragam? label seimbang? contoh visual?
print("ukuran:", {im.size for im in [Image.open(ROOT / p) for p in tr["path"].iloc[:50]]}, "| per kelas:", dict(pd.Series(y).value_counts()))

# 3. Jika data lomba berupa FOLDER PER KELAS, buat manifest CSV sekali jalan:
rows = [(str(p.relative_to(ROOT)), p.parent.name) for p in sorted(ROOT.glob("train/*/*.png"))[:5]]
print("manifest dari folder:", rows[:2])

# 4. Transfer learning tercepat: fitur ResNet18 (bobot lokal) + Logistic Regression
from superdb.vision import extract_features
F = extract_features("resnet18", X, size=96, tag="cookbook25")                                # (N, 512) fitur; di-cache
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
print("akurasi CV fitur ResNet18 + LogReg:", cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), F, y, cv=3).mean().round(3))

# 5. Semakin kuat: fine-tuning penuh (notebooks/vision/classification/finetune_*), CLIP zero-shot (clip_zeroshot), augmentasi
from superdb.vision import augment, to_batch
import torch
xb = to_batch(X[:4], 64, "cpu"); print("augmentasi flip/geser/rotasi:", augment(xb).shape)
```

## Jebakan
- **Berkas sering terurut menurut kelas**: `head()` bukan sampel representatif. Gunakan `sample(n, random_state=0)` untuk eksperimen cepat dan `StratifiedKFold`/`train_test_split(stratify=y)`.
- Gambar kecil (32px) di-**upsample** ke 64-224 px saat memakai bobot ImageNet; makin besar makin akurat dan lambat.
- Normalisasi ImageNet (`mean/std`) wajib untuk bobot pretrained (sudah di `to_batch`).
- **Jangan flip horizontal** untuk teks/huruf/arah yang bermakna; augmentasi harus sesuai domain.
- Gambar yang sama/mirip di train dan valid = kebocoran: pisahkan menurut sumber (grup).
- Pilihan model lengkap (klasik, CNN, transfer, CLIP, deteksi, segmentasi, OCR): `notebooks/vision/`.
