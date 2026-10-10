# 09 · Split data yang benar (train/valid, grup, waktu)

**Kapan:** sebelum melatih apa pun. Salah split = skor lokal bohong. **Kata kunci:** split, train_test_split, stratify, group, GroupKFold, time series split, validation, holdout.

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, TimeSeriesSplit
df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")
y = df["survived"]; X = df.drop(columns=["survived", "alive"])

# ---- Kasus 1: baris saling independen -> split acak. Klasifikasi: stratify agar proporsi kelas sama
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(ytr.mean().round(3), yva.mean().round(3))      # proporsi mirip

# ---- Kasus 2: ada GRUP (pasien sama, pelanggan sama, gambar dari video yang sama)
#      Baris satu grup JANGAN terpecah antara train dan valid, kalau tidak -> bocor.
grup = df["pclass"].astype(str) + "_" + df["embarked"].fillna("x")     # contoh: anggap ini kelompok
tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups=grup))
print("grup beririsan:", set(grup.iloc[tr]) & set(grup.iloc[va]))

# ---- Kasus 3: data berurutan waktu -> valid HARUS lebih baru dari train (bukan acak!)
t = pd.DataFrame({"tgl": pd.date_range("2023-01-01", periods=300), "y": np.random.default_rng(0).normal(size=300)})
batas = t.tgl.quantile(0.8)
latih, uji = t[t.tgl <= batas], t[t.tgl > batas]
print("train sampai", latih.tgl.max().date(), "| valid mulai", uji.tgl.min().date())
for i, (a, b) in enumerate(TimeSeriesSplit(n_splits=3).split(t)):      # CV berurutan
    print("fold", i, "train", len(a), "valid", len(b))

# ---- Kasus 4: split dibuat panitia (public/private leaderboard) -> tiru cara test dibuat
#      Kalau test = periode setelah train, valid = periode terakhir train. Kalau test = pelanggan baru, split per pelanggan.

# ---- Preprocessing: fit HANYA di train. Paling aman pakai Pipeline.
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
Xn = X[["age", "fare", "pclass"]]
Xtr, Xva, ytr, yva = train_test_split(Xn, y, test_size=.2, random_state=42, stratify=y)
pipa = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
pipa.fit(Xtr, ytr)                    # imputer & scaler belajar dari train saja
print("valid acc:", round(pipa.score(Xva, yva), 3))

# ---- Setelah memilih model: latih ulang pada SEMUA data lalu prediksi test
final = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xn, y)
```

## Aturan emas
1. Valid harus meniru hubungan train→test yang sebenarnya (acak / grup / waktu).
2. Semua yang "belajar dari data" (imputer, scaler, encoder, seleksi fitur, target encoding) → di dalam Pipeline.
3. Jangan menyetel model berulang-ulang pada satu valid kecil: kamu overfit ke valid. Pakai CV (resep 10).
4. `random_state` tetap → hasil bisa diulang (resep 21).
