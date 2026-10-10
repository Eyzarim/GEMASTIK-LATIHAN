# 20 · Performa: data besar, memori penuh, model lambat

**Kapan:** kernel mati, `MemoryError`, atau pelatihan terlalu lama. **Kata kunci:** memori, memory, MemoryError, lambat, dtype, downcast, chunksize, usecols, parquet, n_jobs, subsample, kernel mati.

```python
import numpy as np, pandas as pd, time

# 1. Buat data besar simulasi (1 juta baris) untuk melihat efek tiap trik
n = 1_000_000
rng = np.random.default_rng(0)
df = pd.DataFrame({"id": np.arange(n), "kota": rng.choice(["Jakarta", "Bandung", "Surabaya", "Medan"], n), "umur": rng.integers(18, 70, n),
                   "nilai": rng.normal(100, 15, n), "flag": rng.integers(0, 2, n)})
mb = lambda d: round(d.memory_usage(deep=True).sum() / 1e6, 1)
print("memori awal:", mb(df), "MB")

# 2. Turunkan dtype: int64->int8/16/32, float64->float32, string berulang->category
def hemat(d):
    d = d.copy()
    for c in d.select_dtypes("integer"): d[c] = pd.to_numeric(d[c], downcast="integer")
    for c in d.select_dtypes("float"): d[c] = d[c].astype("float32")
    for c in d.select_dtypes(["object", "string"]):
        if d[c].nunique() / len(d) < 0.5: d[c] = d[c].astype("category")
    return d
df2 = hemat(df); print("memori setelah hemat:", mb(df2), "MB (", round(100 * (1 - mb(df2) / mb(df))), "% lebih kecil )")

# 3. Baca CSV besar: hanya kolom perlu + dtype eksplisit + potongan (chunk)
df.head(100_000).to_csv("outputs/_cookbook_besar.csv", index=False)
t0 = time.time(); kecil = pd.read_csv("outputs/_cookbook_besar.csv", usecols=["kota", "nilai"], dtype={"kota": "category", "nilai": "float32"}); print("usecols+dtype:", kecil.shape, round(time.time() - t0, 2), "dtk")
total = sum(ch["nilai"].sum() for ch in pd.read_csv("outputs/_cookbook_besar.csv", usecols=["nilai"], chunksize=20_000))      # agregasi tanpa memuat semua
print("jumlah via chunk:", round(total, 1))

# 4. Format biner jauh lebih cepat dari CSV untuk dibaca ulang
df2.head(200_000).to_parquet("outputs/_cookbook.parquet"); t0 = time.time(); pd.read_parquet("outputs/_cookbook.parquet"); print("baca parquet:", round(time.time() - t0, 3), "dtk")

# 5. Hindari loop Python di pandas: vektorisasi
s = df2["nilai"].head(200_000)
t0 = time.time(); a = [x * 2 for x in s]; t1 = time.time() - t0
t0 = time.time(); b = s * 2; t2 = time.time() - t0
print(f"loop {t1:.3f} dtk vs vektor {t2:.4f} dtk")

# 6. Model lambat: (a) subsampel dulu untuk eksplorasi, (b) n_jobs=-1, (c) kurangi n_estimators + naikkan learning_rate saat eksperimen
from lightgbm import LGBMClassifier
X = df2[["umur", "nilai"]].head(300_000); y = df2["flag"].head(300_000)
for nama, kw in (("penuh 300 pohon", dict(n_estimators=300)), ("eksperimen cepat 60 pohon lr 0.2", dict(n_estimators=60, learning_rate=0.2))):
    t0 = time.time(); LGBMClassifier(verbose=-1, n_jobs=-1, **kw).fit(X, y); print(f"{nama:34s}", round(time.time() - t0, 1), "dtk")
print("subsampel 10% untuk eksplorasi:", X.sample(frac=0.1, random_state=0).shape)
```

## Aturan praktis
- Urutan hemat waktu: subsampel 10% → pilih pendekatan → latih penuh sekali saja.
- `float32` cukup untuk hampir semua fitur; `category` untuk teks berulang menghemat memori besar.
- Hapus objek besar yang tidak dipakai: `del df; import gc; gc.collect()`.
- Pohon: `n_jobs=-1`; boosting: `max_bin=63` dan `colsample_bytree` rendah mempercepat.
- GPU: untuk deep learning saja; boosting pada data kecil-menengah sering lebih cepat di CPU.
- Pantau waktu per sel (`%%time` di Jupyter) sebelum menjalankan CV 5-fold penuh.
