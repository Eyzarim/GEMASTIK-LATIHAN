# 04 · EDA cepat (10 menit pertama)

**Kapan:** setiap dataset baru, sebelum memodelkan. **Kata kunci:** EDA, eksplorasi, distribusi, korelasi, missing, outlier, target, ringkasan.

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")
TARGET = "survived"

# 1. Ukuran, tipe, kosong, unik
ringkas = pd.DataFrame({
    "tipe": df.dtypes.astype(str),
    "kosong_%": (df.isna().mean() * 100).round(1),
    "unik": df.nunique(),
    "contoh": df.iloc[0],
})
print(df.shape); print(ringkas)

# 2. Statistik numerik & kategorikal
print(df.describe().T.round(2))
print(df.describe(include="object").T if (df.dtypes == "object").any() else "")

# 3. Target: seimbang? (klasifikasi) / miring? (regresi)
print(df[TARGET].value_counts(normalize=True).round(3))

# 4. Distribusi semua kolom numerik
num = df.select_dtypes("number").columns.drop([TARGET, "id"], errors="ignore")
df[num].hist(bins=30, figsize=(10, 6)); plt.tight_layout(); plt.savefig("outputs/eda_hist.png", dpi=100); plt.close()

# 5. Hubungan fitur dengan target
print(df.groupby("sex")[TARGET].mean().round(3))
print(df[num].corrwith(df[TARGET]).sort_values().round(3))      # korelasi dengan target

# 6. Korelasi antar fitur (cari fitur kembar -> multikolinier)
plt.figure(figsize=(6, 5)); sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.tight_layout(); plt.savefig("outputs/eda_corr.png", dpi=100); plt.close()

# 7. Missing: pola
print(df.isna().sum()[lambda s: s > 0].sort_values(ascending=False))

# 8. Duplikat & ID
print("baris duplikat:", df.duplicated().sum(), "| id unik:", df["id"].is_unique)

# 9. Kolom yang mencurigakan (bocoran target): kolom yang hampir identik dengan target
for c in df.columns.drop(TARGET):
    t = pd.crosstab(df[c].astype(str), df[TARGET])
    if len(t) <= 20 and (t.max(axis=1).sum() / t.values.sum()) > 0.99:
        print("CURIGA bocoran:", c)
```

## Daftar periksa EDA (jawab sebelum memodelkan)
1. Apa metrik penilaian? → tentukan skema CV (resep 10, 11).
2. Target seimbang? Regresi: miring ke kanan → pertimbangkan `np.log1p` (resep 08).
3. Ada kolom ID / tanggal / teks panjang? Ada kolom yang tidak ada di test?
4. Train dan test mirip distribusinya? (bandingkan `describe()` keduanya)
5. Kolom bocoran? (resep 12)
6. Berapa persen kosong per kolom? (resep 05)

## Jebakan
- `df.describe()` bawaan hanya numerik; kategorikal butuh `include="object"` (atau `"str"` di pandas 3 bila kolom bertipe str).
- Korelasi Pearson hanya menangkap hubungan linear; fitur penting bisa berkorelasi ~0.
