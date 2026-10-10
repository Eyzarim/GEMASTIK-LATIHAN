# 06 · Outlier & pembersihan data

**Kapan:** nilai ekstrem, typo, satuan campur, kategori berantakan. **Kata kunci:** outlier, pencilan, IQR, z-score, clip, winsorize, kategori kotor, typo, rare category.

```python
import numpy as np, pandas as pd
df = pd.read_csv("data_latihan/tabular/regression/diamonds_train.csv")

# 1. Deteksi: IQR (tahan terhadap distribusi miring)
def batas_iqr(s, k=1.5):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return q1 - k * i, q3 + k * i

lo, hi = batas_iqr(df["price"])
print("price di luar batas:", ((df.price < lo) | (df.price > hi)).sum(), "dari", len(df))

# 2. Deteksi: z-score (hanya cocok kalau kira-kira normal)
z = (df["carat"] - df["carat"].mean()) / df["carat"].std()
print("carat |z|>3:", (z.abs() > 3).sum())

# 3. Tindakan -- URUTAN PREFERENSI untuk lomba:
#    a) biarkan (model pohon kebal) / ganti ke model yang kuat
#    b) transformasi (log1p) agar ekor panjang menyusut
#    c) clip / winsorize (batasi ke persentil)
#    d) hapus -- hanya di TRAIN, jangan pernah dari test, dan hanya bila jelas salah input
df["carat_clip"] = df["carat"].clip(*df["carat"].quantile([.01, .99]))
df["price_log"] = np.log1p(df["price"])

# 4. Nilai mustahil (cek aturan bisnis)
print("x,y,z = 0:", ((df.x == 0) | (df.y == 0) | (df.z == 0)).sum())
df.loc[(df.x == 0) | (df.y == 0) | (df.z == 0), ["x", "y", "z"]] = np.nan   # ubah jadi NaN lalu imputasi

# 5. Kategori kotor: huruf besar/kecil, spasi, typo
k = pd.Series([" Jakarta", "jakarta ", "JAKARTA", "Jkt", "Bandung", "bandung"])
bersih = k.str.strip().str.lower().replace({"jkt": "jakarta"})
print(bersih.value_counts())

# 6. Kategori jarang -> gabung jadi "lain"
vc = df["clarity"].value_counts()
jarang = vc[vc < 100].index
df["clarity_g"] = df["clarity"].where(~df["clarity"].isin(jarang), "lain")

# 7. Satuan campur (mis. "5 kg", "500 g")
w = pd.Series(["5 kg", "500 g", "2.5 kg"])
angka = w.str.extract(r"([\d.]+)")[0].astype(float)
kg = np.where(w.str.contains("kg"), angka, angka / 1000)
print(kg)

# 8. Hapus duplikat, strip kolom
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.drop_duplicates()
```

## Jebakan
- Menghapus outlier di data test tidak mungkin (kamu tetap harus memprediksinya). Jadi model harus tahan terhadap outlier.
- Jangan hapus "outlier" yang justru penting (transaksi besar = target prediksi).
- Clipping memakai persentil dari train dan terapkan batas yang SAMA ke test.
