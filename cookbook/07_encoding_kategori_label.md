# 07 · Encoding kategori & label string ke angka

**Kapan:** kolom teks pendek (kategori) atau target berupa string. **Kata kunci:** encoding, one-hot, ordinal, label encoder, target encoding, frequency encoding, get_dummies, kategori baru di test, cardinality.

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")
X = df[["sex", "embarked", "class", "who", "deck", "pclass", "fare"]]; y = df["survived"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=.2, random_state=0, stratify=y)

# ---- Target (label) string -> angka & balik lagi
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label = pd.Series(["rendah", "tinggi", "sedang", "rendah"])
kode = le.fit_transform(label)                  # urutan alfabet
print(kode, dict(enumerate(le.classes_)))
print(le.inverse_transform(kode))               # kembalikan ke teks untuk submission

# ---- Fitur kategori. PILIH berdasar jenis model:
#   linear / SVM / KNN / NN : One-Hot
#   pohon (RF/LightGBM/XGB) : Ordinal (angka kode) cukup; CatBoost bisa terima kategori langsung
#   kardinalitas tinggi (ratusan nilai): frequency / target encoding

# One-Hot di dalam ColumnTransformer (cara aman: kategori baru di test diabaikan, bukan error)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
kat = ["sex", "embarked", "class", "who", "deck"]
onehot = ColumnTransformer([("oh", make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore", sparse_output=False)), kat)], remainder="passthrough")
print("one-hot ->", onehot.fit_transform(Xtr.fillna({"fare": 0})).shape)

# Ordinal (untuk pohon); kategori tak dikenal -> -1
ordi = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
print(ordi.fit_transform(Xtr[kat].astype(object).fillna("kosong"))[:2])

# Ordinal BERURUTAN (ada tingkatan): beri urutan eksplisit
ukuran = pd.Series(["S", "L", "M", "XL"])
print(OrdinalEncoder(categories=[["S", "M", "L", "XL"]]).fit_transform(ukuran.to_frame()).ravel())

# ---- Frequency encoding (aman, tanpa target)
frek = Xtr["deck"].value_counts(normalize=True)
print(Xva["deck"].map(frek).fillna(0).head(3).tolist())

# ---- Target encoding (rata-rata target per kategori). WAJIB di dalam CV / pakai sklearn TargetEncoder
from sklearn.preprocessing import TargetEncoder
te = TargetEncoder(cv=5, random_state=0)        # menghitung dengan cross-fitting agar tidak bocor
print(te.fit_transform(Xtr[["deck"]].astype(object).fillna("kosong"), ytr)[:3].ravel().round(3))

# ---- get_dummies cepat untuk eksplorasi (JANGAN untuk pipeline train/test terpisah)
print(pd.get_dummies(df["sex"], prefix="sex").head(2))

# ---- tipe category (hemat memori, dipakai LightGBM native)
df["sex"] = df["sex"].astype("category")
print(df["sex"].cat.codes.head(3).tolist())
```

## Jebakan
- `pd.get_dummies` pada train dan test **terpisah** menghasilkan kolom berbeda. Pakai `OneHotEncoder` yang di-fit di train.
- `LabelEncoder` untuk FITUR kategori pada model linear menyiratkan urutan palsu (Jakarta<Bandung<Surabaya). Dia hanya untuk TARGET.
- Target encoding tanpa cross-fitting = bocoran target ke fitur → skor CV palsu tinggi.
- XGBoost: label klasifikasi harus 0..k-1 (pakai LabelEncoder dulu).
