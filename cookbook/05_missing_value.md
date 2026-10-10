# 05 · Missing value / NaN / data kosong

**Kapan:** ada sel kosong. **Kata kunci:** missing, NaN, null, isna, fillna, imputasi, SimpleImputer, KNNImputer, add_indicator.

```python
import numpy as np, pandas as pd
df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")

# 1. Seberapa parah?
print((df.isna().mean() * 100).round(1).sort_values(ascending=False).head(5))

# 2. Aturan praktis
#   > 60-70% kosong : buang kolom ATAU ubah jadi fitur "ada/tidak ada"
#   sedikit kosong  : isi (imputasi)
#   pohon (LightGBM/XGBoost/CatBoost/HistGB): boleh dibiarkan NaN, mereka menanganinya sendiri
#   linear/SVM/KNN/NN : WAJIB diisi

# 3. Fitur "apakah kosong" -- sering informatif (mis. deck kosong ~ penumpang kelas bawah)
df["deck_kosong"] = df["deck"].isna().astype(int)

# 4. Isi manual
df["age_isi"] = df["age"].fillna(df["age"].median())
df["embarked_isi"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["age_per_kelas"] = df["age"].fillna(df.groupby("pclass")["age"].transform("median"))   # lebih cerdas

# 5. Cara BENAR di dalam pipeline (belajar median dari train saja -> tanpa leakage)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X = df[["age", "fare", "pclass", "sibsp"]]; y = df["survived"]
pipa = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), LogisticRegression(max_iter=1000))
print("median + indikator:", cross_val_score(pipa, X, y, cv=5).mean().round(3))

# 6. Imputasi berbasis tetangga (lebih lambat, kadang lebih baik)
pipa2 = make_pipeline(KNNImputer(n_neighbors=5), LogisticRegression(max_iter=1000))
print("KNN imputer       :", cross_val_score(pipa2, X, y, cv=5).mean().round(3))

# 7. Imputasi iteratif (model per kolom)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
pipa3 = make_pipeline(IterativeImputer(max_iter=10, random_state=0), LogisticRegression(max_iter=1000))
print("Iterative imputer :", cross_val_score(pipa3, X, y, cv=5).mean().round(3))

# 8. Pohon: biarkan NaN
from lightgbm import LGBMClassifier
print("LightGBM tanpa imputasi:", cross_val_score(LGBMClassifier(verbose=-1, n_estimators=100), X, y, cv=5).mean().round(3))

# 9. Time series: isi dengan nilai sebelumnya / interpolasi
s = pd.Series([1, np.nan, np.nan, 4, 5.0], index=pd.date_range("2024-01-01", periods=5))
print(s.ffill().tolist(), s.interpolate().tolist())
```

## Jebakan
- **Jangan** `fillna(df.mean())` di seluruh data sebelum split: statistik test ikut masuk (leakage ringan). Taruh imputer di `Pipeline` (resep 09).
- Kosong yang berarti sesuatu ("0 pembelian", "tidak punya") isi dengan 0/'tidak ada', bukan median.
- Isi kategori kosong dengan label baru `"missing"` sering lebih baik daripada modus.
- Kolom yang SELURUH barisnya kosong membuat `SimpleImputer` membuang kolom itu (kolom hilang di output); pakai `keep_empty_features=True` bila perlu.
