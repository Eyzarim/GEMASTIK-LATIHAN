# 08 · Scaling & transformasi (fitur dan target)

**Kapan:** model berbasis jarak/gradien (KNN, SVM, linear, NN) atau target miring. **Kata kunci:** scaling, normalisasi, standardisasi, StandardScaler, MinMax, RobustScaler, log, skew, PowerTransformer, TransformedTargetRegressor.

```python
import numpy as np, pandas as pd
from sklearn.model_selection import cross_val_score, KFold
df = pd.read_csv("data_latihan/tabular/regression/diamonds_train.csv")
X = df[["carat", "depth", "table", "x", "y", "z"]]; y = df["price"]
cv = KFold(5, shuffle=True, random_state=0)

# ---- Kapan perlu scaling?
#   WAJIB : KNN, SVM, regresi linear/logistik berpenalti, PCA, NN, K-Means
#   TIDAK  : Random Forest, Decision Tree, LightGBM, XGBoost, CatBoost (tidak peduli skala)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
#  StandardScaler : (x-mean)/std   -> default, cocok sebagian besar kasus
#  MinMaxScaler   : ke rentang [0,1] -> NN / gambar / data tanpa outlier
#  RobustScaler   : memakai median & IQR -> ada outlier
#  PowerTransformer: membuat distribusi miring jadi mendekati normal
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
for nama, sc in [("tanpa scaling", None), ("Standard", StandardScaler()), ("Robust", RobustScaler())]:
    m = KNeighborsRegressor(10) if sc is None else make_pipeline(sc, KNeighborsRegressor(10))
    print(f"KNN {nama:14s} R2 =", cross_val_score(m, X, y, cv=cv, scoring="r2").mean().round(3))

# ---- Target miring ke kanan (harga, jumlah, durasi): latih pada log lalu kembalikan
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
print("skew target:", round(y.skew(), 2), "-> setelah log1p:", round(np.log1p(y).skew(), 2))
polos = Ridge()
logis = TransformedTargetRegressor(Ridge(), func=np.log1p, inverse_func=np.expm1)   # prediksi otomatis dikembalikan
for nama, m in [("target asli", polos), ("log1p target", logis)]:
    print(f"Ridge {nama:13s} RMSE =", -cross_val_score(m, X, y, cv=cv, scoring="neg_root_mean_squared_error").mean().round(1))

# ---- SVR/NN pada target ribuan: scale target juga
from sklearn.svm import SVR
svr = TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR()), transformer=StandardScaler())
print("SVR + scale target RMSE =", -cross_val_score(svr, X.iloc[:2000], y.iloc[:2000], cv=3, scoring="neg_root_mean_squared_error").mean().round(1))

# ---- Fitur miring: log1p / PowerTransformer
Xp = PowerTransformer().fit_transform(X[["carat"]])
print("skew carat:", round(X.carat.skew(), 2), "->", round(pd.Series(Xp.ravel()).skew(), 2))
```

## Jebakan
- `fit` scaler di TRAIN saja, lalu `transform` test. Cara paling aman: taruh di `Pipeline`.
- Kalau metrik lomba RMSLE, melatih pada `log1p(target)` dengan RMSE biasa = optimasi yang tepat.
- `np.log1p` butuh nilai ≥ -1. Target negatif → pakai `PowerTransformer(method="yeo-johnson")`.
- Prediksi dari model yang dilatih pada log HARUS dikembalikan (`np.expm1`) sebelum submission (atau pakai `TransformedTargetRegressor`).
