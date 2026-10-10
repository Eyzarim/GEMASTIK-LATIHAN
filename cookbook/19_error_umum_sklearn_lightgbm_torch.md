# 19 · Error umum sklearn / LightGBM / XGBoost / PyTorch

**Kapan:** error saat melatih atau memprediksi. **Kata kunci:** shape, feature names, XGBoost label, CUDA out of memory, device, 1d array, classes, NaN, fit, predict, DataLoader, CUDA.

```python
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
def tampil(judul, fn):
    try: fn(); print("OK        ", judul)
    except Exception as ex: print(f"{type(ex).__name__:18s}", judul, "->", str(ex)[:85])

from sklearn.linear_model import LogisticRegression
X = pd.DataFrame({"a": [1., 2, 3, 4, 5, 6], "b": [6., 5, 4, 3, 2, 1]}); y = np.array([0, 0, 0, 1, 1, 1])

# 1. "The feature names should match those that were passed during fit" -> kolom test berbeda urutan/nama dari train
m = LogisticRegression().fit(X, y)
tampil("urutan kolom test beda", lambda: m.predict(X[["b", "a"]]))
tampil("urutan kolom benar", lambda: m.predict(X[["b", "a"]][m.feature_names_in_]))        # perbaikan: X_test = X_test[X_train.columns]

# 2. "X has 3 features, but ... expecting 2" -> jumlah fitur test beda (mis. get_dummies train vs test menghasilkan kolom berbeda)
tampil("jumlah fitur beda", lambda: m.predict(X.assign(c=0)))
# perbaikan: pakai OneHotEncoder(handle_unknown="ignore") di dalam Pipeline, atau X_test.reindex(columns=X_train.columns, fill_value=0)

# 3. "y should be a 1d array" / DataConversionWarning -> y berbentuk (n,1). Gunakan y.ravel() atau df["target"]
tampil("y 2D", lambda: LogisticRegression().fit(X, y.reshape(-1, 1)))

# 4. XGBoost: "Invalid classes inferred ... Expected [0 1 2], got [1 2 3]" -> label harus 0..k-1
from xgboost import XGBClassifier
tampil("label XGBoost mulai dari 1", lambda: XGBClassifier(n_estimators=2, verbosity=0).fit(X, y + 1))
from sklearn.preprocessing import LabelEncoder
tampil("label XGBoost di-encode", lambda: XGBClassifier(n_estimators=2, verbosity=0).fit(X, LabelEncoder().fit_transform(y + 1)))

# 5. LightGBM: "Do not support special JSON characters in feature name" -> nama kolom mengandung [ ] , : " dsb
from lightgbm import LGBMClassifier
kolom_buruk = X.copy(); kolom_buruk.columns = ["a[1]", "b:x"]
tampil("nama fitur LightGBM bermasalah", lambda: LGBMClassifier(n_estimators=2, verbose=-1).fit(kolom_buruk, y))
kolom_buruk.columns = kolom_buruk.columns.str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)   # perbaikan
tampil("nama fitur dibersihkan", lambda: LGBMClassifier(n_estimators=2, verbose=-1).fit(kolom_buruk, y))

# 6. "Found input variables with inconsistent numbers of samples" -> X dan y beda panjang (filter hanya pada salah satunya)
tampil("X dan y beda panjang", lambda: LogisticRegression().fit(X.iloc[:4], y))

# 7. PyTorch
import torch
print("CUDA tersedia:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU saja")
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = torch.nn.Linear(2, 1).to(dev)
# "Expected all tensors to be on the same device" -> model di GPU tetapi data di CPU (atau sebaliknya): x = x.to(dev)
tampil("tensor di CPU, model di GPU" if dev == "cuda" else "device sama (CPU)", lambda: net(torch.zeros(1, 2)))
tampil("tensor dipindah ke device", lambda: net(torch.zeros(1, 2).to(dev)))
# "mat1 and mat2 shapes cannot be multiplied" -> jumlah fitur masukan tidak cocok dengan in_features lapisan pertama
tampil("bentuk masukan salah", lambda: net(torch.zeros(1, 3).to(dev)))
# "CUDA out of memory" -> kecilkan batch_size / ukuran gambar; torch.cuda.empty_cache(); gunakan mixed precision; with torch.no_grad() saat evaluasi
# Windows: DataLoader(..., num_workers=0) bila muncul error multiprocessing / macet
# "element 0 of tensors does not require grad" -> evaluasi di dalam torch.no_grad() lalu memanggil .backward(), atau tensor di-detach
# Lupa net.eval() saat prediksi -> BatchNorm/Dropout berperilaku seperti saat latih: hasil acak dan lebih buruk
```

## Daftar periksa cepat saat model aneh
1. Bentuk: `print(X.shape, y.shape, X.dtypes.unique())`.
2. Urutan dan nama kolom train = test (`X_test = X_test[X_train.columns]`).
3. Label: `np.unique(y)`; XGBoost/PyTorch butuh 0..k-1.
4. Model dasar sederhana (LogisticRegression) berjalan? Jika ya, masalah ada di model rumit, bukan data.
5. NaN/inf: `np.isfinite(X.select_dtypes("number")).all().all()`.
