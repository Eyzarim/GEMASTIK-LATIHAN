# 27 · Template pipeline tabular lengkap (salin dan ubah)

**Kapan:** awal lomba tabular. Ini kerangka end-to-end paling aman: tanpa leakage, siap CV, siap submission. **Kata kunci:** pipeline, ColumnTransformer, template, baseline, end to end, tabular, submission.

```python
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer, make_column_selector as sel
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# ======================= UBAH BAGIAN INI =======================
TRAIN, TEST = "data_latihan/tabular/classification/titanic_train.csv", "data_latihan/tabular/classification/titanic_test.csv"
TARGET, ID = "survived", "id"
BUANG = ["alive"]                       # kolom bocoran target / tak berguna
SCORING = "roc_auc"                     # metrik lomba
# ================================================================

tr, te = pd.read_csv(TRAIN), pd.read_csv(TEST)
y = tr[TARGET]; X = tr.drop(columns=[TARGET, ID, *BUANG]); Xte = te.drop(columns=[ID, *BUANG])
Xte = Xte.reindex(columns=X.columns)                                     # urutan kolom test = train

num = sel(dtype_include="number"); cat = sel(dtype_exclude="number")
pre_linear = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler())]), num),
                                ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])
pre_tree = ColumnTransformer([("n", SimpleImputer(strategy="median"), num),
                              ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
kandidat = {"logreg": Pipeline([("p", pre_linear), ("m", LogisticRegression(max_iter=3000))]),
            "lightgbm": Pipeline([("p", pre_tree), ("m", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0))])}

# 1. CV semua kandidat (skema validasi dipilih SEKALI; pakai terus). Ganti ke GroupKFold/TimeSeriesSplit bila perlu.
cv = StratifiedKFold(5, shuffle=True, random_state=0)
hasil = {k: cross_val_score(p, X, y, cv=cv, scoring=SCORING) for k, p in kandidat.items()}
print(pd.DataFrame({k: {"cv_mean": v.mean(), "cv_std": v.std()} for k, v in hasil.items()}).T.round(4))

# 2. Pilih terbaik -> refit di SELURUH data -> prediksi test
terbaik = max(hasil, key=lambda k: hasil[k].mean()); final = kandidat[terbaik].fit(X, y)
proba = final.predict_proba(Xte)[:, 1]

# 3. Submission valid + simpan model
out = Path("outputs/_cookbook"); out.mkdir(parents=True, exist_ok=True)
sub = pd.DataFrame({ID: te[ID], TARGET: (proba >= 0.5).astype(int)}); sub.to_csv(out / "submission_template.csv", index=False)
assert len(sub) == len(te) and sub[ID].is_unique and sub[TARGET].notna().all()
joblib.dump(final, out / "model_template.joblib"); print("model terbaik:", terbaik, "| submission:", sub.shape)
```

## Cara memakainya di lomba
1. Ubah blok `UBAH BAGIAN INI`. Jalankan: dapat skor CV dua kandidat dalam beberapa detik.
2. Itu **baseline** dan submission pertama. Dari sini: fitur (resep 16), tuning (14), ensemble (15), validasi yang tepat (09, 12).
3. Notebook penuh dengan 17 model klasifikasi dan 15 regresi: `notebooks/tabular/classification/`, `regression/`.
