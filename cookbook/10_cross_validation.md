# 10 · Cross-validation

**Kapan:** memilih/membandingkan model dan hiperparameter; memperkirakan skor yang jujur. **Kata kunci:** CV, cross_val_score, KFold, StratifiedKFold, GroupKFold, out-of-fold, OOF, cross_val_predict, repeated.

```python
import numpy as np, pandas as pd
from sklearn.model_selection import (cross_val_score, cross_validate, StratifiedKFold, KFold,
                                      RepeatedStratifiedKFold, cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")
X = df[["age", "fare", "pclass", "sibsp", "parch"]]; y = df["survived"]
model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(200, random_state=0, n_jobs=-1))

# 1. CV standar (klasifikasi: Stratified; regresi: KFold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skor = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
print(skor.round(3), "-> rata-rata", skor.mean().round(3), "+/-", skor.std().round(3))

# 2. Banyak metrik + skor train sekaligus (cek overfit: train >> valid)
r = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro", "roc_auc"], return_train_score=True)
print({k: round(v.mean(), 3) for k, v in r.items() if k.startswith(("test_", "train_"))})

# 3. Data kecil / skor berisik: ulangi CV beberapa kali
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
print("repeated:", cross_val_score(model, X, y, cv=rcv, scoring="f1_macro").mean().round(3))

# 4. Prediksi out-of-fold (OOF): tiap baris diprediksi oleh model yang TIDAK pernah melihatnya
#    -> dasar untuk stacking, mencari ambang batas, dan analisis kesalahan
oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
from sklearn.metrics import roc_auc_score, f1_score
print("AUC OOF:", round(roc_auc_score(y, oof), 3))

# 5. Cari ambang probabilitas terbaik dari OOF (bukan dari valid kecil)
ambang = np.linspace(.2, .8, 25)
terbaik = max(ambang, key=lambda t: f1_score(y, oof > t, average="macro"))
print("ambang terbaik:", terbaik.round(2))

# 6. Ada grup / waktu: pakai GroupKFold / TimeSeriesSplit (lihat resep 09)
from sklearn.model_selection import GroupKFold
grup = (df["pclass"].astype(str) + df["embarked"].fillna("x")).factorize()[0]
print("group CV:", cross_val_score(model, X, y, cv=GroupKFold(3), groups=grup, scoring="accuracy").mean().round(3))
```

## Memilih `scoring`
Samakan dengan metrik lomba. Nama umum: `accuracy`, `f1_macro`, `f1_weighted`, `roc_auc`, `neg_log_loss`,
`neg_root_mean_squared_error`, `neg_mean_absolute_error`, `r2`. Tanda `neg_` = makin besar makin baik (nilai dikali -1).
Daftar lengkap: `from sklearn.metrics import get_scorer_names; print(get_scorer_names())`.

## Jebakan
- Skor CV >> skor leaderboard → hampir selalu **leakage** atau skema split tidak meniru test (resep 12).
- `shuffle=True` pada time series = bocor masa depan.
- Jangan tuning habis-habisan lalu melaporkan skor CV yang sama: itu optimistis. Sisakan holdout atau nested CV bila memungkinkan.
- `n_jobs=-1` di model DAN di `cross_val_score` bisa berebut CPU; pilih salah satu.
