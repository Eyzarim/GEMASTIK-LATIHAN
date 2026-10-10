# 15 · Ensemble: rata-rata, blend, stacking

**Kapan:** ada beberapa model berbeda dan ingin dorongan skor kecil tapi konsisten. **Kata kunci:** ensemble, blend, blending, stacking, voting, rata-rata, rank average, OOF.

```python
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")
X = pd.get_dummies(df.drop(columns=["survived", "id", "alive"]), drop_first=True).astype(float).fillna(0); y = df["survived"].to_numpy()
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
models = {"lgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0),
          "rf": RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=0),
          "et": ExtraTreesClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=0),
          "linear": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}

# 1. Prediksi out-of-fold (OOF) pada training + prediksi validation
cv = StratifiedKFold(5, shuffle=True, random_state=0)
OOF = pd.DataFrame({k: cross_val_predict(m, Xtr, ytr, cv=cv, method="predict_proba")[:, 1] for k, m in models.items()})
VAL = pd.DataFrame({k: m.fit(Xtr, ytr).predict_proba(Xva)[:, 1] for k, m in models.items()})
print("korelasi antar model (makin rendah makin berguna):\n", OOF.corr().round(2))

# 2. Pembanding: tiap model sendiri, lalu rata-rata sederhana
res = {k: roc_auc_score(yva, VAL[k]) for k in VAL}; res["rata-rata"] = roc_auc_score(yva, VAL.mean(axis=1))
# 3. Rata-rata rank (untuk metrik berbasis urutan seperti AUC)
res["rata-rata rank"] = roc_auc_score(yva, VAL.rank().mean(axis=1))
# 4. Stacking: model sederhana yang belajar dari OOF (HARUS OOF, bukan prediksi in-sample)
meta = LogisticRegression(max_iter=1000).fit(OOF, ytr); res["stacking (logreg)"] = roc_auc_score(yva, meta.predict_proba(VAL)[:, 1])
print(pd.Series(res).sort_values(ascending=False).round(4).to_frame("AUC validation"))
print("bobot meta:", dict(zip(OOF.columns, meta.coef_[0].round(2))))
```

## Jebakan
- **Bobot/meta-model dicari dari OOF training**, bukan dari validation/test (kebocoran).
- Ensemble model yang mirip (tiga pohon) tidak menambah apa-apa: cari **keragaman** (pohon + linear + KNN/NN, atau fitur berbeda).
- Selalu bandingkan dengan model tunggal terbaik; ensemble yang lebih rumit tapi tidak lebih baik = buang.
- Simpan OOF dan prediksi test tiap model ke disk agar bisa diblend ulang tanpa melatih lagi.
- Detail dan variasi: `notebooks/tabular/ensembling_classification/` dan `ensembling_regression/`.
