# 13 · Data tidak seimbang (imbalanced)

**Kapan:** satu kelas jauh lebih langka (fraud, penyakit, churn). **Kata kunci:** imbalanced, tidak seimbang, SMOTE, class_weight, threshold, ambang, recall, PR-AUC, oversampling, undersampling.

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from lightgbm import LGBMClassifier

df = pd.read_csv("data_latihan/tabular_extra/imbalanced/mammography_train.csv")
X, y = df.drop(columns=["id", "label"]), df["label"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# Urutan langkah yang disarankan (dari paling murah):
# 1) metrik yang benar (PR-AUC / F1 positif), 2) class_weight, 3) atur ambang, 4) resampling DI DALAM CV, 5) ensemble.

# 1-2. class_weight
plain = LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=0).fit(Xtr, ytr)
bal = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced", verbose=-1, random_state=0).fit(Xtr, ytr)
for name, m in (("tanpa bobot", plain), ("class_weight=balanced", bal)):
    print(f"{name:24s} PR-AUC={average_precision_score(yva, m.predict_proba(Xva)[:, 1]):.3f}")

# 3. Atur AMBANG dari OOF pada TRAINING (bukan dari validation)
oof = cross_val_predict(bal, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
p, r, t = precision_recall_curve(ytr, oof); f1 = 2 * p * r / (p + r + 1e-12); thr = t[np.argmax(f1[:-1])]
proba = bal.predict_proba(Xva)[:, 1]
print(f"F1 ambang 0.5 = {f1_score(yva, proba >= .5):.3f} | F1 ambang OOF {thr:.2f} = {f1_score(yva, proba >= thr):.3f}")

# 4. SMOTE: HARUS memakai imblearn.pipeline.Pipeline agar resampling hanya pada data latih tiap fold
from imblearn.pipeline import Pipeline as ImbPipe
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
pipe = ImbPipe([("smote", SMOTE(random_state=0)), ("m", LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=0))])
print("CV PR-AUC dengan SMOTE di dalam pipeline:", cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="average_precision").mean().round(3))
```

## Jebakan
- **SMOTE sebelum split/CV = kebocoran**: sampel sintetis dari baris validation masuk training. Selalu di dalam `imblearn.pipeline.Pipeline`.
- Jangan menyeimbangkan data validation/test; distribusi aslinya harus tetap agar metrik jujur.
- Setelah `class_weight`/resampling probabilitas tidak terkalibrasi; bila butuh probabilitas asli, kalibrasi (`CalibratedClassifierCV`) atau koreksi prior.
- Jika lomba menilai ranking (AUC), ambang tidak penting; jika F1/recall, ambang sangat penting.
- Lebih banyak variasi (lihat `notebooks/tabular/imbalanced/`): BalancedRandomForest, EasyEnsemble, undersampling, SMOTEENN.
