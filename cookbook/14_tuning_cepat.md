# 14 · Tuning hiperparameter cepat

**Kapan:** baseline sudah jalan, ingin menaikkan skor dalam waktu terbatas. **Kata kunci:** tuning, hyperparameter, optuna, GridSearch, RandomizedSearch, learning_rate, early stopping.

```python
import numpy as np, pandas as pd, optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")
X = pd.get_dummies(df.drop(columns=["survived", "id", "alive"]), drop_first=True).astype(float); y = df["survived"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# --- Urutan prioritas hiperparameter LightGBM (paling berpengaruh dulu)
#   learning_rate + n_estimators  -> num_leaves / max_depth -> min_child_samples -> subsample, colsample_bytree -> reg_lambda
def objective(trial):
    m = LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 64, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        random_state=0, verbose=-1, n_jobs=-1)
    return cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=15, timeout=120)        # BATASI waktu: tuning bisa menghabiskan jam
print("CV terbaik:", round(study.best_value, 4), "| param:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in study.best_params.items()})

# --- Early stopping: pilih jumlah pohon otomatis dari validation
import lightgbm as lgb
m = LGBMClassifier(n_estimators=2000, learning_rate=0.03, verbose=-1, random_state=0)
m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
print("pohon terpilih early stopping:", m.best_iteration_, "dari 2000")

# --- RandomizedSearchCV (tanpa optuna)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
rs = RandomizedSearchCV(LGBMClassifier(verbose=-1, random_state=0), {"learning_rate": loguniform(0.01, 0.2), "num_leaves": randint(4, 64)},
                        n_iter=10, cv=cv, scoring="roc_auc", random_state=0, n_jobs=-1).fit(Xtr, ytr)
print("RandomizedSearchCV terbaik:", round(rs.best_score_, 4))
```

## Aturan praktis
- Tuning memberi gain kecil dibanding **fitur yang lebih baik, validasi yang benar, dan ensemble**. Jangan mulai dari tuning.
- Selalu pakai CV, dan nilai akhir pada validation yang **tidak disentuh** tuning (selisih besar = overfit ke CV).
- Turunkan `learning_rate` (0.01-0.05) dan naikkan pohon + early stopping untuk hasil akhir; naikkan saat cadangan waktu ada.
- Batasi `timeout`/`n_trials` pada lomba; simpan `study.best_params` ke file agar bisa dipakai ulang.
- Notebook lengkap per model: `notebooks/tabular/tuning_classification/`, `tuning_regression/`.
