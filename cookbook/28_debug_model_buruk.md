# 28 · Debug: skor rendah, overfit, atau tidak masuk akal

**Kapan:** skor jelek, atau jauh berbeda antara latih/validasi/leaderboard. **Kata kunci:** overfit, underfit, bias, variance, skor rendah, learning curve, sanity check, debug, tidak belajar.

```python
import numpy as np, pandas as pd
from sklearn.model_selection import cross_validate, learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")
X = df[["pclass", "age", "sibsp", "parch", "fare"]].fillna(df[["age", "fare"]].median()); y = df["survived"]
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# LANGKAH 1: pembanding paling bodoh. Model Anda harus jelas lebih baik dari ini.
print("tebak kelas mayoritas:", cross_validate(DummyClassifier(strategy="most_frequent"), X, y, cv=cv)["test_score"].mean().round(3))

# LANGKAH 2: model sederhana dulu. Jika LogisticRegression sudah bagus, kerumitan tak perlu.
print("logistic regression  :", cross_validate(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y, cv=cv)["test_score"].mean().round(3))

# LANGKAH 3: latih vs validasi -> diagnosis
r = cross_validate(RandomForestClassifier(300, random_state=0, n_jobs=-1), X, y, cv=cv, return_train_score=True)
tr_s, va_s = r["train_score"].mean(), r["test_score"].mean()
print(f"skor latih {tr_s:.3f} | validasi {va_s:.3f} | selisih {tr_s - va_s:.3f}")
print("DIAGNOSIS:", "OVERFIT (latih >> validasi): regularisasi, model lebih sederhana, data/fitur lebih baik" if tr_s - va_s > 0.1
      else "UNDERFIT (keduanya rendah): fitur lebih informatif, model lebih kuat" if va_s < 0.7 else "wajar")

# LANGKAH 4: kurva belajar: menambah data akan membantu?
n, a, b = learning_curve(RandomForestClassifier(100, min_samples_leaf=3, random_state=0, n_jobs=-1), X, y, cv=cv, train_sizes=[.2, .5, 1.0])
print("validasi per ukuran data:", b.mean(1).round(3), "-> masih naik berarti lebih banyak data akan membantu; datar berarti butuh fitur/model lain")

# LANGKAH 5: SANITY CHECK kebocoran: acak label -> skor harus ~ acak
ys = y.sample(frac=1, random_state=1).to_numpy()
print("label diacak:", cross_validate(RandomForestClassifier(100, random_state=0, n_jobs=-1), X, ys, cv=cv)["test_score"].mean().round(3), "(harus mendekati kelas mayoritas)")

# LANGKAH 6: periksa galat terbesar secara manual: pola kesalahan memberi ide fitur
from sklearn.model_selection import cross_val_predict
p = cross_val_predict(RandomForestClassifier(300, random_state=0, n_jobs=-1), X, y, cv=cv, method="predict_proba")[:, 1]
salah = df.assign(p=p).assign(err=lambda d: (d["survived"] - d["p"]).abs()).sort_values("err", ascending=False)
print(salah[["sex", "pclass", "age", "fare", "survived", "p"]].head(5).round(2).to_string(index=False))
```

## Peta diagnosis
| Gejala | Kemungkinan | Tindakan |
|---|---|---|
| Latih dan validasi sama-sama rendah | underfit / fitur lemah | fitur baru, model lebih kuat, kurangi regularisasi |
| Latih tinggi, validasi rendah | overfit | regularisasi, data lebih banyak, model lebih sederhana, buang fitur noise |
| CV bagus, leaderboard jelek | skema validasi tidak meniru test / bocor | resep 12 dan `notebooks/tabular/validation/` |
| Skor sama untuk semua model | fitur tidak membawa sinyal / bug label | label diacak, cek `y`, cek kolom ID |
| Skor lompat-lompat antar fold | data kecil / ada grup | `RepeatedStratifiedKFold`, `GroupKFold`, lihat std |
| Model deep tidak belajar (loss datar) | learning rate/scale/label | kecilkan lr, standarisasi, cek label 0..k-1, overfit 20 contoh dulu |
