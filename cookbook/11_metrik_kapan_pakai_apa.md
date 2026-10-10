# 11 · Metrik: kapan pakai apa

**Kapan:** bingung metrik mana yang dipakai, atau skor tampak bagus tapi tidak berguna. **Kata kunci:** metrik, metric, accuracy, F1, macro, micro, AUC, PR-AUC, RMSE, MAE, MAPE, log loss, make_scorer.

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics as M

df = pd.read_csv("data_latihan/tabular_extra/imbalanced/mammography_train.csv")      # kelas positif hanya ~2.3%
X, y = df.drop(columns=["id", "label"]), df["label"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
m = RandomForestClassifier(200, random_state=0, n_jobs=-1).fit(Xtr, ytr)
pred, proba = m.predict(Xva), m.predict_proba(Xva)[:, 1]

# --- Klasifikasi: SATU metrik tidak cukup, lihat beberapa sekaligus
print("accuracy   :", round(M.accuracy_score(yva, pred), 4), "<- menipu bila kelas langka (tebak semua 0 sudah ~97.7%)")
print("precision  :", round(M.precision_score(yva, pred), 4), "(dari yang diprediksi positif, berapa benar)")
print("recall     :", round(M.recall_score(yva, pred), 4), "(dari yang sebenarnya positif, berapa tertangkap)")
print("F1         :", round(M.f1_score(yva, pred), 4), "(rata-rata harmonik precision & recall)")
print("ROC-AUC    :", round(M.roc_auc_score(yva, proba), 4), "(kualitas urutan; tidak peduli ambang)")
print("PR-AUC     :", round(M.average_precision_score(yva, proba), 4), "(lebih jujur untuk kelas langka; pembanding = proporsi positif", round(yva.mean(), 4), ")")
print("log loss   :", round(M.log_loss(yva, proba), 4), "(menghukum probabilitas yang yakin tapi salah)")
print(M.classification_report(yva, pred, digits=3))

# --- Multikelas: micro vs macro vs weighted
yt = np.array([0] * 90 + [1] * 8 + [2] * 2); yp = np.array([0] * 90 + [0] * 8 + [2] * 2)       # kelas 1 salah semua
for avg in ("micro", "macro", "weighted"):
    print(f"F1 {avg:8s}:", round(M.f1_score(yt, yp, average=avg), 3))
print("micro mirip accuracy (didominasi kelas besar), macro memperlakukan semua kelas setara, weighted dibobot jumlah.")

# --- Regresi
rng = np.random.default_rng(0); yr = rng.gamma(2, 50, 500); pr = yr * rng.normal(1, 0.2, 500)
print("RMSE :", round(float(np.sqrt(M.mean_squared_error(yr, pr))), 2), "(menghukum galat besar; sensitif outlier)")
print("MAE  :", round(M.mean_absolute_error(yr, pr), 2), "(rata-rata galat absolut; mudah dijelaskan)")
print("MAPE :", round(float(np.mean(np.abs((yr - pr) / yr))), 3), "(galat relatif; rusak bila nilai asli mendekati 0)")
print("RMSLE:", round(float(np.sqrt(M.mean_squared_log_error(yr, pr))), 3), "(galat relatif pada skala log; target positif berekor panjang)")
print("R2   :", round(M.r2_score(yr, pr), 3), "(porsi varians terjelaskan; 0 = sama dengan menebak rata-rata)")

# --- Metrik KUSTOM sesuai aturan lomba
from sklearn.metrics import make_scorer
def smape(y, p): return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9)))
smape_scorer = make_scorer(smape, greater_is_better=False)       # greater_is_better=False -> sklearn memakai negatifnya
print("sMAPE kustom:", round(smape(yr, pr), 4), "| dipakai di CV: cross_val_score(model, X, y, scoring=smape_scorer)")
```

## Memilih metrik
| Situasi | Metrik utama |
|---|---|
| Klasifikasi seimbang | accuracy, F1 macro |
| Klasifikasi tidak seimbang | PR-AUC, F1 kelas positif, recall pada precision tertentu |
| Butuh probabilitas yang baik | log loss, Brier score, kurva kalibrasi |
| Ranking / skor risiko | ROC-AUC, nDCG, MAP |
| Regresi umum | RMSE (galat besar mahal), MAE (robust) |
| Target positif berekor panjang (harga) | RMSLE, MAPE, atau latih pada log(target) |
| Forecasting | MAE/RMSE + MASE (bandingkan dengan naive musiman) |

## Jebakan
- **Optimalkan metrik yang DINILAI lomba**, bukan metrik favorit. Pakai metrik yang sama di CV, tuning, dan early stopping.
- AUC tidak bergantung ambang; F1/precision/recall bergantung ambang: atur ambang dari OOF (lihat `notebooks/tabular/imbalanced/`).
- Log-loss: jangan memotong probabilitas ke 0/1 (`np.clip(p, 1e-15, 1 - 1e-15)`).
- RMSE pada target yang di-log berbeda arti dari RMSE pada skala asli; ubah kembali sebelum menghitung metrik lomba.
