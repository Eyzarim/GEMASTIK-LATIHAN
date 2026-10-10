# 21 · Grafik untuk laporan dan presentasi

**Kapan:** menyiapkan gambar untuk laporan/slide. **Kata kunci:** plot, grafik, matplotlib, seaborn, visualisasi, laporan, confusion matrix, importance, distribusi, savefig, warna.

```python
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")                                  # (di Jupyter tidak perlu baris ini)
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
OUT = Path("outputs/_cookbook"); OUT.mkdir(parents=True, exist_ok=True)

# 0. Gaya seragam: ukuran font terbaca di slide, palet ramah buta warna
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]       # palet Okabe-Ito (aman untuk buta warna)

df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")

# 1. Distribusi target + proporsi (SELALU judul yang menyatakan temuan, bukan 'Grafik 1')
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
df["survived"].value_counts(normalize=True).sort_index().plot.bar(ax=ax[0], color=PAL[:2]); ax[0].set(title="38% penumpang selamat", xlabel="selamat (1) / tidak (0)", ylabel="proporsi")
sns.histplot(df["age"].dropna(), bins=25, ax=ax[1], color=PAL[0]); ax[1].set(title="Mayoritas penumpang berusia 20-40", xlabel="usia")
fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=200)

# 2. Hubungan fitur-target: tingkat selamat per kelompok (dengan jumlah sampel)
t = df.groupby(["sex", "pclass"])["survived"].agg(["mean", "size"]).reset_index()
fig, ax = plt.subplots(figsize=(5.5, 3.3)); sns.barplot(data=t, x="pclass", y="mean", hue="sex", palette=PAL[:2], ax=ax); ax.set(title="Perempuan kelas 1-2 hampir semuanya selamat", ylabel="tingkat selamat", xlabel="kelas tiket")
fig.tight_layout(); fig.savefig(OUT / "kelompok.png", dpi=200)

# 3. Perbandingan model: batang dengan galat (std antar-fold) -- lebih jujur daripada satu angka
res = pd.DataFrame({"model": ["Logistic", "RandomForest", "LightGBM"], "cv": [0.80, 0.81, 0.82], "std": [0.03, 0.02, 0.025]})
fig, ax = plt.subplots(figsize=(5, 2.8)); ax.barh(res["model"], res["cv"], xerr=res["std"], color=PAL[0]); ax.set(xlim=(0.7, 0.88), title="Selisih antar-model di dalam noise antar-fold", xlabel="CV akurasi")
fig.tight_layout(); fig.savefig(OUT / "model.png", dpi=200)

# 4. Confusion matrix ternormalisasi + feature importance (fungsi siap pakai di superdb.common)
from superdb.common import plot_confusion, plot_importance
plot_confusion(np.r_[np.zeros(60), np.ones(40)].astype(int), np.r_[np.zeros(50), np.ones(10), np.zeros(8), np.ones(32)].astype(int), path=OUT / "confusion.png")
plot_importance(["usia", "tarif", "kelas", "jenis kelamin"], [0.2, 0.25, 0.15, 0.4], path=OUT / "importance.png")

# 5. Kurva belajar: apakah menambah data/kapasitas membantu?
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
X = df[["pclass", "age", "sibsp", "parch", "fare"]].fillna(0); y = df["survived"]
n, trn, val = learning_curve(RandomForestClassifier(100, min_samples_leaf=3, random_state=0, n_jobs=-1), X, y, cv=5, train_sizes=np.linspace(.2, 1, 5), scoring="accuracy")
fig, ax = plt.subplots(figsize=(5, 3.2)); ax.plot(n, trn.mean(1), "o-", label="latih", color=PAL[1]); ax.plot(n, val.mean(1), "o-", label="validasi", color=PAL[0]); ax.legend(); ax.set(title="Jarak latih-validasi besar: model overfit", xlabel="jumlah data latih", ylabel="akurasi")
fig.tight_layout(); fig.savefig(OUT / "learning_curve.png", dpi=200)
print("gambar tersimpan:", sorted(p.name for p in OUT.glob("*.png")))
```

## Aturan grafik untuk laporan lomba
- **Judul = temuan** ("Perempuan kelas 1 hampir semuanya selamat"), bukan jenis grafik.
- Satu pesan per grafik; sumbu berlabel dengan satuan; legenda tidak menutupi data.
- Simpan `dpi=200` dan lebar 5-9 inci agar tajam di Word/slide; gunakan palet buta warna.
- Sertakan **ketidakpastian** (std antar-fold, selang kepercayaan) saat membandingkan model.
- Setiap notebook di repo sudah menyimpan gambar ke `outputs/<modalitas>/<task>/<model>/*.png` dan tabel metrik di `metrics.json`: tinggal ditempel; gabungkan dengan `laporan/make_figures.py`.
