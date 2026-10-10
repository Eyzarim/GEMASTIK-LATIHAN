# 16 · Feature engineering cepat (daftar ide per jenis kolom)

**Kapan:** model sudah jalan, skor stagnan. **Kata kunci:** fitur baru, feature engineering, rasio, agregasi, groupby, frekuensi, binning, tanggal, interaksi.

```python
import numpy as np, pandas as pd
df = pd.read_csv("data_latihan/tabular/regression/diamonds_train.csv")          # kolom: carat, cut, color, clarity, depth, table, x, y, z, price

# --- NUMERIK: rasio, selisih, transformasi, interaksi
df["volume"] = df["x"] * df["y"] * df["z"]
df["carat_per_volume"] = df["carat"] / df["volume"].replace(0, np.nan)
df["log_carat"] = np.log(df["carat"])                       # target/fitur berekor panjang -> log membantu model linear
df["carat_x_depth"] = df["carat"] * df["depth"]             # interaksi (model pohon bisa menemukannya sendiri; linear tidak)

# --- KATEGORI: urutan alami -> ordinal manual (JAUH lebih baik daripada urutan abjad)
cut_order = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
df["cut_ord"] = df["cut"].map(cut_order)
# frekuensi (berapa sering kategori muncul): informasi tanpa memakai target
df["clarity_freq"] = df["clarity"].map(df["clarity"].value_counts(normalize=True))

# --- AGREGASI PER GRUP: konteks relatif suatu baris
g = df.groupby("cut")["carat"]
df["carat_vs_cut_mean"] = df["carat"] - g.transform("mean")
df["carat_rank_in_cut"] = g.rank(pct=True)
# PENTING pada lomba: hitung agregat dari TRAIN saja lalu petakan ke test (jangan menggabung test untuk statistik target)

# --- BINNING: ubah numerik kontinu jadi kategori (membantu model linear menangkap non-linearitas)
df["carat_bin"] = pd.qcut(df["carat"], 5, labels=False, duplicates="drop")

# --- TANGGAL
d = pd.Series(pd.to_datetime(["2024-03-15 14:30", "2024-12-25 08:00", "2025-01-01 23:59"]))
tgl = pd.DataFrame({"tahun": d.dt.year, "bulan": d.dt.month, "hari_ke": d.dt.dayofyear, "hari_minggu": d.dt.dayofweek,
                    "akhir_pekan": (d.dt.dayofweek >= 5).astype(int), "jam": d.dt.hour,
                    "bulan_sin": np.sin(2 * np.pi * d.dt.month / 12), "bulan_cos": np.cos(2 * np.pi * d.dt.month / 12)})   # siklik: Desember dekat Januari
print(tgl.round(2))

# --- TEKS di dalam tabel
s = pd.Series(["Harga murah sekali!!!", "ok", "Pengiriman CEPAT dan barang bagus sekali"])
print(pd.DataFrame({"panjang": s.str.len(), "n_kata": s.str.split().str.len(), "n_tanda_seru": s.str.count("!"), "huruf_besar": s.str.count(r"[A-Z]")}))

# --- UJI: apakah fitur baru berguna? bandingkan CV dengan/tanpa
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
base = ["carat", "depth", "table", "x", "y", "z"]; new = base + ["volume", "log_carat", "cut_ord", "carat_vs_cut_mean"]
m = LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=0)
for name, cols in (("dasar", base), ("+ fitur baru", new)):
    print(f"{name:14s} CV RMSE =", round(-cross_val_score(m, df[cols], df["price"], cv=5, scoring="neg_root_mean_squared_error").mean(), 1))
```

## Jebakan
- **Gain di bawah `std` antar-fold = noise**; jangan menambah kompleksitas yang tidak terbukti.
- Fitur yang memakai target (target encoding, rata-rata target per grup) harus di-cross-fit (`sklearn.preprocessing.TargetEncoder`).
- Model pohon tidak butuh log/scaling untuk fitur tunggal, tetapi rasio, selisih, dan agregasi tetap sering membantu.
- Fitur yang tidak tersedia saat prediksi nyata (mis. data masa depan) = kebocoran.
- Eksperimen terstruktur: `notebooks/tabular/feature_engineering_classification/` dan `feature_engineering_regression/`.
