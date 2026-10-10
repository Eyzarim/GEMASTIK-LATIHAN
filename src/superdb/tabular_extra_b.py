"""Tabular lanjutan bagian B: explainability, ensembling, validasi, feature engineering, regresi lanjutan.

Helper kecil ada di sini; selebihnya kode ditampilkan langsung di notebook agar bisa disalin.
"""
import numpy as np
import pandas as pd

from .registry import register, register_skip
from .tabular import DEMO
from .tabular_extra import GRP

CLF = dict(DEMO["classification"], base_task="classification", group_col=None)
REG = dict(DEMO["regression"], base_task="regression", group_col=None)
LGBM = ("from lightgbm import LGBMClassifier",
        "model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,\n                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)"), \
       ("from lightgbm import LGBMRegressor",
        "model = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10,\n                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)")


# ------------------------------------------------------------------ helper dipakai notebook
def rank_normalize(p):
    """Ubah skor jadi rank 0..1 (untuk rata-rata rank; hanya relevan untuk metrik berbasis urutan seperti AUC)."""
    return pd.Series(np.asarray(p)).rank(method="average").to_numpy() / len(p)


def blend_weights(preds: np.ndarray, y, task="regression"):
    """Cari bobot (non-negatif, jumlah 1) yang meminimalkan RMSE (regresi) atau log-loss (klasifikasi biner) pada OOF.
    preds: (n_sampel, n_model)."""
    from scipy.optimize import minimize
    from sklearn.metrics import log_loss, mean_squared_error

    def loss(theta):
        w = np.exp(theta) / np.exp(theta).sum()
        b = preds @ w
        if task == "regression":
            return np.sqrt(mean_squared_error(y, b))
        return log_loss(y, np.clip(b, 1e-6, 1 - 1e-6))

    r = minimize(loss, np.zeros(preds.shape[1]), method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-7})
    w = np.exp(r.x) / np.exp(r.x).sum()
    return w


def interval_coverage(y, lo, hi):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def split_conformal_halfwidth(residuals, alpha=0.1):
    """Setengah lebar interval conformal split: kuantil (1-alpha) dari |residual| pada data kalibrasi (koreksi sampel hingga)."""
    r = np.sort(np.abs(np.asarray(residuals)))
    k = int(np.ceil((len(r) + 1) * (1 - alpha))) - 1
    return float(r[min(k, len(r) - 1)])


# ------------------------------------------------------------------ registrasi
_EXPLAIN = {
    "shap_tree": ("SHAP (TreeExplainer)",
                  "Penjelasan global (fitur mana yang penting dan arah pengaruhnya) dan lokal (mengapa satu baris diprediksi begitu). Bahan utama bagian analisis di laporan.",
                  "Hitungnya berat untuk data besar (pakai subset). Nilai SHAP menjelaskan model, bukan sebab-akibat dunia nyata. Fitur berkorelasi membagi kredit secara tak intuitif.", ["shap", "lightgbm"]),
    "pdp_ice": ("Partial Dependence & ICE",
                "Melihat bentuk hubungan satu/dua fitur dengan prediksi (naik/turun/non-linear/ambang). ICE menunjukkan variasi antar baris di balik rata-rata PDP.",
                "PDP menyesatkan bila fitur saling berkorelasi kuat (membuat kombinasi yang tidak realistis). Rata-rata bisa menyembunyikan efek yang saling meniadakan; lihat garis ICE.", ["lightgbm"]),
    "perm_vs_gain": ("Perbandingan metode importance",
                     "Mengecek apakah kesimpulan 'fitur penting' stabil antar metode (permutation, gain, SHAP). Kalau berbeda jauh, jangan terlalu percaya satu metode.",
                     "Gain bias ke fitur kardinalitas tinggi. Permutation importance menurun pada fitur berkorelasi (kredit terbagi). Selalu hitung pada data validation, bukan training.", ["shap", "lightgbm"]),
    "global_surrogate": ("Surrogate tree global",
                         "Menjelaskan model kotak-hitam dengan pohon keputusan dangkal berisi aturan yang bisa dibaca dan ditulis di laporan.",
                         "Hanya perkiraan. Periksa fidelity (R² terhadap prediksi model, bukan terhadap label); fidelity rendah berarti aturannya tidak bisa dipercaya.", ["lightgbm"]),
    "local_surrogate": ("Penjelasan lokal ala LIME",
                        "Menjelaskan satu prediksi dengan model linear sederhana di sekitar baris itu. Tanpa library tambahan.",
                        "Versi sederhana (bukan paket LIME asli): perturbasi Gaussian pada ruang fitur terproses. Hasil bergantung lebar kernel dan seed; jalankan beberapa seed dan lihat kestabilan.", ["lightgbm"]),
}

_ENS = {
    "oof_blend": ("Blending berbobot (OOF)",
                  "Menggabungkan beberapa model dengan bobot yang dicari dari prediksi out-of-fold. Cara ensembling paling umum di lomba.",
                  "Bobot HARUS dicari dari OOF training, bukan dari validation/test. Ensembel dari model yang mirip tidak menambah apa-apa; cek korelasi prediksi."),
    "rank_average": ("Rata-rata rank",
                     "Metrik berbasis urutan (AUC, ranking). Skor dari model berbeda skala digabung lewat rank tanpa kalibrasi.",
                     "Tidak cocok untuk metrik yang bergantung pada nilai probabilitas (log-loss) atau untuk threshold tetap."),
    "voting": ("Voting (sklearn)",
               "Cara tercepat menggabungkan model dengan API sklearn (VotingClassifier/Regressor); bobot sama.",
               "Bobot sama membuat model lemah menyeret yang kuat. Soft voting butuh predict_proba; model yang tak terkalibrasi memengaruhi hasil."),
    "stacking_oof": ("Stacking manual (meta-learner)",
                     "Model tingkat-2 belajar menggabungkan prediksi OOF model dasar. Bisa menangkap kapan model mana lebih dipercaya.",
                     "Meta-learner harus sederhana (linear) agar tidak overfit. Kebocoran paling sering terjadi di sini: prediksi dasar untuk training meta WAJIB out-of-fold."),
}

_VAL = {
    "adversarial_validation": ("Adversarial validation (train vs test)", "classification", CLF,
                               "Mengecek apakah data train dan test berasal dari distribusi yang sama. AUC jauh di atas 0.5 berarti ada pergeseran dan CV biasa akan menipu.",
                               "AUC tinggi bisa disebabkan kolom ID/tanggal/urutan baris. Telusuri fitur teratas sebelum menyimpulkan. Tidak otomatis memberi tahu cara memperbaikinya."),
    "group_vs_random_cv": ("CV acak vs CV per grup", "regression", dict(GRP, base_task="regression"),
                           "Data punya grup (pasien, pelanggan, pabrikan, lokasi) yang berulang. Menunjukkan seberapa optimis CV acak dibanding GroupKFold.",
                           "Kalau test lomba berisi grup BARU, CV yang jujur adalah per grup. CV acak menaikkan skor palsu karena grup yang sama ada di train dan valid."),
    "time_split_cv": ("CV berurutan waktu (forward chaining)", "regression", dict(GRP, base_task="regression", group_col=None, time_col="model_year"),
                      "Data punya dimensi waktu dan test ada di masa depan. Latih di masa lalu, uji di masa berikutnya.",
                      "Jangan pernah mengacak data berurutan waktu. Fitur yang tidak akan tersedia saat prediksi (data masa depan) harus dibuang."),
}

_FE = {
    "numeric_transforms": ("Transformasi numerik", "Log/akar untuk distribusi miring, rasio antar kolom, interaksi, binning. Sering memberi gain terbesar per menit kerja.",
                           "Boosting pohon tidak butuh transformasi monotonik (log) untuk fitur tunggal, tapi rasio dan interaksi tetap membantu. Selalu ukur lewat CV, jangan dipercaya karena 'masuk akal'."),
    "categorical_encoding": ("Encoding kategori", "Membandingkan one-hot, ordinal, frequency, dan target encoding aman-CV untuk model linear dan pohon.",
                             "Target encoding tanpa cross-fitting membocorkan target (skor palsu). Di sini memakai sklearn TargetEncoder yang cross-fit otomatis di dalam pipeline."),
    "groupby_aggregates": ("Fitur agregasi per grup", "Rata-rata/median/rank fitur per kategori (mis. harga rata-rata per kelas). Menangkap konteks relatif suatu baris.",
                           "Hitung agregat dari TRAINING saja lalu petakan ke test. Agregat memakai target = kebocoran kecuali di-cross-fit."),
}

_RADV = {
    "log_target": ("Transformasi target (log)", "Target positif dengan ekor panjang (harga, jumlah). Mengubah target membuat galat proporsional dan sering menurunkan RMSLE/MAPE.",
                   "Prediksi harus dikembalikan ke skala asli (expm1). Metrik RMSE pada skala asli bisa memburuk walau RMSLE membaik; pilih sesuai metrik lomba."),
    "quantile_interval": ("Regresi kuantil (interval prediksi)", "Butuh rentang (mis. 90%) di samping tebakan titik, atau metrik pinball/interval dinilai.",
                          "Kuantil yang dilatih terpisah bisa saling menyilang. Periksa coverage nyata di validation, jangan percaya nominalnya."),
    "conformal_interval": ("Interval conformal (split)", "Interval dengan jaminan coverage pada sampel hingga untuk model apa pun, hanya dari residual data kalibrasi.",
                           "Menganggap data kalibrasi dan test exchangeable (tidak ada pergeseran). Lebar interval konstan; tidak adaptif terhadap heteroskedastisitas."),
    "robust_regressors": ("Regresi robust (Huber/RANSAC/Theil-Sen)", "Target/fitur mengandung outlier ekstrem yang merusak regresi linear biasa.",
                          "Notebook ini menyuntik outlier sintetis pada training agar terlihat efeknya (ditandai jelas). Theil-Sen lambat untuk fitur banyak."),
    "gamma_tweedie": ("Loss Gamma/Tweedie", "Target positif, miring ke kanan, dengan galat yang tumbuh seiring level (harga, klaim, durasi).",
                      "Perlu target > 0 (Gamma). Bandingkan dengan transformasi log; metrik evaluasi harus selaras dengan loss."),
}


def _register_all():
    for task_name, tdef in (("classification", CLF), ("regression", REG)):
        imp, code = LGBM[0 if task_name == "classification" else 1]
        for key, (title, when, pit, req) in _EXPLAIN.items():
            register(modality="tabular", task=f"explain_{task_name}", key=key, recipe="tabular_explain",
                     title=title, imports=imp, model_code=code, kind="tree", when=when, pitfalls=pit,
                     params=dict(tdef, method=key), requires=req)
        for key, (title, when, pit) in _FE.items():
            register(modality="tabular", task=f"feature_engineering_{task_name}", key=key, recipe="tabular_fe",
                     title=title, imports="from lightgbm import " + ("LGBMClassifier" if task_name == "classification" else "LGBMRegressor"),
                     model_code=code, kind="tree", when=when, pitfalls=pit,
                     params=dict(tdef, method=key), requires=["lightgbm"])
        for key, (title, when, pit) in _ENS.items():
            if key == "rank_average" and task_name == "regression":
                register_skip("tabular", "ensembling_regression", "rank_average", "rata-rata rank tidak bermakna untuk target kontinu")
                continue
            register(modality="tabular", task=f"ensembling_{task_name}", key=key, recipe="tabular_ensemble",
                     title=title, imports="", model_code="", kind="tree", when=when, pitfalls=pit,
                     params=dict(tdef, method=key), requires=["lightgbm"])
    for key, (title, when, pit) in _RADV.items():
        register(modality="tabular", task="regression_advanced", key=key, recipe="tabular_regadv",
                 title=title, imports="", model_code="", kind="tree", when=when, pitfalls=pit,
                 params=dict(REG, method=key), requires=["lightgbm"])
    for key, (title, base, params, when, pit) in _VAL.items():
        register(modality="tabular", task="validation", key=key, recipe="tabular_validation",
                 title=title, imports="", model_code="", kind="tree", when=when, pitfalls=pit,
                 params=dict(params, method=key), requires=["lightgbm"])


_register_all()


# ------------------------------------------------------------------ transformer sklearn-compatible
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Ganti kategori dengan frekuensi relatifnya di TRAINING. Kategori baru -> 0."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = [X.iloc[:, j].astype(str).value_counts(normalize=True).to_dict() for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([X.iloc[:, j].astype(str).map(self.maps_[j]).fillna(0.0).to_numpy(float)
                                for j in range(X.shape[1])])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [f"f{j}" for j in range(len(self.maps_))])


class GroupAggregator(BaseEstimator, TransformerMixin):
    """Tambah fitur agregat per grup (mean/median) + selisih baris terhadap rata-rata grupnya.
    Tabel agregat dipelajari HANYA dari data yang di-fit (training), lalu dipetakan ke data lain, jadi aman di dalam CV."""

    def __init__(self, groups, values, aggs=("mean",), add_diff=True):
        self.groups, self.values, self.aggs, self.add_diff = groups, values, aggs, add_diff

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.tables_ = {tuple(g): X.groupby(list(g))[list(self.values)].agg(list(self.aggs)) for g in self.groups}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for g, tab in self.tables_.items():
            key = X[list(g)]
            for v in self.values:
                for a in self.aggs:
                    s = tab[(v, a)]
                    name = f"{v}_{a}_by_{'_'.join(g)}"
                    mapped = pd.MultiIndex.from_frame(key).map(s) if len(g) > 1 else key.iloc[:, 0].map(s)
                    X[name] = np.asarray(mapped, dtype=float)
                    if self.add_diff and a == "mean":
                        X[f"{v}_minus_{a}_by_{'_'.join(g)}"] = X[v].astype(float) - X[name]
        return X
