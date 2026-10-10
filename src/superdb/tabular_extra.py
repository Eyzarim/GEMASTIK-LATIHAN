"""Tabular lanjutan: tuning, imbalanced, explainability, ensembling, validasi, feature engineering, regresi lanjutan.

Memakai helper dari superdb.tabular (detect_columns, prepare_features, build_pipeline, cv_score, ...).
"""
import numpy as np
import pandas as pd

from .registry import register, register_skip
from .tabular import DEMO

IMB = dict(
    train_path="data_latihan/tabular_extra/imbalanced/mammography_train.csv",
    test_path="data_latihan/tabular_extra/imbalanced/mammography_test.csv",
    target="label", id_col="id", drop_cols=[], scoring="average_precision", group_col=None)
GRP = dict(
    train_path="data_latihan/tabular_extra/grouped/mpg_train.csv",
    test_path="data_latihan/tabular_extra/grouped/mpg_test.csv",
    target="mpg", id_col="id", drop_cols=["name"], scoring="neg_root_mean_squared_error", group_col="maker")


# =============================================================== threshold & helper dipakai notebook
def best_f1_threshold(y_true, proba):
    """Threshold yang memaksimalkan F1 (pakai OOF pada TRAIN, bukan pada validation/test)."""
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f1[:-1])]), float(f1[:-1].max())


def plot_pr_curve(y_true, proba, thr=None, path=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax)
    ax.axhline(np.mean(y_true), color="grey", ls=":", label="baseline (proporsi positif)")
    ax.legend()
    ax.set_title("Precision-Recall")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


# =============================================================== 1. TUNING (Optuna)
_TUNE = {
    "lightgbm": dict(
        title="LightGBM", requires=["lightgbm", "optuna"], kind="tree",
        imports=("from lightgbm import LGBMClassifier", "from lightgbm import LGBMRegressor"),
        space=("""def make_model(trial):
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=42, n_jobs=-1, verbose=-1)""",
               """def make_model(trial):
    return LGBMRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=42, n_jobs=-1, verbose=-1)"""),
        when="Tuning pertama yang sebaiknya dicoba: LightGBM cepat, ruang pencariannya jelas. Cocok bila waktu lomba terbatas.",
        pitfalls="Tuning menaikkan skor CV tapi bisa overfit ke fold (selisih CV vs valid membesar). Jaga jumlah trial wajar, dan nilai ulang pada validation yang tidak disentuh."),
    "xgboost": dict(
        title="XGBoost", requires=["xgboost", "optuna"], kind="tree",
        imports=("from xgboost import XGBClassifier", "from xgboost import XGBRegressor"),
        space=("""def make_model(trial):
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 700),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=42, n_jobs=-1, eval_metric="logloss")""",
               """def make_model(trial):
    return XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 700),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=42, n_jobs=-1)"""),
        when="Kandidat kedua setelah LightGBM; hasil tuning-nya berguna untuk ensemble.",
        pitfalls="max_depth besar + banyak pohon memperlambat trial; batasi n_trials atau pakai timeout."),
    "catboost": dict(
        title="CatBoost", requires=["catboost", "optuna"], kind="tree",
        imports=("from catboost import CatBoostClassifier", "from catboost import CatBoostRegressor"),
        space=("""def make_model(trial):
    return CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        depth=trial.suggest_int("depth", 3, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
        random_seed=42, verbose=0)""",
               """def make_model(trial):
    return CatBoostRegressor(
        iterations=trial.suggest_int("iterations", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        depth=trial.suggest_int("depth", 3, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
        random_seed=42, verbose=0)"""),
        when="Data kecil-menengah; default CatBoost sudah kuat jadi gain dari tuning biasanya lebih kecil.",
        pitfalls="Paling lambat per trial di antara tiga boosting. Kurangi iterations di ruang pencarian bila waktu mepet."),
    "random_forest": dict(
        title="Random Forest", requires=["optuna"], kind="tree",
        imports=("from sklearn.ensemble import RandomForestClassifier", "from sklearn.ensemble import RandomForestRegressor"),
        space=("""def make_model(trial):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 25),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
        n_jobs=-1, random_state=42)""",
               """def make_model(trial):
    return RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 25),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
        n_jobs=-1, random_state=42)"""),
        when="Bila Random Forest sudah jadi model dasar dan ingin mencoba memerasnya sebelum pindah ke boosting.",
        pitfalls="Gain tuning RF biasanya kecil dibanding pindah ke boosting. max_features paling berpengaruh."),
    "linear": dict(
        title="Logistic / Ridge", requires=["optuna"], kind="linear",
        imports=("from sklearn.linear_model import LogisticRegression", "from sklearn.linear_model import Ridge"),
        space=("""def make_model(trial):
    return LogisticRegression(C=trial.suggest_float("C", 1e-3, 100, log=True), max_iter=3000)""",
               """def make_model(trial):
    return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 1000, log=True))"""),
        when="Model linear dengan regularisasi yang dipilih otomatis; cepat sekali, cocok sebagai pembanding tuning.",
        pitfalls="Hanya satu hiperparameter utama; untuk itu RidgeCV/LogisticRegressionCV sudah cukup (lihat cookbook)."),
}

_SEARCH = {
    "randomized_search_hgb": dict(
        title="RandomizedSearchCV (HistGradientBoosting)", kind="tree", requires=[],
        imports=("from sklearn.ensemble import HistGradientBoostingClassifier\nfrom scipy.stats import loguniform, randint",
                 "from sklearn.ensemble import HistGradientBoostingRegressor\nfrom scipy.stats import loguniform, randint"),
        code=("""base = HistGradientBoostingClassifier(random_state=42)
param_dist = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_iter": randint(100, 600),
    "model__max_leaf_nodes": randint(8, 64),
    "model__min_samples_leaf": randint(5, 60),
    "model__l2_regularization": loguniform(1e-3, 10),
}
SEARCH = "random"   # "random" atau "grid\"""",
              """base = HistGradientBoostingRegressor(random_state=42)
param_dist = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_iter": randint(100, 600),
    "model__max_leaf_nodes": randint(8, 64),
    "model__min_samples_leaf": randint(5, 60),
    "model__l2_regularization": loguniform(1e-3, 10),
}
SEARCH = "random\""""),
        when="Tuning tanpa Optuna, hanya sklearn. Pencarian acak biasanya jauh lebih efisien daripada grid untuk >3 hiperparameter.",
        pitfalls="Nama parameter harus diawali 'model__' karena model ada di dalam Pipeline."),
    "grid_search_rf": dict(
        title="GridSearchCV (Random Forest)", kind="tree", requires=[],
        imports=("from sklearn.ensemble import RandomForestClassifier", "from sklearn.ensemble import RandomForestRegressor"),
        code=("""base = RandomForestClassifier(n_jobs=-1, random_state=42)
param_dist = {"model__n_estimators": [200, 400], "model__max_depth": [6, 12, None],
              "model__min_samples_leaf": [1, 5], "model__max_features": ["sqrt", 0.6]}
SEARCH = "grid\"""",
              """base = RandomForestRegressor(n_jobs=-1, random_state=42)
param_dist = {"model__n_estimators": [200, 400], "model__max_depth": [6, 12, None],
              "model__min_samples_leaf": [1, 5], "model__max_features": ["sqrt", 0.6]}
SEARCH = "grid\""""),
        when="Ruang pencarian kecil dan ingin semua kombinasi dicoba (grid) agar hasilnya mudah dijelaskan di laporan.",
        pitfalls="Jumlah kombinasi berlipat ganda cepat: 2x3x2x2=24 kombinasi x 5 fold = 120 fit."),
}

# =============================================================== 2. IMBALANCED
_IMB = {
    "class_weight_lgbm": dict(
        title="LightGBM + class_weight", kind="tree", requires=["lightgbm"],
        imports="from lightgbm import LGBMClassifier",
        code="sampler = None\nmodel = LGBMClassifier(n_estimators=400, learning_rate=0.05, class_weight='balanced',\n                       random_state=42, n_jobs=-1, verbose=-1)",
        when="Langkah pertama untuk data tidak seimbang: bobot kelas langsung di model, tanpa mengubah data.",
        pitfalls="class_weight='balanced' menggeser probabilitas (tidak terkalibrasi); pilih threshold sendiri, jangan 0.5."),
    "class_weight_logreg": dict(
        title="Logistic Regression + class_weight", kind="linear", requires=[],
        imports="from sklearn.linear_model import LogisticRegression",
        code="sampler = None\nmodel = LogisticRegression(class_weight='balanced', max_iter=3000)",
        when="Baseline cepat untuk data tidak seimbang dan mudah dijelaskan.",
        pitfalls="Probabilitas bergeser; evaluasi dengan PR-AUC dan atur threshold."),
    "scale_pos_weight_xgb": dict(
        title="XGBoost + scale_pos_weight", kind="tree", requires=["xgboost"],
        imports="from xgboost import XGBClassifier",
        code="sampler = None\nspw = (ytr == 0).sum() / (ytr == 1).sum()   # rasio negatif:positif\nmodel = XGBClassifier(n_estimators=400, learning_rate=0.05, scale_pos_weight=spw,\n                      random_state=42, n_jobs=-1, eval_metric='aucpr')",
        when="Bobot positif eksplisit di XGBoost; untuk biner saja.",
        pitfalls="Hanya untuk dua kelas. Nilai penuh rasio kadang terlalu agresif; coba sqrt(rasio) sebagai alternatif."),
    "smote_logreg": dict(
        title="SMOTE + Logistic Regression", kind="linear", requires=["imblearn"],
        imports="from imblearn.over_sampling import SMOTE\nfrom sklearn.linear_model import LogisticRegression",
        code="sampler = SMOTE(random_state=42)\nmodel = LogisticRegression(max_iter=3000)",
        when="Menambah contoh sintetis kelas minoritas; sering membantu model linear.",
        pitfalls="SMOTE HARUS di dalam pipeline/CV (notebook ini sudah begitu). SMOTE di luar CV membocorkan sampel sintetis ke validation dan menaikkan skor palsu."),
    "smote_lgbm": dict(
        title="SMOTE + LightGBM", kind="tree", requires=["imblearn", "lightgbm"],
        imports="from imblearn.over_sampling import SMOTE\nfrom lightgbm import LGBMClassifier",
        code="sampler = SMOTE(random_state=42)\nmodel = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)",
        when="Dibandingkan dengan class_weight; kadang menang, kadang kalah, selalu uji.",
        pitfalls="Pada boosting, SMOTE sering tidak lebih baik daripada class_weight. Pada kategori yang di-ordinal-kan, SMOTE membuat nilai kategori pecahan tak bermakna."),
    "borderline_smote_rf": dict(
        title="BorderlineSMOTE + Random Forest", kind="tree", requires=["imblearn"],
        imports="from imblearn.over_sampling import BorderlineSMOTE\nfrom sklearn.ensemble import RandomForestClassifier",
        code="sampler = BorderlineSMOTE(random_state=42)\nmodel = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42)",
        when="Minoritas tersebar dekat batas kelas; sintesis difokuskan di perbatasan.",
        pitfalls="Mahal bila fitur banyak. Hasil sensitif terhadap k_neighbors."),
    "undersample_rf": dict(
        title="RandomUnderSampler + Random Forest", kind="tree", requires=["imblearn"],
        imports="from imblearn.under_sampling import RandomUnderSampler\nfrom sklearn.ensemble import RandomForestClassifier",
        code="sampler = RandomUnderSampler(sampling_strategy=0.2, random_state=42)   # minoritas = 20% dari mayoritas\nmodel = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42)",
        when="Data sangat besar sehingga membuang sebagian mayoritas mempercepat latihan.",
        pitfalls="Membuang informasi mayoritas. Jangan seimbangkan 1:1 penuh bila rasio asli sangat timpang."),
    "smoteenn_lgbm": dict(
        title="SMOTEENN + LightGBM", kind="tree", requires=["imblearn", "lightgbm"],
        imports="from imblearn.combine import SMOTEENN\nfrom lightgbm import LGBMClassifier",
        code="sampler = SMOTEENN(random_state=42)\nmodel = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)",
        when="Oversample lalu bersihkan sampel ambigu (ENN); alternatif kombinasi.",
        pitfalls="Lebih lambat; ukuran data setelah resampling tak bisa diprediksi."),
    "balanced_random_forest": dict(
        title="BalancedRandomForest", kind="tree", requires=["imblearn"],
        imports="from imblearn.ensemble import BalancedRandomForestClassifier",
        code="sampler = None\nmodel = BalancedRandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42)",
        when="Tiap pohon dilatih pada bootstrap yang diseimbangkan; baseline kuat untuk data timpang.",
        pitfalls="Probabilitas cenderung over-prediksi minoritas; atur threshold."),
    "easy_ensemble": dict(
        title="EasyEnsemble", kind="tree", requires=["imblearn"],
        imports="from imblearn.ensemble import EasyEnsembleClassifier",
        code="sampler = None\nmodel = EasyEnsembleClassifier(n_estimators=20, random_state=42, n_jobs=-1)",
        when="Banyak AdaBoost pada subset mayoritas berbeda-beda; sedikit informasi mayoritas terbuang.",
        pitfalls="Kurang fleksibel dibanding LightGBM; jarang menang tapi bagus untuk blend."),
}

# =============================================================== daftar entry
_FOR_TUNE = lambda task: dict(DEMO[task], group_col=None)


def _register_all():
    for key, d in _TUNE.items():
        for i, task in enumerate(("classification", "regression")):
            register(modality="tabular", task=f"tuning_{task}", key=f"{key}_optuna", recipe="tabular_tuning",
                     title=f"Optuna: {d['title']}", imports=d["imports"][i], model_code=d["space"][i], kind=d["kind"],
                     when=d["when"], pitfalls=d["pitfalls"], params=dict(_FOR_TUNE(task), base_task=task),
                     requires=d["requires"])
    for key, d in _SEARCH.items():
        for i, task in enumerate(("classification", "regression")):
            register(modality="tabular", task=f"tuning_{task}", key=key, recipe="tabular_search",
                     title=d["title"], imports=d["imports"][i], model_code=d["code"][i], kind=d["kind"],
                     when=d["when"], pitfalls=d["pitfalls"], params=dict(_FOR_TUNE(task), base_task=task),
                     requires=d["requires"])
    for key, d in _IMB.items():
        register(modality="tabular", task="imbalanced", key=key, recipe="tabular_imbalanced", title=d["title"],
                 imports=d["imports"], model_code=d["code"], kind=d["kind"], when=d["when"], pitfalls=d["pitfalls"],
                 params=dict(IMB, base_task="classification"), requires=d["requires"])


_register_all()
