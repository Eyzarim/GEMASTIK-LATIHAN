"""Tabular: helper + pendaftaran semua model klasifikasi & regresi.

Alur yang dipakai semua notebook tabular:
  baca CSV -> deteksi kolom -> split -> Pipeline(preprocess + model) -> CV -> evaluasi -> importance -> submission
Imputer/scaler/encoder selalu ada DI DALAM Pipeline sehingga hanya belajar dari data training (tanpa leakage).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .registry import register


# ------------------------------------------------------------------ kolom
def detect_columns(df: pd.DataFrame, target=None, id_col=None, drop_cols=(), verbose=True) -> dict:
    """Pisahkan kolom jadi numeric / categorical / date. Kolom teks panjang dan ID-like dibuang dengan peringatan."""
    skip = {c for c in [target, id_col, *drop_cols] if c}
    num, cat, date, dropped = [], [], [], []
    for c in df.columns:
        if c in skip:
            continue
        s = df[c]
        if s.nunique(dropna=True) <= 1:
            dropped.append((c, "konstan/kosong"))
        elif pd.api.types.is_bool_dtype(s):
            cat.append(c)
        elif pd.api.types.is_datetime64_any_dtype(s):
            date.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(c)
        else:
            sample = s.dropna().astype(str).head(200)
            if len(sample) and sample.str.len().mean() > 60:
                dropped.append((c, "teks panjang (pakai notebook text)"))
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if len(sample) and parsed.notna().mean() > .9 and not sample.str.fullmatch(r"-?\d+(\.\d+)?").all():
                date.append(c)
            else:
                cat.append(c)
    if verbose:
        print(f"numeric ({len(num)}): {num}\ncategorical ({len(cat)}): {cat}\ndate ({len(date)}): {date}")
        for c, why in dropped:
            print(f"  [dibuang] {c}: {why}")
    return {"num": num, "cat": cat, "date": date, "dropped": [c for c, _ in dropped]}


def add_date_features(df: pd.DataFrame, date_cols) -> pd.DataFrame:
    """Ganti tiap kolom tanggal dengan year, month, day, dayofweek, dayofyear, is_weekend."""
    df = df.copy()
    for c in date_cols:
        d = pd.to_datetime(df[c], errors="coerce", format="mixed")
        df[f"{c}_year"] = d.dt.year
        df[f"{c}_month"] = d.dt.month
        df[f"{c}_day"] = d.dt.day
        df[f"{c}_dow"] = d.dt.dayofweek
        df[f"{c}_doy"] = d.dt.dayofyear
        df[f"{c}_weekend"] = (d.dt.dayofweek >= 5).astype(float)
        df = df.drop(columns=c)
    return df


def prepare_features(df, cols: dict, keep_extra=()):
    """Pilih kolom fitur dan ubah kolom tanggal menjadi fitur numerik. Kembalikan (X, num_cols, cat_cols)."""
    use = cols["num"] + cols["cat"] + cols["date"] + list(keep_extra)
    X = add_date_features(df[use], cols["date"])
    num = [c for c in X.columns if c not in cols["cat"]]
    return X, num, list(cols["cat"])


# ------------------------------------------------------------------ preprocessing
def make_preprocessor(num_cols, cat_cols, kind="linear", max_categories=30) -> ColumnTransformer:
    """kind: 'linear'/'nn'/'other' -> scale + one-hot; 'tree' -> tanpa scale + ordinal encoding."""
    scale = kind in ("linear", "nn")
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    if kind == "tree":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             encoded_missing_value=-2)
        cat_steps = [("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("enc", enc)]
    else:
        enc = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=max_categories,
                            sparse_output=False)
        cat_steps = [("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("enc", enc)]
    parts = []
    if num_cols:
        parts.append(("num", Pipeline(num_steps), list(num_cols)))
    if cat_cols:
        parts.append(("cat", Pipeline(cat_steps), list(cat_cols)))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(model, num_cols, cat_cols, kind="linear") -> Pipeline:
    return Pipeline([("pre", make_preprocessor(num_cols, cat_cols, kind)), ("model", model)])


# ------------------------------------------------------------------ split & CV
def split_train_valid(X, y, task, valid_size=0.2, seed=42, groups=None):
    """Stratified untuk klasifikasi, random untuk regresi. Kalau ada groups: tidak ada grup yang terpecah."""
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed).split(X, y, groups))
        return X.iloc[tr], X.iloc[va], y[tr], y[va]
    strat = y if task == "classification" else None
    return train_test_split(X, y, test_size=valid_size, random_state=seed, stratify=strat)


def cv_score(pipe, X, y, task, scoring=None, n_splits=5, seed=42, groups=None):
    """Cross-validation. Kembalikan (tabel per-fold, ringkasan mean/std)."""
    scoring = scoring or ("f1_macro" if task == "classification" else "neg_root_mean_squared_error")
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
    elif task == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, groups=groups, return_train_score=True, error_score="raise")
    df = pd.DataFrame({"fold": range(n_splits), "train": res["train_score"], "valid": res["test_score"]})
    summary = {"scoring": scoring, "valid_mean": float(df.valid.mean()), "valid_std": float(df.valid.std()),
               "train_mean": float(df.train.mean())}
    return df, summary


def feature_names(pipe) -> list:
    try:
        return list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        return []


# ------------------------------------------------------------------ pendaftaran model
_CLF_REG = {
    "linear": dict(
        title=("Logistic Regression", "Ridge Regression"), kind="linear", requires=[],
        imports=("from sklearn.linear_model import LogisticRegression", "from sklearn.linear_model import Ridge"),
        code=("model = LogisticRegression(C=1.0, max_iter=2000)", "model = Ridge(alpha=1.0)"),
        when="Baseline pertama. Cepat, stabil, koefisien bisa dijelaskan di laporan. Bagus kalau hubungan fitur-target kira-kira linear.",
        pitfalls="Wajib scaling (sudah di pipeline). Tidak menangkap interaksi/non-linearitas; kalah dari boosting di data tabular rumit."),
    "lasso_elasticnet": dict(
        title=("Logistic Regression L1/ElasticNet", "ElasticNet Regression"), kind="linear", requires=[],
        imports=("from sklearn.linear_model import LogisticRegression", "from sklearn.linear_model import ElasticNet"),
        code=("model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=5000)",
              "model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)"),
        when="Banyak fitur dan sebagian tidak berguna: L1 membuat koefisien jadi nol (seleksi fitur otomatis).",
        pitfalls="Sensitif terhadap skala (sudah di pipeline). alpha/C perlu di-tuning; saga lambat untuk data besar."),
    "knn": dict(
        title=("K-Nearest Neighbors", "K-Nearest Neighbors"), kind="linear", requires=[],
        imports=("from sklearn.neighbors import KNeighborsClassifier", "from sklearn.neighbors import KNeighborsRegressor"),
        code=("model = KNeighborsClassifier(n_neighbors=7, weights='distance')",
              "model = KNeighborsRegressor(n_neighbors=7, weights='distance')"),
        when="Data kecil-menengah, fitur sedikit dan bermakna jarak. Cocok sebagai pembanding cepat.",
        pitfalls="Lambat saat prediksi untuk data besar. Rusak kalau banyak fitur tidak relevan atau skala tidak seragam."),
    "svm": dict(
        title=("SVM (RBF)", "SVR (RBF)"), kind="linear", requires=[],
        imports=("from sklearn.svm import SVC", "from sklearn.svm import SVR"),
        code=("model = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)", "model = SVR(C=1.0, kernel='rbf')"),
        when="Data kecil-menengah (<~20 ribu baris) dengan batas keputusan non-linear.",
        pitfalls="Waktu latih naik cepat seiring jumlah baris. probability=True menambah waktu. Wajib scaling. SVR juga peka skala TARGET: target ribuan (mis. harga) membuat SVR sangat buruk; bungkus dengan TransformedTargetRegressor(StandardScaler) atau log-kan target."),
    "decision_tree": dict(
        title=("Decision Tree", "Decision Tree"), kind="tree", requires=[],
        imports=("from sklearn.tree import DecisionTreeClassifier", "from sklearn.tree import DecisionTreeRegressor"),
        code=("model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)",
              "model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42)"),
        when="Butuh model yang bisa digambar dan dijelaskan aturannya (untuk presentasi).",
        pitfalls="Overfit kalau tidak dibatasi (max_depth, min_samples_leaf). Tidak stabil terhadap perubahan data kecil."),
    "random_forest": dict(
        title=("Random Forest", "Random Forest"), kind="tree", requires=[],
        imports=("from sklearn.ensemble import RandomForestClassifier", "from sklearn.ensemble import RandomForestRegressor"),
        code=("model = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)",
              "model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)"),
        when="Baseline kuat yang hampir tanpa tuning. Tahan terhadap outlier dan skala fitur.",
        pitfalls="Tidak bisa ekstrapolasi (regresi). Biasanya kalah tipis dari boosting yang di-tuning. Model besar di disk."),
    "extra_trees": dict(
        title=("Extra Trees", "Extra Trees"), kind="tree", requires=[],
        imports=("from sklearn.ensemble import ExtraTreesClassifier", "from sklearn.ensemble import ExtraTreesRegressor"),
        code=("model = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)",
              "model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)"),
        when="Mirip Random Forest tapi lebih cepat dan kadang lebih halus di data ber-noise.",
        pitfalls="Sama seperti Random Forest: tidak bisa ekstrapolasi."),
    "gradient_boosting": dict(
        title=("Gradient Boosting (sklearn)", "Gradient Boosting (sklearn)"), kind="tree", requires=[],
        imports=("from sklearn.ensemble import GradientBoostingClassifier", "from sklearn.ensemble import GradientBoostingRegressor"),
        code=("model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)",
              "model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)"),
        when="Boosting klasik tanpa dependensi tambahan. Cadangan kalau lightgbm/xgboost tidak terpasang.",
        pitfalls="Lebih lambat dari LightGBM/XGBoost/HistGradientBoosting. Tidak menangani missing value sendiri (pipeline mengimputasi)."),
    "hist_gradient_boosting": dict(
        title=("HistGradientBoosting (sklearn)", "HistGradientBoosting (sklearn)"), kind="tree", requires=[],
        imports=("from sklearn.ensemble import HistGradientBoostingClassifier", "from sklearn.ensemble import HistGradientBoostingRegressor"),
        code=("model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True, random_state=42)",
              "model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, early_stopping=True, random_state=42)"),
        when="Boosting cepat bawaan sklearn, sangat mirip LightGBM; cocok kalau ingin tanpa library tambahan.",
        pitfalls="early_stopping memakai 10% data training sebagai validasi internal."),
    "adaboost": dict(
        title=("AdaBoost", "AdaBoost"), kind="tree", requires=[],
        imports=("from sklearn.ensemble import AdaBoostClassifier", "from sklearn.ensemble import AdaBoostRegressor"),
        code=("model = AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=42)",
              "model = AdaBoostRegressor(n_estimators=300, learning_rate=0.5, random_state=42)"),
        when="Pembanding boosting klasik; jarang jadi model terbaik.",
        pitfalls="Sensitif terhadap noise/outlier label. Biasanya kalah dari boosting modern."),
    "lightgbm": dict(
        title=("LightGBM", "LightGBM"), kind="tree", requires=["lightgbm"],
        imports=("from lightgbm import LGBMClassifier", "from lightgbm import LGBMRegressor"),
        code=("model = LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,\n                       colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)",
              "model = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,\n                      colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)"),
        when="Pilihan utama untuk data tabular lomba: cepat dan akurat, menangani missing value dan banyak baris.",
        pitfalls="Overfit di data kecil (turunkan num_leaves, naikkan min_child_samples). Tuning learning_rate vs n_estimators."),
    "xgboost": dict(
        title=("XGBoost", "XGBoost"), kind="tree", requires=["xgboost"],
        imports=("from xgboost import XGBClassifier", "from xgboost import XGBRegressor"),
        code=("model = XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,\n                      random_state=42, n_jobs=-1, eval_metric='logloss')",
              "model = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,\n                     random_state=42, n_jobs=-1)"),
        when="Boosting kuat dan teruji; kandidat kedua setelah LightGBM, bagus untuk ensemble.",
        pitfalls="Label klasifikasi harus 0..k-1 (notebook sudah memakai LabelEncoder). Lebih lambat dari LightGBM pada data besar."),
    "catboost": dict(
        title=("CatBoost", "CatBoost"), kind="tree", requires=["catboost"],
        imports=("from catboost import CatBoostClassifier", "from catboost import CatBoostRegressor"),
        code=("model = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, random_seed=42, verbose=0)",
              "model = CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, random_seed=42, verbose=0)"),
        when="Bagus di data kecil-menengah dan default-nya tahan overfit; sering kuat tanpa tuning.",
        pitfalls="Lebih lambat latihnya. Di pipeline ini kategori sudah di-ordinal-kan; untuk fitur kategori native CatBoost lihat cookbook."),
    "mlp": dict(
        title=("MLP (sklearn)", "MLP (sklearn)"), kind="nn", requires=[],
        imports=("from sklearn.neural_network import MLPClassifier", "from sklearn.neural_network import MLPRegressor"),
        code=("model = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=500, random_state=42)",
              "model = MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=500, random_state=42)"),
        when="Pembanding jaringan saraf sederhana, bagus untuk blend dengan model pohon.",
        pitfalls="Wajib scaling. Hasil bervariasi antar seed; di tabular biasanya kalah dari boosting."),
    "stacking": dict(
        title=("Stacking Ensemble", "Stacking Ensemble"), kind="tree", requires=["lightgbm"],
        imports=("from sklearn.ensemble import StackingClassifier, RandomForestClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom lightgbm import LGBMClassifier",
                 "from sklearn.ensemble import StackingRegressor, RandomForestRegressor\nfrom sklearn.linear_model import RidgeCV\nfrom lightgbm import LGBMRegressor"),
        code=("model = StackingClassifier(\n    estimators=[('rf', RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)),\n                ('lgbm', LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1))],\n    final_estimator=LogisticRegression(max_iter=1000), cv=5)",
              "model = StackingRegressor(\n    estimators=[('rf', RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)),\n                ('lgbm', LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1))],\n    final_estimator=RidgeCV(), cv=5)"),
        when="Tahap akhir: menggabungkan beberapa model berbeda untuk dorongan skor kecil tapi konsisten.",
        pitfalls="Lebih lambat (cv di dalam cv). Hanya bermanfaat kalau model dasarnya beragam. Jangan di-stack tanpa baseline yang kuat."),
}

_CLF_ONLY = {
    "naive_bayes": dict(
        title="Gaussian Naive Bayes", kind="linear", requires=[],
        imports="from sklearn.naive_bayes import GaussianNB",
        code="model = GaussianNB()",
        when="Baseline sangat cepat; berguna saat data sangat sedikit.",
        pitfalls="Mengasumsikan fitur independen; probabilitas sering tidak terkalibrasi."),
    "lda": dict(
        title="Linear Discriminant Analysis", kind="linear", requires=[],
        imports="from sklearn.discriminant_analysis import LinearDiscriminantAnalysis",
        code="model = LinearDiscriminantAnalysis()",
        when="Data kecil dengan kelas yang kira-kira Gaussian; juga untuk reduksi dimensi berlabel.",
        pitfalls="Asumsi Gaussian dan kovarian sama. Bermasalah dengan fitur kolinier."),
}

DEMO = {
    "classification": dict(
        train_path="data_latihan/tabular/classification/titanic_train.csv",
        test_path="data_latihan/tabular/classification/titanic_test.csv",
        target="survived", id_col="id", drop_cols=["alive"], scoring="f1_macro"),
    "regression": dict(
        train_path="data_latihan/tabular/regression/diamonds_train.csv",
        test_path="data_latihan/tabular/regression/diamonds_test.csv",
        target="price", id_col="id", drop_cols=[], scoring="neg_root_mean_squared_error"),
}


def _register_all():
    for key, d in _CLF_REG.items():
        for i, task in enumerate(("classification", "regression")):
            register(modality="tabular", task=task, key=key, title=d["title"][i], recipe="tabular_supervised",
                     imports=d["imports"][i], model_code=d["code"][i], kind=d["kind"], when=d["when"],
                     pitfalls=d["pitfalls"], params=DEMO[task], requires=d["requires"])
    for key, d in _CLF_ONLY.items():
        register(modality="tabular", task="classification", key=key, title=d["title"], recipe="tabular_supervised",
                 imports=d["imports"], model_code=d["code"], kind=d["kind"], when=d["when"],
                 pitfalls=d["pitfalls"], params=DEMO["classification"], requires=d["requires"])


_register_all()
