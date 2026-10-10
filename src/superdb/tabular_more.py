"""Tabular tambahan: multikelas (covertype), multilabel (yeast), survival analysis (rossi)."""
from .registry import register
from .tabular import _CLF_ONLY, _CLF_REG

T = "data_latihan/tabular_extra/"
MC = dict(base_task="classification", train_path=T + "multiclass/covertype_train.csv", test_path=T + "multiclass/covertype_test.csv", target="Cover_Type", id_col="id", drop_cols=[], scoring="f1_macro")
ML = dict(train_path=T + "multilabel/yeast_train.csv", test_path=T + "multilabel/yeast_test.csv", id_col="id", n_labels=14, name_data="Yeast (2417 gen, 103 fitur, 14 label fungsi protein)")
SV = dict(train_path=T + "survival/rossi_train.csv", test_path=T + "survival/rossi_test.csv", duration="week", event="arrest", id_col="id",
          features=["fin", "age", "race", "wexp", "mar", "paro", "prio"], name_data="Rossi (432 narapidana; durasi sampai ditangkap lagi, 74% tersensor)")

# ------------------------------------------------------------------ multilabel (kode memakai pipeline prapemrosesan di recipe)
IMPM = "import numpy as np\nfrom sklearn.multiclass import OneVsRestClassifier"
_ML = {
    "ovr_logreg": ("One-vs-Rest Logistic Regression", "linear", "from sklearn.linear_model import LogisticRegression",
                   "clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced'))",
                   "Baseline multilabel: satu classifier biner per label; mudah dijelaskan.", "Mengabaikan korelasi antar-label."),
    "ovr_lightgbm": ("One-vs-Rest LightGBM", "tree", "from lightgbm import LGBMClassifier",
                     "clf = OneVsRestClassifier(LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10, colsample_bytree=0.7, random_state=42, n_jobs=1, verbose=-1))",
                     "Hubungan non-linear per label.", "14 model terpisah; label sangat langka butuh penanganan khusus."),
    "random_forest_native": ("Random Forest (multi-output native)", "tree", "from sklearn.ensemble import RandomForestClassifier",
                             "clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42)",
                             "Satu model untuk semua label (pohon berbagi struktur); menangkap korelasi antar-label secara implisit; baseline kuat.", "Probabilitas per label kurang terkalibrasi."),
    "extra_trees_native": ("Extra Trees (multi-output native)", "tree", "from sklearn.ensemble import ExtraTreesClassifier",
                           "clf = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42)",
                           "Seperti Random Forest, lebih cepat dan sering lebih halus.", "Sama seperti Random Forest."),
    "knn_native": ("KNN (multi-output native)", "linear", "from sklearn.neighbors import KNeighborsClassifier",
                   "clf = KNeighborsClassifier(n_neighbors=15, weights='distance')",
                   "Pembanding berbasis kemiripan; memakai kombinasi label tetangga.", "Peka skala (sudah distandarkan) dan dimensi."),
    "mlp_native": ("MLP (multi-output native)", "nn", "from sklearn.neural_network import MLPClassifier",
                   "clf = MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, max_iter=500, random_state=42)",
                   "Jaringan satu model untuk semua label; menangkap korelasi antar-label.", "Hasil bervariasi antar seed."),
    "classifier_chain_lgbm": ("Classifier Chain (LightGBM)", "tree", "from sklearn.multioutput import ClassifierChain\nfrom lightgbm import LGBMClassifier",
                              "clf = ClassifierChain(LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=10, random_state=42, n_jobs=1, verbose=-1), order='random', random_state=42)",
                              "Memodelkan ketergantungan antar-label eksplisit (label sebelumnya jadi fitur).", "Galat merambat sepanjang rantai; urutan memengaruhi hasil."),
    "catboost_multilogloss": ("CatBoost MultiLogloss (native)", "tree", "from catboost import CatBoostClassifier",
                              "clf = CatBoostClassifier(loss_function='MultiLogloss', iterations=300, learning_rate=0.08, depth=5, random_seed=42, verbose=0)",
                              "Satu model boosting untuk semua label dengan loss multilabel; sering sangat kuat.", "Lebih lambat dilatih."),
}

# ------------------------------------------------------------------ survival
IMPS = "import numpy as np, pandas as pd"
_SV = {
    "kaplan_meier": ("Kaplan-Meier + log-rank (deskriptif)", "from lifelines import KaplanMeierFitter\nfrom lifelines.statistics import logrank_test", "descriptive", [],
                     "Langkah pertama pada data survival: kurva kelangsungan hidup per kelompok dan uji perbedaan antar kelompok.", "Tidak memperhitungkan banyak fitur sekaligus (univariat); tidak memprediksi individu."),
    "cox_ph": ("Cox Proportional Hazards", "from lifelines import CoxPHFitter",
               "def fit_predict(train, test, ctx):\n    m = CoxPHFitter(penalizer=0.1).fit(train[ctx.features + [ctx.duration, ctx.event]], ctx.duration, ctx.event)\n    ctx.state['model'] = m\n    return m.predict_partial_hazard(test[ctx.features]).to_numpy()", ["lifelines"],
               "Model survival standar: rasio hazard tiap fitur dapat dibaca (bahan laporan: 'usia lebih muda meningkatkan risiko x%').", "Asumsi hazard proporsional (periksa dengan uji asumsi); fitur korelasi tinggi tidak stabil (gunakan penalizer)."),
    "weibull_aft": ("Weibull AFT", "from lifelines import WeibullAFTFitter",
                    "def fit_predict(train, test, ctx):\n    m = WeibullAFTFitter(penalizer=0.05).fit(train[ctx.features + [ctx.duration, ctx.event]], ctx.duration, ctx.event)\n    ctx.state['model'] = m\n    return -m.predict_median(test[ctx.features]).to_numpy()", ["lifelines"],
                    "Model parametrik yang meramalkan waktu (bukan hanya hazard); efek fitur memperpanjang/memperpendek waktu.", "Bentuk distribusi Weibull diasumsikan; salah spesifikasi menurunkan akurasi."),
    "lognormal_aft": ("LogNormal AFT", "from lifelines import LogNormalAFTFitter",
                      "def fit_predict(train, test, ctx):\n    m = LogNormalAFTFitter(penalizer=0.05).fit(train[ctx.features + [ctx.duration, ctx.event]], ctx.duration, ctx.event)\n    ctx.state['model'] = m\n    return -m.predict_median(test[ctx.features]).to_numpy()", ["lifelines"],
                      "Hazard naik lalu turun (mis. waktu sembuh); alternatif Weibull.", "Sama seperti Weibull AFT."),
    "xgboost_cox": ("XGBoost (objective Cox)", "import xgboost as xgb",
                    "def fit_predict(train, test, ctx):\n    y = np.where(train[ctx.event] == 1, train[ctx.duration], -train[ctx.duration])          # xgboost: waktu negatif = tersensor\n    m = xgb.XGBRegressor(objective='survival:cox', n_estimators=150, learning_rate=0.05, max_depth=2, min_child_weight=5, subsample=0.8, random_state=42).fit(train[ctx.features], y)\n    ctx.state['model'] = m\n    return m.predict(test[ctx.features])", ["xgboost"],
                    "Hubungan non-linear/interaksi antar-fitur dengan boosting; tetap berbasis hazard proporsional.", "Mudah overfit pada data kecil; tidak ada interval/kurva survival langsung."),
    "xgboost_aft": ("XGBoost (AFT)", "import xgboost as xgb",
                    "def fit_predict(train, test, ctx):\n    d = xgb.DMatrix(train[ctx.features]); lo = train[ctx.duration].to_numpy(float); hi = np.where(train[ctx.event] == 1, lo, np.inf)    # tersensor kanan: batas atas tak hingga\n    d.set_float_info('label_lower_bound', lo); d.set_float_info('label_upper_bound', hi)\n    m = xgb.train({'objective': 'survival:aft', 'aft_loss_distribution': 'normal', 'aft_loss_distribution_scale': 1.2, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 5, 'subsample': 0.8, 'seed': 42}, d, num_boost_round=150)\n    ctx.state['model'] = m\n    return -m.predict(xgb.DMatrix(test[ctx.features]))", ["xgboost"],
                    "Meramalkan waktu kejadian dengan boosting; menangani sensor kanan lewat batas bawah/atas.", "Pilihan distribusi dan skala AFT memengaruhi hasil; butuh tuning."),
}


def _register_all():
    for key in ("linear", "knn", "svm", "random_forest", "extra_trees", "hist_gradient_boosting", "lightgbm", "xgboost", "catboost", "mlp"):
        d = _CLF_REG[key]
        register(modality="tabular", task="multiclass", key=key, recipe="tabular_supervised", title=d["title"][0] + " (multikelas)", imports=d["imports"][0],
                 model_code=d["code"][0], kind=d["kind"], when=d["when"] + " Di sini 7 kelas dengan distribusi tidak seimbang (covertype): perhatikan F1 macro, bukan hanya accuracy.",
                 pitfalls=d["pitfalls"], params=dict(MC), requires=d["requires"])
    d = _CLF_ONLY["naive_bayes"]
    register(modality="tabular", task="multiclass", key="naive_bayes", recipe="tabular_supervised", title=d["title"] + " (multikelas)", imports=d["imports"], model_code=d["code"],
             kind=d["kind"], when=d["when"], pitfalls=d["pitfalls"], params=dict(MC), requires=[])
    for key, (title, kind, imp, code, when, pit) in _ML.items():
        register(modality="tabular", task="multilabel", key=key, recipe="tab_multilabel", title=title, imports=IMPM + "\n" + imp, model_code=code, kind=kind,
                 when=when, pitfalls=pit, params=dict(ML), requires=(["lightgbm"] if "lightgbm" in imp else ["catboost"] if "catboost" in imp else []))
    for key, (title, imp, mode, req, when, pit) in _SV.items():
        flag = "descriptive" if mode == "descriptive" else "model"        # `mode` berisi kode fit_predict untuk model, atau 'descriptive'
        register(modality="tabular", task="survival", key=key, recipe="tab_survival", title=title, imports=IMPS + "\n" + imp,
                 model_code="" if flag == "descriptive" else mode, when=when, pitfalls=pit, params=dict(SV, mode=flag), requires=req + ["lifelines"])


_register_all()
