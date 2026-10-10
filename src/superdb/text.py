"""Teks: helper + pendaftaran model (klasik: TF-IDF/BoW + classifier).

Alur: baca CSV -> (bersihkan) -> split stratified -> Pipeline(vectorizer + model) -> CV -> evaluasi -> kata penting -> analisis error -> submission.
Vectorizer ada DI DALAM Pipeline sehingga kosakata hanya dipelajari dari data training (tanpa leakage).
"""
import re

import numpy as np
import pandas as pd

from .registry import register

# Stopword Indonesia ringkas. Kata negasi (tidak, bukan, belum, jangan, tanpa) SENGAJA tidak dimasukkan karena penting untuk sentimen.
ID_STOPWORDS = sorted(set("""
yang di dan itu dengan untuk ini dari dalam pada ke juga akan atau ada saya kami kita mereka dia anda sudah telah
sedang oleh karena sebagai adalah agar saat bisa dapat lebih sangat para hanya masih lagi sehingga serta
tersebut yaitu yakni namun tetapi tapi jika kalau bila maka lalu kemudian setelah sebelum sambil
""".split()))


def clean_text(s, lower=True, urls=True, mentions=True, digits=False, repeat=True):
    """Pembersihan ringan untuk teks media sosial: URL, @mention, angka (opsional), huruf berulang (bagusss -> baguss)."""
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    if lower:
        s = s.lower()
    if urls:
        s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    if mentions:
        s = re.sub(r"@\w+|\[username\]|\[url\]|\[sensitive-no\]", " ", s, flags=re.I)
    if digits:
        s = re.sub(r"\d+", " 0 ", s)
    if repeat:
        s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    return re.sub(r"\s+", " ", s).strip()


def top_terms(pipe, class_names, n=10, vec_step="vec", model_step="model"):
    """Kata/n-gram paling khas tiap kelas (untuk model linear / Naive Bayes). Kembalikan DataFrame."""
    vec, mdl = pipe.named_steps[vec_step], pipe.named_steps[model_step]
    names = np.asarray(vec.get_feature_names_out())
    if hasattr(mdl, "coef_"):
        W = np.atleast_2d(mdl.coef_)
    elif hasattr(mdl, "feature_log_prob_"):
        W = mdl.feature_log_prob_ - mdl.feature_log_prob_.mean(0)
    else:
        raise ValueError("model tidak punya coef_/feature_log_prob_")
    if W.shape[0] == 1 and len(class_names) == 2:
        W = np.vstack([-W[0], W[0]])
    return pd.DataFrame({str(c): names[np.argsort(-W[i])[:n]] for i, c in enumerate(class_names)})


def show_errors(texts, y_true, y_pred, class_names, n=8, width=140):
    """Contoh teks yang salah diprediksi (untuk analisis error di laporan)."""
    texts = pd.Series(list(texts)).reset_index(drop=True)
    wrong = np.where(np.asarray(y_true) != np.asarray(y_pred))[0][:n]
    return pd.DataFrame({"teks": [texts[i][:width] for i in wrong],
                         "benar": [class_names[y_true[i]] for i in wrong],
                         "prediksi": [class_names[y_pred[i]] for i in wrong]})


# ---------------------------------------------------------------- pendaftaran
WORD = 'TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode", stop_words=STOP)'
CHAR = 'TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200_000)'
IMP_V = "from sklearn.feature_extraction.text import TfidfVectorizer"

_MODELS = {
    "tfidf_logreg_word": dict(
        title="TF-IDF kata + Logistic Regression", imports=IMP_V + "\nfrom sklearn.linear_model import LogisticRegression",
        code=f"vec = {WORD}\nmodel = LogisticRegression(C=5, max_iter=2000)", requires=[],
        when="Baseline pertama untuk klasifikasi teks. Cepat, kuat, dan kata penting tiap kelas bisa dibaca (bahan laporan).",
        pitfalls="C terlalu besar overfit; coba C di 1-20. Tidak memahami urutan kata/negasi kecuali ada bigram."),
    "tfidf_logreg_char": dict(
        title="TF-IDF karakter + Logistic Regression", imports=IMP_V + "\nfrom sklearn.linear_model import LogisticRegression",
        code=f"vec = {CHAR}\nmodel = LogisticRegression(C=5, max_iter=2000)", requires=[],
        when="Teks media sosial berisi typo, singkatan, slang, imbuhan bahasa Indonesia: n-gram karakter tahan variasi penulisan.",
        pitfalls="Fitur sangat banyak dan lebih lambat. Kata penting yang dihasilkan potongan karakter, kurang mudah dibaca."),
    "tfidf_logreg_wordchar": dict(
        title="TF-IDF kata+karakter + Logistic Regression", imports=IMP_V + "\nfrom sklearn.pipeline import make_union\nfrom sklearn.linear_model import LogisticRegression",
        code=f"vec = make_union({WORD}, {CHAR})\nmodel = LogisticRegression(C=5, max_iter=2000)", requires=[],
        when="Kombinasi terkuat dari pendekatan klasik: kata menangkap makna, karakter menangkap variasi tulisan.",
        pitfalls="Paling lambat di antara varian klasik. Jangan menilai berdasar satu split; selisih kecil bisa noise."),
    "tfidf_linearsvc_word": dict(
        title="TF-IDF kata + Linear SVM", imports=IMP_V + "\nfrom sklearn.svm import LinearSVC",
        code=f"vec = {WORD}\nmodel = LinearSVC(C=0.5)", requires=[],
        when="Sering sedikit lebih akurat dari Logistic Regression pada teks; sangat cepat bahkan untuk ratusan ribu dokumen.",
        pitfalls="Tidak menghasilkan probabilitas (hanya skor keputusan). Bungkus CalibratedClassifierCV bila butuh probabilitas."),
    "tfidf_linearsvc_wordchar": dict(
        title="TF-IDF kata+karakter + Linear SVM", imports=IMP_V + "\nfrom sklearn.pipeline import make_union\nfrom sklearn.svm import LinearSVC",
        code=f"vec = make_union({WORD}, {CHAR})\nmodel = LinearSVC(C=0.3)", requires=[],
        when="Kandidat terkuat untuk klasifikasi teks klasik di lomba; coba setelah baseline.",
        pitfalls="Tidak ada probabilitas. Tuning C penting (0.1-1)."),
    "tfidf_sgd": dict(
        title="TF-IDF + SGDClassifier (modified huber)", imports=IMP_V + "\nfrom sklearn.linear_model import SGDClassifier",
        code=f"vec = {WORD}\nmodel = SGDClassifier(loss='modified_huber', alpha=2e-5, max_iter=50, random_state=42)", requires=[],
        when="Data teks sangat besar (jutaan baris) atau butuh pelatihan bertahap (partial_fit).",
        pitfalls="Sensitif terhadap alpha dan jumlah epoch; hasil bervariasi antar seed. Skala fitur sudah L2 dari TF-IDF."),
    "tfidf_complementnb": dict(
        title="TF-IDF + Complement Naive Bayes", imports=IMP_V + "\nfrom sklearn.naive_bayes import ComplementNB",
        code=f"vec = {WORD}\nmodel = ComplementNB(alpha=0.3)", requires=[],
        when="Baseline super cepat, bagus untuk kelas tidak seimbang dan data kecil.",
        pitfalls="Asumsi independensi kata; probabilitas tidak terkalibrasi. Atur alpha (0.1-1)."),
    "tfidf_multinomialnb": dict(
        title="TF-IDF + Multinomial Naive Bayes", imports=IMP_V + "\nfrom sklearn.naive_bayes import MultinomialNB",
        code=f"vec = {WORD}\nmodel = MultinomialNB(alpha=0.1)", requires=[],
        when="Baseline klasik klasifikasi teks; cepat dan stabil di data kecil.",
        pitfalls="Cenderung condong ke kelas mayoritas bila tidak seimbang (coba ComplementNB)."),
    "tfidf_ridge": dict(
        title="TF-IDF + RidgeClassifier", imports=IMP_V + "\nfrom sklearn.linear_model import RidgeClassifier",
        code=f"vec = {WORD}\nmodel = RidgeClassifier(alpha=1.0)", requires=[],
        when="Alternatif linear yang cepat dan stabil di data teks berdimensi tinggi.",
        pitfalls="Tanpa probabilitas. alpha perlu disetel."),
    "tfidf_passive_aggressive": dict(
        title="TF-IDF + Passive-Aggressive", imports=IMP_V + "\nfrom sklearn.linear_model import PassiveAggressiveClassifier",
        code=f"vec = {WORD}\nmodel = PassiveAggressiveClassifier(C=0.5, max_iter=100, random_state=42)", requires=[],
        when="Pembelajaran online untuk aliran teks besar; kuat untuk klasifikasi berita.",
        pitfalls="Sensitif terhadap noise label. Hasil bergantung urutan data (set random_state)."),
    "tfidf_knn": dict(
        title="TF-IDF + KNN (cosine)", imports=IMP_V + "\nfrom sklearn.neighbors import KNeighborsClassifier",
        code=f"vec = {WORD}\nmodel = KNeighborsClassifier(n_neighbors=15, metric='cosine', weights='distance')", requires=[],
        when="Pembanding berbasis kemiripan dokumen; berguna bila dokumen mirip-mirip (duplikat/templat).",
        pitfalls="Lambat saat prediksi untuk data besar; biasanya kalah dari model linear di teks."),
    "tfidf_lightgbm": dict(
        title="TF-IDF + LightGBM", imports=IMP_V + "\nfrom lightgbm import LGBMClassifier",
        code='vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True, dtype=np.float32)\nmodel = LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, colsample_bytree=0.3, random_state=42, n_jobs=-1, verbose=-1)',
        requires=["lightgbm"],
        when="Dicampur dengan fitur numerik lain, atau untuk ensemble dengan model linear (kesalahannya berbeda).",
        pitfalls="Biasanya kalah dari linear di teks murni dan lebih lambat. Batasi max_features."),
    "tfidf_svd_rf": dict(
        title="TF-IDF + SVD (LSA) + Random Forest", imports=IMP_V + "\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.decomposition import TruncatedSVD\nfrom sklearn.ensemble import RandomForestClassifier",
        code=f"vec = make_pipeline({WORD}, TruncatedSVD(n_components=150, random_state=42))\nmodel = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)",
        requires=[],
        when="Reduksi dimensi (LSA) lalu model non-linear; fitur padat 150 dimensi mudah digabung dengan fitur lain.",
        pitfalls="LSA membuang detail kata langka; biasanya kalah dari linear di sparse. get_feature_names tidak bermakna (tanpa kata penting)."),
    "tfidf_svd_svc": dict(
        title="TF-IDF + SVD (LSA) + SVM RBF", imports=IMP_V + "\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.decomposition import TruncatedSVD\nfrom sklearn.svm import SVC",
        code=f"vec = make_pipeline({WORD}, TruncatedSVD(n_components=150, random_state=42))\nmodel = SVC(C=3, kernel='rbf', probability=True, random_state=42)",
        requires=[],
        when="Batas keputusan non-linear di ruang LSA; alternatif ketika data tidak terlalu besar.",
        pitfalls="Lambat di data besar (probability=True menambah CV internal). Tanpa kata penting."),
    "bow_logreg": dict(
        title="Bag-of-Words (hitungan) + Logistic Regression", imports="from sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.linear_model import LogisticRegression",
        code='vec = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True, stop_words=STOP)\nmodel = LogisticRegression(C=0.5, max_iter=2000)', requires=[],
        when="Pembanding paling sederhana (kata ada/tidak). Kadang cukup kuat untuk teks pendek.",
        pitfalls="Tanpa bobot IDF kata umum mendominasi; kurangi C untuk regularisasi."),
    "hashing_sgd": dict(
        title="HashingVectorizer + SGD (skala besar)", imports="from sklearn.feature_extraction.text import HashingVectorizer\nfrom sklearn.linear_model import SGDClassifier",
        code='vec = HashingVectorizer(n_features=2**18, ngram_range=(1, 2), alternate_sign=False, norm="l2")\nmodel = SGDClassifier(loss="hinge", alpha=2e-5, max_iter=50, random_state=42)', requires=[],
        when="Memori terbatas atau teks sangat besar: tanpa kosakata disimpan, bisa diproses streaming.",
        pitfalls="Tabrakan hash kecil mengaburkan fitur; tidak bisa menampilkan kata penting (tidak ada pemetaan balik)."),
}

TASKS = {
    "sentiment": dict(train_path="data_latihan/text/sentiment/smsa_train.csv", test_path="data_latihan/text/sentiment/smsa_test.csv",
                      text_col="text", target="label", id_col="id", lang="id", scoring="f1_macro"),
    "classification": dict(train_path="data_latihan/text/classification/emot_train.csv", test_path="data_latihan/text/classification/emot_test.csv",
                           text_col="text", target="label", id_col="id", lang="id", scoring="f1_macro"),
    "classification_en": dict(train_path="data_latihan/text/classification_en/news4_train.csv", test_path="data_latihan/text/classification_en/news4_test.csv",
                              text_col="text", target="label", id_col="id", lang="en", scoring="f1_macro"),
}


def _register_all():
    for task, params in TASKS.items():
        for key, d in _MODELS.items():
            register(modality="text", task=task, key=key, recipe="text_supervised", title=d["title"],
                     imports=d["imports"], model_code=d["code"], kind="linear", when=d["when"], pitfalls=d["pitfalls"],
                     params=dict(params), requires=d["requires"])


_register_all()
