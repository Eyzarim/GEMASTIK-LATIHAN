"""Pendaftaran notebook recsys: prediksi rating dan top-N (implicit)."""
from .registry import register, register_skip

ML = dict(folder="data_latihan/recsys/ml-100k", name_data="MovieLens 100k (943 pengguna, 1682 film, 100.000 rating)")
IMPR = ("import numpy as np, pandas as pd\nfrom superdb.recsys import BiasModel, surprise_fit_predict, torch_mf")
_RATE = {
    "global_mean": ("Rata-rata global", "", "def fit_predict(train, test, ctx):\n    return np.full(len(test), train['rating'].mean())", [],
                    "Batas bawah wajib: semua prediksi sama dengan rata-rata rating.", "Mengabaikan pengguna dan film."),
    "item_mean": ("Rata-rata per film", "", "def fit_predict(train, test, ctx):\n    m = train.groupby('i')['rating'].mean(); return test['i'].map(m).fillna(train['rating'].mean()).to_numpy()", [],
                  "Baseline berbasis popularitas/kualitas film.", "Film dengan sedikit rating tidak stabil (tanpa penyusutan); mengabaikan selera pengguna."),
    "bias_model": ("Model bias (mu + bu + bi, teregularisasi)", "",
                   "def fit_predict(train, test, ctx):\n    return BiasModel(reg_u=10, reg_i=25).fit(train, ctx.n_users, ctx.n_items).predict(test)", [],
                   "Baseline kuat yang sering sulit dikalahkan: pengguna ramah/pelit dan film bagus/jelek.", "Tidak menangkap selera personal (interaksi pengguna-film)."),
    "svd_surprise": ("SVD / FunkSVD (surprise)", "from surprise import SVD",
                     "def fit_predict(train, test, ctx):\n    return surprise_fit_predict(SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.05, random_state=42), train, test)", ["surprise"],
                     "Faktorisasi matriks klasik untuk rating eksplisit; titik awal standar.", "Hiperparameter (faktor, regularisasi, epoch) menentukan; overfit tanpa regularisasi."),
    "svdpp_surprise": ("SVD++ (surprise)", "from surprise import SVDpp",
                       "def fit_predict(train, test, ctx):\n    return surprise_fit_predict(SVDpp(n_factors=30, n_epochs=15, random_state=42), train, test)", ["surprise"],
                       "Menambahkan sinyal implisit (film yang dinilai) ke faktorisasi; sering sedikit lebih akurat dari SVD.", "Lebih lambat."),
    "nmf_surprise": ("NMF (surprise)", "from surprise import NMF",
                     "def fit_predict(train, test, ctx):\n    return surprise_fit_predict(NMF(n_factors=30, n_epochs=40, random_state=42), train, test)", ["surprise"],
                     "Faktor non-negatif yang lebih mudah ditafsirkan (preferensi 'bagian').", "Biasanya sedikit kalah dari SVD pada RMSE."),
    "knn_item_surprise": ("KNN item (surprise, baseline-adjusted)", "from surprise import KNNWithMeans",
                          "def fit_predict(train, test, ctx):\n    return surprise_fit_predict(KNNWithMeans(k=40, sim_options={'name': 'pearson_baseline', 'user_based': False}, verbose=False), train, test)", ["surprise"],
                          "Penjelasan intuitif ('karena film mirip yang Anda sukai'); stabil pada banyak pengguna.", "Matriks kemiripan film besar; kurang baik pada data sangat jarang."),
    "slope_one_surprise": ("Slope One (surprise)", "from surprise import SlopeOne",
                           "def fit_predict(train, test, ctx):\n    return surprise_fit_predict(SlopeOne(), train, test)", ["surprise"],
                           "Sederhana, cepat, tanpa hiperparameter.", "Akurasi umumnya di bawah SVD."),
    "co_clustering_surprise": ("Co-clustering (surprise)", "from surprise import CoClustering",
                               "def fit_predict(train, test, ctx):\n    return surprise_fit_predict(CoClustering(n_cltr_u=5, n_cltr_i=5, n_epochs=20, random_state=42), train, test)", ["surprise"],
                               "Pengelompokan bersama pengguna dan film.", "Kualitas sedang; sensitif jumlah cluster."),
    "lightgbm_hybrid": ("LightGBM hibrida (bias + demografi + genre)", "from lightgbm import LGBMRegressor\nfrom sklearn.model_selection import KFold",
                        "def fit_predict(train, test, ctx):\n    mv, us = ctx.movies, ctx.users\n    G = [c for c in mv.columns if c in ctx.genres]\n    def feats(df, bias):\n        X = pd.DataFrame({'bias': bias, 'age': us.set_index('u').loc[df.u, 'age'].to_numpy(), 'gender': (us.set_index('u').loc[df.u, 'gender'] == 'M').astype(int).to_numpy(),\n                          'occupation': us.set_index('u').loc[df.u, 'occupation'].astype('category').cat.codes.to_numpy(), 'year': pd.to_datetime(mv.set_index('i').loc[df.i, 'release_date'], errors='coerce').dt.year.fillna(1990).to_numpy()})\n        for g in G: X[g] = mv.set_index('i').loc[df.i, g].to_numpy()\n        return X\n    oof = np.zeros(len(train))\n    for a, b in KFold(5, shuffle=True, random_state=0).split(train):\n        oof[b] = BiasModel().fit(train.iloc[a], ctx.n_users, ctx.n_items).predict(train.iloc[b])   # bias OOF agar tidak bocor\n    m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=0, verbose=-1).fit(feats(train, oof), train['rating'])\n    return m.predict(feats(test, BiasModel().fit(train, ctx.n_users, ctx.n_items).predict(test)))", ["lightgbm"],
                        "Ada fitur sisi (demografi, genre, tahun) dan ingin memanfaatkannya; mengatasi sebagian cold-start.", "Bias harus dihitung out-of-fold untuk baris latih (sudah ditangani) atau terjadi kebocoran."),
    "torch_mf": ("Faktorisasi matriks + bias (PyTorch, GPU)", "",
                 "def fit_predict(train, test, ctx):\n    return torch_mf(train, test, ctx.n_users, ctx.n_items, dim=32, epochs=40, lr=0.01, wd=1e-4)", ["torch"],
                 "Dasar untuk model rekomendasi neural (two-tower, NCF); kontrol penuh atas loss dan fitur tambahan.", "Wajib regularisasi (weight decay) atau overfit cepat; pantau RMSE validation."),
}
IMPT = "import numpy as np, scipy.sparse as sp\nfrom superdb.recsys import item_knn_scores, user_knn_scores"
_TOPN = {
    "random": ("Acak (batas bawah)", "", "def score_matrix(R, ctx):\n    return np.random.default_rng(0).random(R.shape)", [],
               "Garis dasar: metrik yang tidak jauh di atas acak berarti model tidak belajar apa pun.", "-"),
    "popularity": ("Popularitas", "", "def score_matrix(R, ctx):\n    return np.tile(np.asarray(R.sum(0)).ravel(), (R.shape[0], 1))", [],
                   "Baseline top-N wajib dan sering sulit dikalahkan: rekomendasikan film paling populer yang belum ditonton.", "Tidak personal; cakupan katalog sangat rendah (itu-itu saja)."),
    "item_knn": ("Item-KNN (cosine)", "", "def score_matrix(R, ctx):\n    return item_knn_scores(R, k=50, shrink=10)", [],
                 "Baseline kolaboratif kuat dan mudah dijelaskan: 'orang yang menonton ini juga menonton itu'.", "Matriks kemiripan item berukuran n_item²; shrink mencegah item langka mendominasi."),
    "user_knn": ("User-KNN (cosine)", "", "def score_matrix(R, ctx):\n    return user_knn_scores(R, k=50, shrink=10)", [],
                 "Pengguna dengan selera mirip menonton film mirip; baik bila pengguna lebih sedikit daripada item.", "Mahal bila pengguna sangat banyak."),
    "svd_latent": ("SVD terpotong (faktor laten)", "from sklearn.decomposition import TruncatedSVD",
                   "def score_matrix(R, ctx):\n    svd = TruncatedSVD(n_components=50, random_state=0).fit(R)\n    return svd.transform(R) @ svd.components_", [],
                   "Faktor laten murah tanpa pustaka tambahan; baik untuk data implisit besar.", "Merekonstruksi 0 sebagai sinyal; item populer cenderung mendominasi."),
    "als_implicit": ("ALS (implicit)", "from implicit.als import AlternatingLeastSquares",
                     "def score_matrix(R, ctx):\n    m = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, random_state=0, use_gpu=False)\n    m.fit((R * 40).astype('float32'), show_progress=False)       # fit() melatih di tempat (tidak mengembalikan model)\n    return np.asarray(m.user_factors) @ np.asarray(m.item_factors).T", ["implicit"],
                     "Standar industri untuk umpan balik implisit (klik, tonton) dengan bobot keyakinan.", "Bobot keyakinan (alpha) dan faktor perlu disetel; hasil acak tanpa random_state."),
    "bpr_implicit": ("BPR (implicit)", "from implicit.bpr import BayesianPersonalizedRanking",
                     "def score_matrix(R, ctx):\n    m = BayesianPersonalizedRanking(factors=64, iterations=100, learning_rate=0.05, regularization=0.01, random_state=0, use_gpu=False)\n    m.fit(R.astype('float32'), show_progress=False)\n    return np.asarray(m.user_factors) @ np.asarray(m.item_factors).T", ["implicit"],
                     "Mengoptimalkan urutan (item ditonton di atas yang tidak) langsung, sesuai metrik ranking.", "Sensitif learning rate dan iterasi; hasil bervariasi antar seed."),
    "content_genre": ("Berbasis konten (profil genre)", "",
                      "def score_matrix(R, ctx):\n    Gm = ctx.movies[ctx.genre_cols].to_numpy(np.float32)                 # item x genre\n    prof = R @ Gm                                                         # profil genre pengguna\n    prof = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9); Gn = Gm / (np.linalg.norm(Gm, axis=1, keepdims=True) + 1e-9)\n    return prof @ Gn.T", [],
                      "Item baru tanpa interaksi (cold-start item): hanya butuh atribut konten.", "Terbatas pada genre kasar; kurang personal dari kolaboratif."),
    "hybrid_knn_content": ("Hibrida Item-KNN + konten", "",
                           "def score_matrix(R, ctx):\n    rn = lambda M: (np.argsort(np.argsort(M, axis=1), axis=1) / M.shape[1])         # normalisasi rank per pengguna\n    Gm = ctx.movies[ctx.genre_cols].to_numpy(np.float32); prof = R @ Gm; prof /= (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9)\n    content = prof @ (Gm / (np.linalg.norm(Gm, axis=1, keepdims=True) + 1e-9)).T\n    return 0.8 * rn(item_knn_scores(R, k=50, shrink=10)) + 0.2 * rn(content)", [],
                           "Menggabungkan sinyal kolaboratif dan konten; praktik umum untuk menutup kelemahan masing-masing.", "Bobot campuran perlu disetel di validation; normalisasi skor antar-metode wajib (di sini lewat rank)."),
}


def _register_all():
    for key, (title, imp, code, req, when, pit) in _RATE.items():
        register(modality="recsys", task="rating_prediction", key=key, recipe="rec_rating", title=title, imports=IMPR + ("\n" + imp if imp else ""),
                 model_code=code, when=when, pitfalls=pit, params=dict(ML), requires=req, heavy=key == "torch_mf")
    for key, (title, imp, code, req, when, pit) in _TOPN.items():
        register(modality="recsys", task="top_n", key=key, recipe="rec_topn", title=title, imports=IMPT + ("\n" + imp if imp else ""),
                 model_code=code, when=when, pitfalls=pit, params=dict(ML), requires=req)
    register_skip("recsys", "sequence_recommendation", "*", "model berurutan (GRU4Rec, SASRec) membutuhkan data sesi besar; lihat cookbook untuk pola dasar")


_register_all()
