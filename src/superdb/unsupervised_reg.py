"""Pendaftaran notebook unsupervised."""
from .registry import register

U = "data_latihan/unsupervised/"
WINE = dict(path=U + "wine.csv", label_col="cultivar", drop_cols=[], k=3, scale=True, sweep=True, name_data="Wine (3 kultivar, 13 fitur)")
MOONS = dict(path=U + "moons_SINTETIS.csv", label_col="true", drop_cols=[], k=2, scale=True, sweep=False, name_data="Dua bulan sabit (SINTETIS)")
CIRCLES = dict(path=U + "circles_SINTETIS.csv", label_col="true", drop_cols=[], k=2, scale=True, sweep=False, name_data="Dua lingkaran (SINTETIS)")

# ------------------------------------------------------------------------------ clustering
# (judul, imports, kode `model` (memakai Z dan K), nama parameter jumlah cluster atau None, kapan, jebakan)
_CL = {
    "kmeans": ("KMeans", "from sklearn.cluster import KMeans", "model = KMeans(n_clusters=K, n_init=10, random_state=42)", "n_clusters",
               "Baseline clustering: cluster berbentuk bola dengan ukuran mirip; cepat; centroid mudah dijelaskan.", "Harus memilih K; peka skala fitur (data sudah di-standarisasi); gagal pada bentuk non-konveks."),
    "minibatch_kmeans": ("MiniBatchKMeans", "from sklearn.cluster import MiniBatchKMeans", "model = MiniBatchKMeans(n_clusters=K, n_init=10, batch_size=256, random_state=42)", "n_clusters",
                         "Data sangat besar: KMeans versi mini-batch jauh lebih cepat dengan hasil mendekati.", "Sedikit kurang akurat; hasil bervariasi antar seed."),
    "bisecting_kmeans": ("Bisecting KMeans", "from sklearn.cluster import BisectingKMeans", "model = BisectingKMeans(n_clusters=K, random_state=42)", "n_clusters",
                         "KMeans hierarkis (membelah cluster terbesar); sering lebih stabil dan menghasilkan hierarki.", "Tetap berasumsi cluster konveks."),
    "kmeans_pca": ("PCA lalu KMeans", "from sklearn.cluster import KMeans\nfrom sklearn.decomposition import PCA\nfrom sklearn.pipeline import make_pipeline",
                   "model = make_pipeline(PCA(n_components=0.9, random_state=42), KMeans(n_clusters=K, n_init=10, random_state=42))", "kmeans__n_clusters",
                   "Fitur banyak dan berkorelasi: kompresi PCA (90% varians) sebelum clustering mengurangi noise.", "PCA bisa membuang sinyal kecil yang justru membedakan cluster."),
    "agglomerative_ward": ("Hierarchical Ward", "from sklearn.cluster import AgglomerativeClustering", "model = AgglomerativeClustering(n_clusters=K, linkage='ward')", "n_clusters",
                           "Ingin hierarki/dendrogram; Ward meminimalkan varians dalam cluster (mirip KMeans).", "O(n²) memori; hanya beberapa ribu baris. Tidak bisa memprediksi data baru."),
    "agglomerative_average": ("Hierarchical average linkage", "from sklearn.cluster import AgglomerativeClustering", "model = AgglomerativeClustering(n_clusters=K, linkage='average')", "n_clusters",
                              "Pembanding linkage lain; lebih tahan terhadap bentuk non-bola dibanding Ward pada data tertentu.", "Rentan menghasilkan satu cluster raksasa + cluster kecil bila ada outlier."),
    "agglomerative_complete": ("Hierarchical complete linkage", "from sklearn.cluster import AgglomerativeClustering", "model = AgglomerativeClustering(n_clusters=K, linkage='complete')", "n_clusters",
                               "Cluster ringkas dengan diameter kecil.", "Sensitif terhadap outlier."),
    "gmm_full": ("Gaussian Mixture (full)", "from sklearn.mixture import GaussianMixture", "model = GaussianMixture(n_components=K, covariance_type='full', n_init=3, random_state=42)", "n_components",
                 "Cluster berbentuk elips berorientasi bebas, dan butuh probabilitas keanggotaan (soft clustering). K bisa dipilih lewat BIC.", "Banyak parameter; butuh data cukup; bisa gagal konvergen di dimensi tinggi."),
    "gmm_diag": ("Gaussian Mixture (diag)", "from sklearn.mixture import GaussianMixture", "model = GaussianMixture(n_components=K, covariance_type='diag', n_init=3, random_state=42)", "n_components",
                 "Seperti GMM penuh tetapi jauh lebih sedikit parameter; stabil untuk data berdimensi tinggi.", "Mengabaikan korelasi antar fitur."),
    "spectral": ("Spectral Clustering", "from sklearn.cluster import SpectralClustering", "model = SpectralClustering(n_clusters=K, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', random_state=42)", "n_clusters",
                 "Cluster berbentuk rumit (bulan sabit, cincin) yang terhubung secara lokal.", "O(n²-n³); hanya beberapa ribu baris; harus memilih K dan n_neighbors."),
    "birch": ("BIRCH", "from sklearn.cluster import Birch", "model = Birch(n_clusters=K, threshold=0.7)", "n_clusters",
              "Data sangat besar dan streaming: membangun ringkasan pohon dalam satu lintasan.", "Asumsi cluster bola; threshold perlu disetel."),
    "affinity_propagation": ("Affinity Propagation", "from sklearn.cluster import AffinityPropagation", "model = AffinityPropagation(damping=0.9, random_state=42)", None,
                             "Jumlah cluster tidak diketahui dan ingin contoh perwakilan (exemplar) nyata untuk tiap cluster.", "O(n²) dan sering menghasilkan terlalu banyak cluster; peka terhadap preference."),
    "meanshift": ("Mean Shift", "from sklearn.cluster import MeanShift, estimate_bandwidth", "model = MeanShift(bandwidth=estimate_bandwidth(Z, quantile=0.3, random_state=42))", None,
                  "Mencari puncak kepadatan tanpa menentukan K; hanya bandwidth.", "Lambat; sangat peka terhadap bandwidth; dimensi tinggi buruk."),
    "dbscan": ("DBSCAN", "from sklearn.cluster import DBSCAN\nfrom sklearn.neighbors import NearestNeighbors",
               "kd = np.sort(NearestNeighbors(n_neighbors=5).fit(Z).kneighbors(Z)[0][:, -1])      # jarak ke tetangga ke-5 (k-distance)\nEPS = float(np.percentile(kd, 90))                              # eps dari persentil ke-90 k-distance\nmodel = DBSCAN(eps=EPS, min_samples=5)", None,
               "Bentuk bebas dan ada outlier: titik jarang diberi label noise (-1). Tidak perlu K.", "eps sangat menentukan (pilih dari grafik k-distance). Gagal bila kepadatan antar cluster berbeda."),
    "hdbscan": ("HDBSCAN", "from sklearn.cluster import HDBSCAN", "model = HDBSCAN(min_cluster_size=8)", None,
                "Seperti DBSCAN tetapi menangani kepadatan berbeda dan tidak butuh eps; pilihan modern untuk bentuk bebas.", "min_cluster_size menentukan; banyak titik bisa menjadi noise; dimensi tinggi buruk (kutukan dimensi)."),
    "optics": ("OPTICS", "from sklearn.cluster import OPTICS", "model = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)", None,
               "Varian DBSCAN untuk kepadatan bervariasi, dengan plot reachability.", "Lambat; parameter xi menentukan hasil; sering noise berlebih."),
}
_SHAPES = {k: _CL[k] for k in ("kmeans", "gmm_full", "agglomerative_ward", "agglomerative_average", "spectral", "dbscan", "hdbscan", "meanshift", "optics")}
_SHAPES["agglomerative_single"] = ("Hierarchical single linkage", "from sklearn.cluster import AgglomerativeClustering",
                                   "model = AgglomerativeClustering(n_clusters=K, linkage='single')", "n_clusters",
                                   "Cluster berbentuk rantai/bulan sabit: single linkage menyambung titik terdekat.", "Efek rantai: sebuah jembatan noise menyatukan dua cluster; peka outlier.")

# ------------------------------------------------------------------------------ anomali
AN = dict(path=U + "breast_cancer_anomaly.csv", label_col="is_anomaly", test_size=0.3, name_data="Breast cancer: 357 jinak (normal) + 22 ganas (anomali)")
_AN = {
    "isolation_forest": ("Isolation Forest", "from sklearn.ensemble import IsolationForest",
                         "def score_fn(Xfit, Xeval):\n    m = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1).fit(Xfit)\n    return -m.score_samples(Xeval)", [],
                         "Pilihan pertama untuk data tabular: cepat, tanpa asumsi distribusi, skala besar.", "Kurang baik bila anomali berada di subruang tertentu (fitur banyak yang tidak relevan)."),
    "lof_novelty": ("Local Outlier Factor (novelty=True)", "from sklearn.neighbors import LocalOutlierFactor",
                    "def score_fn(Xfit, Xeval):\n    m = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xfit)\n    return -m.score_samples(Xeval)", [],
                    "Kepadatan lokal berbeda antar wilayah: anomali = jauh lebih jarang dari tetangganya.", "Dimensi tinggi melemahkannya. novelty=True untuk skor pada data baru; fit_predict hanya untuk data yang sama."),
    "ocsvm": ("One-Class SVM", "from sklearn.svm import OneClassSVM",
              "def score_fn(Xfit, Xeval):\n    m = OneClassSVM(nu=0.05, gamma='scale').fit(Xfit)\n    return -m.decision_function(Xeval)", [],
              "Batas normal non-linear; baik bila data latih hampir seluruhnya normal.", "Peka terhadap nu/gamma; lambat untuk data besar; kontaminasi di data latih merusak batas."),
    "elliptic_envelope": ("Elliptic Envelope (kovarian robust)", "from sklearn.covariance import EllipticEnvelope",
                          "def score_fn(Xfit, Xeval):\n    m = EllipticEnvelope(support_fraction=0.9, contamination=0.05, random_state=42).fit(Xfit)\n    return -m.score_samples(Xeval)", [],
                          "Data kira-kira Gaussian: jarak Mahalanobis dengan kovarian robust (MCD).", "Gagal bila distribusi multimodal atau fitur sangat banyak dibanding sampel."),
    "mahalanobis": ("Jarak Mahalanobis (Ledoit-Wolf)", "from sklearn.covariance import LedoitWolf",
                    "def score_fn(Xfit, Xeval):\n    return LedoitWolf().fit(Xfit).mahalanobis(Xeval)", [],
                    "Baseline statistik klasik yang mempertimbangkan korelasi antar fitur.", "Asumsi satu gumpalan Gaussian."),
    "knn_distance": ("Jarak ke k tetangga terdekat", "from sklearn.neighbors import NearestNeighbors",
                     "def score_fn(Xfit, Xeval):\n    d, _ = NearestNeighbors(n_neighbors=5).fit(Xfit).kneighbors(Xeval)\n    return d.mean(1)", [],
                     "Sederhana dan sering kompetitif; mudah dijelaskan.", "Peka skala dan dimensi; tidak mengoreksi perbedaan kepadatan lokal (LOF mengoreksi)."),
    "pca_reconstruction": ("Galat rekonstruksi PCA", "from sklearn.decomposition import PCA",
                           "def score_fn(Xfit, Xeval):\n    p = PCA(n_components=0.9, random_state=42).fit(Xfit)\n    return ((Xeval - p.inverse_transform(p.transform(Xeval))) ** 2).sum(1)", [],
                           "Data berkorelasi tinggi: normal hidup di subruang berdimensi rendah, anomali menyimpang darinya.", "Anomali yang searah dengan komponen utama tidak terdeteksi."),
    "gmm_density": ("Kepadatan Gaussian Mixture", "from sklearn.mixture import GaussianMixture",
                    "def score_fn(Xfit, Xeval):\n    m = GaussianMixture(n_components=3, covariance_type='full', reg_covar=1e-2, random_state=42).fit(Xfit)\n    return -m.score_samples(Xeval)", [],
                    "Data normal terdiri dari beberapa gumpalan; skor = log-likelihood negatif.", "Dimensi tinggi butuh regularisasi kovarian; pilih jumlah komponen dengan BIC."),
    "kde": ("Kernel Density Estimation", "from sklearn.neighbors import KernelDensity",
            "def score_fn(Xfit, Xeval):\n    m = KernelDensity(bandwidth=1.5).fit(Xfit)\n    return -m.score_samples(Xeval)", [],
            "Dimensi rendah-menengah; estimasi kepadatan non-parametrik.", "Bandwidth menentukan; buruk di dimensi tinggi."),
    "zscore_max": ("Z-score maksimum per fitur", "",
                   "def score_fn(Xfit, Xeval):\n    mu, sd = Xfit.mean(0), Xfit.std(0) + 1e-9\n    return np.abs((Xeval - mu) / sd).max(1)", [],
                   "Baseline paling sederhana: apakah ada fitur yang ekstrem?", "Mengabaikan korelasi dan anomali yang tiap fiturnya normal tetapi kombinasinya aneh."),
    "ecod": ("ECOD (pyod)", "from pyod.models.ecod import ECOD",
             "def score_fn(Xfit, Xeval):\n    return ECOD().fit(Xfit).decision_function(Xeval)", ["pyod"],
             "Tanpa hiperparameter: baseline kuat yang cepat untuk data tabular berdimensi tinggi.", "Mengasumsikan fitur independen (ekor distribusi per fitur)."),
    "copod": ("COPOD (pyod)", "from pyod.models.copod import COPOD",
              "def score_fn(Xfit, Xeval):\n    return COPOD().fit(Xfit).decision_function(Xeval)", ["pyod"],
              "Berbasis copula; tanpa hiperparameter, cepat.", "Sama seperti ECOD: ketergantungan antar fitur ditangkap kasar."),
    "hbos": ("HBOS (pyod)", "from pyod.models.hbos import HBOS",
             "def score_fn(Xfit, Xeval):\n    return HBOS(n_bins=20).fit(Xfit).decision_function(Xeval)", ["pyod"],
             "Sangat cepat (histogram per fitur); cocok data besar sebagai penyaring awal.", "Mengabaikan interaksi antar fitur."),
    "autoencoder_torch": ("Autoencoder (PyTorch)", "from superdb.unsup_nets import TorchAutoencoder",
                          "def score_fn(Xfit, Xeval):\n    ae = TorchAutoencoder(n_components=5, hidden=32, epochs=150, seed=0).fit(Xfit)\n    return ae.reconstruction_error(Xeval)", ["torch"],
                          "Data besar dan pola normal kompleks; galat rekonstruksi sebagai skor.", "Pada data kecil (di sini 265 baris) cenderung kalah dari metode klasik; belajar anomali jika data latih terkontaminasi."),
}

# ------------------------------------------------------------------------------ reduksi dimensi
DR = dict(path=U + "digits.csv", label_col="digit", scale="minmax", nc=2, name_data="Digits (1797 gambar 8x8, 10 kelas)")
# (judul, imports, kode `reducer` (NC), supervised, bisa-sapuan-NC, kapan, jebakan)
_DR = {
    "pca": ("PCA", "from sklearn.decomposition import PCA", "reducer = PCA(n_components=NC, random_state=42)", False, True,
            "Reduksi linear cepat: kompresi, visualisasi awal, dan pra-pemrosesan sebelum model.", "Hanya struktur linear dan varians besar; tidak selalu memisahkan kelas."),
    "truncated_svd": ("Truncated SVD (LSA)", "from sklearn.decomposition import TruncatedSVD", "reducer = TruncatedSVD(n_components=NC, random_state=42)", False, True,
                      "Matriks sparse (teks TF-IDF) tanpa pemusatan; mirip PCA.", "Tidak memusatkan data; komponen pertama sering mewakili rata-rata."),
    "kernel_pca_rbf": ("Kernel PCA (RBF)", "from sklearn.decomposition import KernelPCA", "reducer = KernelPCA(n_components=NC, kernel='rbf', gamma=0.02, random_state=42)", False, True,
                       "Struktur non-linear; PCA di ruang fitur kernel.", "gamma menentukan; O(n²) memori; tidak ada inverse_transform praktis."),
    "nmf": ("NMF", "from sklearn.decomposition import NMF", "reducer = NMF(n_components=NC, init='nndsvda', max_iter=500, random_state=42)", False, True,
            "Data non-negatif (piksel, hitungan kata): komponen berupa bagian-bagian yang bisa ditafsirkan.", "Input harus non-negatif; hasil tidak unik."),
    "factor_analysis": ("Factor Analysis", "from sklearn.decomposition import FactorAnalysis", "reducer = FactorAnalysis(n_components=NC, random_state=42)", False, True,
                        "Ingin memodelkan noise tiap fitur secara terpisah (psikometri, survei).", "Asumsi Gaussian; lambat di dimensi tinggi."),
    "fastica": ("FastICA", "from sklearn.decomposition import FastICA", "reducer = FastICA(n_components=NC, whiten='unit-variance', max_iter=1000, random_state=42)", False, True,
                "Memisahkan sumber sinyal independen (EEG, suara).", "Komponen tanpa urutan/tanda tetap; sensitif inisialisasi."),
    "random_projection": ("Random Projection", "from sklearn.random_projection import GaussianRandomProjection", "reducer = GaussianRandomProjection(n_components=NC, random_state=42)", False, True,
                          "Reduksi sangat cepat untuk dimensi ribuan dengan jaminan menjaga jarak (Johnson-Lindenstrauss).", "Hasil acak; kurang baik untuk dimensi target sangat kecil (2D)."),
    "lda_supervised": ("LDA (supervised)", "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis", "reducer = LinearDiscriminantAnalysis(n_components=NC)", True, True,
                       "Ada label: proyeksi yang memaksimalkan pemisahan kelas (maks C-1 dimensi).", "Menggunakan label: evaluasi harus di dalam CV agar tidak bocor. Asumsi Gaussian per kelas."),
    "tsne": ("t-SNE", "from sklearn.manifold import TSNE", "reducer = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)", False, False,
             "Visualisasi 2D yang memperlihatkan cluster lokal dengan jelas.", "Hanya untuk visualisasi: jarak antar cluster dan ukuran cluster tidak bermakna; tidak ada transform untuk data baru; perplexity menentukan."),
    "umap": ("UMAP", "import umap", "reducer = umap.UMAP(n_components=NC, n_neighbors=15, min_dist=0.1, random_state=42)", False, True,
             "Visualisasi dan reduksi non-linear yang menjaga struktur lokal dan lebih global dari t-SNE; punya transform.", "Hiperparameter memengaruhi bentuk; jangan menafsirkan jarak antar cluster secara kuantitatif."),
    "isomap": ("Isomap", "from sklearn.manifold import Isomap", "reducer = Isomap(n_components=NC, n_neighbors=10)", False, True,
               "Data terletak pada manifold yang melengkung dengan jarak geodesik bermakna.", "Peka terhadap n_neighbors dan noise (korsleting graf)."),
    "lle": ("Locally Linear Embedding", "from sklearn.manifold import LocallyLinearEmbedding", "reducer = LocallyLinearEmbedding(n_components=NC, n_neighbors=12, random_state=42)", False, True,
            "Manifold halus dengan struktur lokal linear.", "Rapuh pada noise; sering menghasilkan embedding yang terlipat."),
    "spectral_embedding": ("Spectral Embedding", "from sklearn.manifold import SpectralEmbedding", "reducer = SpectralEmbedding(n_components=NC, n_neighbors=15, random_state=42)", False, False,
                           "Dasar untuk spectral clustering; menjaga konektivitas graf tetangga.", "Tidak ada transform untuk data baru."),
    "mds": ("MDS", "from sklearn.manifold import MDS", "reducer = MDS(n_components=NC, n_init=1, max_iter=100, random_state=42, init='random')", False, False,
            "Menjaga jarak antar titik sedekat mungkin; bila matriks jarak yang tersedia.", "Lambat O(n²); tidak ada transform."),
    "autoencoder_torch": ("Autoencoder (PyTorch)", "from superdb.unsup_nets import TorchAutoencoder", "reducer = TorchAutoencoder(n_components=NC, hidden=64, epochs=150, seed=0)", False, True,
                          "Data besar; representasi non-linear yang juga bisa dipakai sebagai fitur.", "Hasil bervariasi antar seed; perlu data cukup dan tuning."),
}

# ------------------------------------------------------------------------------ asosiasi
AS = dict(path=U + "groceries.csv", min_support=0.01, min_conf=0.2, name_data="Groceries (9835 transaksi belanja)")
_AS = {
    "apriori": ("Apriori", "from mlxtend.frequent_patterns import apriori", "algo = lambda df, s: apriori(df, min_support=s, use_colnames=True)",
                "Algoritma klasik itemset sering; mudah dijelaskan; baik untuk data kecil-menengah.", "Lambat bila support rendah atau item banyak (ledakan kombinasi)."),
    "fpgrowth": ("FP-Growth", "from mlxtend.frequent_patterns import fpgrowth", "algo = lambda df, s: fpgrowth(df, min_support=s, use_colnames=True)",
                 "Hasil sama dengan Apriori tetapi jauh lebih cepat pada data besar.", "Memori untuk pohon FP bisa besar pada support sangat rendah."),
    "fpmax": ("FP-Max (itemset maksimal)", "from mlxtend.frequent_patterns import fpmax", "algo = lambda df, s: fpmax(df, min_support=s, use_colnames=True)",
              "Hanya itemset maksimal (ringkasan): berguna bila jumlah itemset sering terlalu banyak.", "Tidak menghasilkan semua subset, jadi tidak cukup untuk membuat aturan lengkap dengan confidence."),
}

# ------------------------------------------------------------------------------ seleksi fitur
FS = dict(path=U + "breast_cancer_noisy.csv", target="diagnosis", k=8, name_data="Breast cancer + 30 fitur noise + 4 duplikat")
IMPFS = "import numpy as np, pandas as pd\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler"
_FS = {
    "variance_threshold": ("Variance Threshold", "from sklearn.feature_selection import VarianceThreshold",
                           "def select(Xtr, ytr, names):\n    Xs = StandardScaler().fit_transform(Xtr)\n    sel = VarianceThreshold(threshold=0.2).fit(Xs)\n    return [n for n, k in zip(names, sel.get_support()) if k], None", [],
                           "Membuang fitur konstan/hampir konstan. Langkah pembersihan awal, murah.", "Tidak mempertimbangkan target: fitur noise bervarians tinggi TIDAK terbuang (lihat hasil)."),
    "correlation_filter": ("Filter korelasi", "",
                           "def select(Xtr, ytr, names):\n    C = pd.DataFrame(Xtr, columns=names).corr().abs(); keep = []\n    for c in names:\n        if all(C.loc[c, k] < 0.95 for k in keep):\n            keep.append(c)\n    return keep, None", [],
                           "Membuang fitur duplikat/sangat berkorelasi (redundan), terutama untuk model linear.", "Tidak memeriksa relevansi terhadap target; urutan kolom menentukan mana yang dipertahankan."),
    "anova_f": ("ANOVA F-test (SelectKBest)", "from sklearn.feature_selection import f_classif",
                "def select(Xtr, ytr, names):\n    F, _ = f_classif(Xtr, ytr); r = pd.Series(F, index=names).sort_values(ascending=False)\n    return list(r.index[:K]), r", [],
                "Filter cepat untuk hubungan linear fitur numerik dengan kelas.", "Hanya efek marginal tiap fitur; melewatkan interaksi; fitur duplikat sama-sama terpilih."),
    "mutual_info": ("Mutual Information", "from sklearn.feature_selection import mutual_info_classif",
                    "def select(Xtr, ytr, names):\n    mi = mutual_info_classif(Xtr, ytr, random_state=0); r = pd.Series(mi, index=names).sort_values(ascending=False)\n    return list(r.index[:K]), r", [],
                    "Hubungan non-linear antara fitur dan target.", "Estimasi noisy pada data kecil; duplikat sama-sama tinggi."),
    "chi2": ("Chi-kuadrat", "from sklearn.feature_selection import chi2",
             "def select(Xtr, ytr, names):\n    Xm = MinMaxScaler().fit_transform(Xtr); c, _ = chi2(Xm, ytr); r = pd.Series(c, index=names).sort_values(ascending=False)\n    return list(r.index[:K]), r", [],
             "Fitur non-negatif/hitungan (teks, kategorikal one-hot) terhadap target kategorikal.", "Wajib non-negatif (di sini MinMax)."),
    "lasso_l1": ("Logistic Regression L1 (SelectFromModel)", "from sklearn.feature_selection import SelectFromModel\nfrom sklearn.linear_model import LogisticRegression",
                 "def select(Xtr, ytr, names):\n    Xs = StandardScaler().fit_transform(Xtr)\n    m = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=0).fit(Xs, ytr)\n    r = pd.Series(np.abs(m.coef_[0]), index=names).sort_values(ascending=False)\n    return [n for n in names if r[n] > 1e-6], r", [],
                 "Seleksi embedded: koefisien nol berarti fitur dibuang, sambil melatih model.", "Dari fitur berkorelasi, L1 hanya menyimpan satu (acak); C menentukan jumlah fitur."),
    "tree_importance": ("Importance Random Forest", "from sklearn.ensemble import RandomForestClassifier",
                        "def select(Xtr, ytr, names):\n    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)\n    r = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)\n    return list(r.index[:K]), r", [],
                        "Menangkap interaksi dan non-linearitas; ranking cepat.", "Bias ke fitur kardinalitas tinggi; fitur berkorelasi membagi importance."),
    "lgbm_gain": ("Importance LightGBM (gain)", "from lightgbm import LGBMClassifier",
                  "def select(Xtr, ytr, names):\n    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5, random_state=0, verbose=-1).fit(Xtr, ytr)\n    r = pd.Series(m.booster_.feature_importance('gain'), index=names).sort_values(ascending=False)\n    return list(r.index[:K]), r", ["lightgbm"],
                  "Ranking dari boosting; cepat dan sering akurat.", "Gain bias ke fitur kontinu; noise bisa mendapat gain kecil tapi tidak nol."),
    "rfe": ("RFE (Recursive Feature Elimination)", "from sklearn.feature_selection import RFE\nfrom sklearn.linear_model import LogisticRegression",
            "def select(Xtr, ytr, names):\n    Xs = StandardScaler().fit_transform(Xtr)\n    r = RFE(LogisticRegression(max_iter=2000), n_features_to_select=K, step=2).fit(Xs, ytr)\n    rk = pd.Series(r.ranking_, index=names).sort_values()\n    return [n for n, s in zip(names, r.support_) if s], (1.0 / rk)", [],
            "Membuang fitur terlemah berulang-ulang menurut model; mempertimbangkan fitur bersama-sama.", "Lebih lambat; K harus ditentukan (atau pakai RFECV)."),
    "rfecv": ("RFECV (jumlah fitur otomatis)", "from sklearn.feature_selection import RFECV\nfrom sklearn.linear_model import LogisticRegression",
              "def select(Xtr, ytr, names):\n    Xs = StandardScaler().fit_transform(Xtr)\n    r = RFECV(LogisticRegression(max_iter=2000), step=2, cv=3, scoring='f1_macro', min_features_to_select=3).fit(Xs, ytr)\n    return [n for n, s in zip(names, r.support_) if s], None", [],
              "Jumlah fitur terbaik dipilih otomatis lewat CV.", "Lebih lambat; CV di dalam CV; bisa memilih terlalu banyak fitur bila kurva datar."),
    "permutation": ("Permutation importance", "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.inspection import permutation_importance\nfrom sklearn.model_selection import train_test_split",
                    "def select(Xtr, ytr, names):\n    a, b, ya, yb = train_test_split(Xtr, ytr, test_size=0.3, random_state=0, stratify=ytr)\n    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(a, ya)\n    pi = permutation_importance(rf, b, yb, n_repeats=10, random_state=0, scoring='f1_macro')\n    r = pd.Series(pi.importances_mean, index=names).sort_values(ascending=False)\n    return [n for n in names if r[n] > 0.002], r", [],
                    "Importance yang dihitung pada data held-out (lebih jujur dari importance bawaan pohon).", "Fitur berkorelasi saling menutupi (masing-masing tampak tidak penting)."),
    "boruta_like": ("Boruta-like (shadow features)", "from superdb.unsupervised import ShadowSelector",
                    "def select(Xtr, ytr, names):\n    s = ShadowSelector(n_iter=15, random_state=0).fit(Xtr, ytr)\n    r = pd.Series(s.hits_, index=names).sort_values(ascending=False)\n    return [n for n, k in zip(names, s.support_) if k], r", [],
                    "Ingin memastikan fitur terpilih benar-benar lebih baik dari noise acak (kontrol negatif bawaan).", "Lebih lambat; mempertahankan fitur redundan (tidak memilih yang minimal)."),
    "sequential_forward": ("Sequential Forward Selection", "from sklearn.feature_selection import SequentialFeatureSelector\nfrom sklearn.linear_model import LogisticRegression",
                           "def select(Xtr, ytr, names):\n    Xs = StandardScaler().fit_transform(Xtr)\n    s = SequentialFeatureSelector(LogisticRegression(max_iter=2000), n_features_to_select=K, direction='forward', cv=3, n_jobs=-1).fit(Xs, ytr)\n    return [n for n, k in zip(names, s.get_support()) if k], None", [],
                           "Memilih fitur satu per satu yang paling menaikkan skor CV; sangat relevan untuk model akhir tertentu.", "Mahal (fit sebanyak fitur × K × lipatan); bisa overfit ke CV pada data kecil."),
}

# ------------------------------------------------------------------------------ imputasi
IM = dict(path=U + "wine.csv", label_col="cultivar", frac=0.2, name_data="Wine dengan 20% nilai dihapus acak (MCAR) secara buatan")
_IM = {
    "mean": ("Imputasi rata-rata", "from sklearn.impute import SimpleImputer", "imputer = SimpleImputer(strategy='mean')", "Baseline paling sederhana.", "Mengecilkan varians dan merusak korelasi antar fitur."),
    "median": ("Imputasi median", "from sklearn.impute import SimpleImputer", "imputer = SimpleImputer(strategy='median')", "Fitur miring/ada outlier.", "Sama seperti rata-rata: korelasi rusak."),
    "constant_zero": ("Imputasi konstanta 0", "from sklearn.impute import SimpleImputer", "imputer = SimpleImputer(strategy='constant', fill_value=0)", "Hanya bila 0 memang bermakna 'tidak ada'. Pembanding buruk yang jujur.", "Pada fitur yang tidak bermakna nol, merusak distribusi."),
    "mean_indicator": ("Rata-rata + penanda missing", "from sklearn.impute import SimpleImputer", "imputer = SimpleImputer(strategy='mean', add_indicator=True)", "Missing itu sendiri informatif (tidak acak); model pohon/linear boleh memanfaatkannya.", "Menambah kolom; berguna hanya bila missing berkaitan dengan target."),
    "knn5": ("KNN imputer (k=5)", "from sklearn.impute import KNNImputer", "imputer = KNNImputer(n_neighbors=5)", "Baris mirip punya nilai mirip; menjaga struktur lokal. Data sudah dalam skala sebanding.", "Wajib skala seragam; lambat untuk data besar."),
    "iterative_bayesridge": ("Iterative (MICE, BayesianRidge)", "from sklearn.experimental import enable_iterative_imputer  # noqa\nfrom sklearn.impute import IterativeImputer",
                             "imputer = IterativeImputer(max_iter=10, random_state=42)", "Fitur saling berkorelasi linear: tiap fitur diregresikan pada fitur lain secara bergantian.", "Lambat di fitur sangat banyak; asumsi linear."),
    "iterative_extratrees": ("Iterative (ala MissForest, ExtraTrees)", "from sklearn.experimental import enable_iterative_imputer  # noqa\nfrom sklearn.impute import IterativeImputer\nfrom sklearn.ensemble import ExtraTreesRegressor",
                             "imputer = IterativeImputer(ExtraTreesRegressor(n_estimators=50, n_jobs=-1, random_state=42), max_iter=5, random_state=42)", "Hubungan non-linear dan interaksi; sering paling akurat.", "Paling lambat; hasil bergantung seed."),
}


def _register_all():
    for key, (title, imp, code, kp, when, pit) in _CL.items():
        register(modality="unsupervised", task="clustering", key=key, recipe="uns_cluster", title=title, imports=imp, model_code=code,
                 when=when, pitfalls=pit, params=dict(WINE, kparam=kp))
    for data, task in ((MOONS, "clustering_moons"), (CIRCLES, "clustering_circles")):
        for key, (title, imp, code, kp, when, pit) in _SHAPES.items():
            register(modality="unsupervised", task=task, key=key, recipe="uns_cluster", title=title, imports=imp, model_code=code,
                     when=when, pitfalls=pit, params=dict(data, kparam=kp))
    for key, (title, imp, code, req, when, pit) in _AN.items():
        register(modality="unsupervised", task="anomaly_detection", key=key, recipe="uns_anomaly", title=title, imports="import numpy as np\n" + imp,
                 model_code=code, when=when, pitfalls=pit, params=dict(AN), requires=req, heavy=key.endswith("_torch"))
    for key, (title, imp, code, sup, sweep, when, pit) in _DR.items():
        register(modality="unsupervised", task="dim_reduction", key=key, recipe="uns_dimred", title=title, imports=imp, model_code=code,
                 when=when, pitfalls=pit, params=dict(DR, supervised=sup, sweep=sweep), requires=(["umap"] if key == "umap" else ["torch"] if key.endswith("_torch") else []),
                 heavy=key.endswith("_torch"))
    for key, (title, imp, code, when, pit) in _AS.items():
        register(modality="unsupervised", task="association_rules", key=key, recipe="uns_assoc", title=title, imports=imp, model_code=code,
                 when=when, pitfalls=pit, params=dict(AS), requires=["mlxtend"])
    for key, (title, imp, code, req, when, pit) in _FS.items():
        register(modality="unsupervised", task="feature_selection", key=key, recipe="uns_featsel", title=title, imports=IMPFS + ("\n" + imp if imp else ""),
                 model_code=code, when=when, pitfalls=pit, params=dict(FS), requires=req)
    for key, (title, imp, code, when, pit) in _IM.items():
        register(modality="unsupervised", task="imputation", key=key, recipe="uns_impute", title=title, imports=imp, model_code=code,
                 when=when, pitfalls=pit, params=dict(IM))
    register(modality="unsupervised", task="preprocessing", key="scaling_compare", recipe="uns_preproc", title="Perbandingan scaler/transformer",
             when="Model sensitif skala (KNN, SVM, logistik, neural) dan fitur punya skala/distribusi berbeda.",
             pitfalls="Scaler HARUS dilatih hanya pada data training (di dalam Pipeline). Model pohon tidak butuh scaling.",
             params=dict(path=U + "wine.csv", label_col="cultivar", method="scaling"))
    register(modality="unsupervised", task="preprocessing", key="outlier_treatment", recipe="uns_preproc", title="Penanganan outlier: potong, buang, atau biarkan",
             when="Ada nilai ekstrem yang mungkin galat pengukuran dan merusak model linear.",
             pitfalls="Outlier bisa jadi sinyal sah (mis. transaksi besar). Jangan membuang tanpa alasan domain; validasi pada data yang TIDAK dibersihkan.",
             params=dict(path="data_latihan/tabular/regression/diamonds_train.csv", label_col="price", method="outlier"))


_register_all()
