"""Pendaftaran notebook graph: klasifikasi simpul, link prediction, deteksi komunitas, analisis jaringan."""
from .registry import register, register_skip

CORA = dict(folder="data_latihan/graph/cora", name_data="Cora (2708 makalah, 5429 sitasi, 1433 fitur kata, 7 topik)")
IMPN = ("import numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n"
        "from superdb.graph import structural_features, propagate, ppmi_embedding, label_propagation, train_gnn")

# ------------------------------------------------------------------------- klasifikasi simpul
_NODE = {
    "features_logreg": ("Hanya fitur simpul + Logistic Regression", "",
                        "def fit_predict(g, tr, te):\n    X = g.X / np.maximum(g.X.sum(1, keepdims=True), 1)       # normalisasi baris; JANGAN StandardScaler pada fitur biner jarang\n    m = LogisticRegression(C=10, max_iter=3000).fit(X[tr], g.y[tr])\n    return m.predict_proba(X[te])", [],
                        "Baseline wajib: abaikan graf, pakai fitur simpul saja. Jika graf memberi manfaat, metode berikut harus mengalahkannya.", "Tidak memakai informasi tetangga sama sekali. Fitur teks biner jarang: normalisasi baris, bukan StandardScaler (yang menurunkan akurasi jauh)."),
    "structure_lgbm": ("Fitur struktur graf + LightGBM", "from lightgbm import LGBMClassifier",
                       "def fit_predict(g, tr, te):\n    F = structural_features(g.A)      # derajat, PageRank, klaster, core, betweenness, ...\n    m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5, random_state=0, verbose=-1).fit(F[tr], g.y[tr])\n    return m.predict_proba(F[te])", ["lightgbm"],
                       "Fitur simpul tidak ada, hanya struktur jaringan; peran simpul (hub, jembatan) menentukan label.", "Struktur saja sering lemah untuk label topik; kuat untuk label peran (influencer, bot)."),
    "label_propagation": ("Label propagation", "",
                          "def fit_predict(g, tr, te):\n    return label_propagation(g.A, g.y, tr, alpha=0.9)[te]", [],
                          "Graf bersifat homofili (tetangga cenderung sekelas) dan fitur lemah/tidak ada; sangat cepat dan tanpa pelatihan.", "Gagal pada graf heterofili (tetangga berbeda kelas) dan simpul terisolasi."),
    "sgc_logreg": ("SGC: propagasi fitur A²X + Logistic Regression", "",
                   "def fit_predict(g, tr, te):\n    X = g.X / np.maximum(g.X.sum(1, keepdims=True), 1)       # normalisasi baris\n    Z = propagate(g.A_hat, X, k=2)            # fitur dirata-ratakan dengan tetangga 2 langkah\n    m = LogisticRegression(C=100, max_iter=3000).fit(Z[tr], g.y[tr])\n    return m.predict_proba(Z[te])", [],
                   "Titik awal GNN yang sangat kuat dan murah: tanpa jaringan saraf, hanya penghalusan fitur lewat graf + model linear.", "k terlalu besar menghaluskan berlebihan (semua simpul mirip). Pada fitur biner jarang, normalisasi baris + tanpa StandardScaler (dengan StandardScaler akurasi turun dari ~0.80 ke ~0.65 pada demo)."),
    "deepwalk_logreg": ("Embedding random-walk (PPMI+SVD) + Logistic Regression", "",
                        "def fit_predict(g, tr, te):\n    Z = ppmi_embedding(g.A, dim=64)           # embedding struktur (setara DeepWalk)\n    m = LogisticRegression(C=10, max_iter=3000).fit(Z[tr], g.y[tr])\n    return m.predict_proba(Z[te])", [],
                        "Fitur simpul tidak ada tetapi graf besar; embedding menangkap posisi simpul dalam komunitas.", "Tidak memakai fitur simpul; simpul terisolasi tidak punya embedding bermakna."),
    "neighbors_lgbm": ("Fitur + fitur tetangga (SVD) + LightGBM", "from sklearn.decomposition import TruncatedSVD\nfrom lightgbm import LGBMClassifier",
                       "def fit_predict(g, tr, te):\n    Z = np.hstack([g.X, propagate(g.A_hat, g.X, k=2)])\n    Z = TruncatedSVD(n_components=100, random_state=0).fit_transform(Z)       # padatkan fitur gabungan\n    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.5, min_child_samples=5, random_state=0, verbose=-1).fit(Z[tr], g.y[tr])\n    return m.predict_proba(Z[te])", ["lightgbm"],
                       "Menggabungkan fitur sendiri dan konteks tetangga ke model tabular; mudah ditambahkan ke ensemble.", "Boosting pada fitur teks jarang biasanya kalah dari model linear."),
    "gcn_torch": ("GCN (PyTorch murni)", "",
                  "def fit_predict(g, tr, te):\n    from sklearn.model_selection import train_test_split\n    fit_idx, va = train_test_split(tr, test_size=0.2, stratify=g.y[tr], random_state=42)\n    P, hist = train_gnn('gcn', g, fit_idx, va, te, hidden=64, epochs=200, lr=0.01, dropout=0.5)\n    return P", ["torch"],
                  "Graf dengan fitur simpul dan label sedikit: GCN adalah baseline GNN standar.", "Over-smoothing pada banyak lapisan; sensitif dropout dan weight decay. Bandingkan dengan SGC (hampir setara tapi jauh lebih sederhana)."),
    "graphsage_torch": ("GraphSAGE-mean (PyTorch murni)", "",
                        "def fit_predict(g, tr, te):\n    from sklearn.model_selection import train_test_split\n    fit_idx, va = train_test_split(tr, test_size=0.2, stratify=g.y[tr], random_state=42)\n    P, hist = train_gnn('sage', g, fit_idx, va, te, hidden=64, epochs=200, lr=0.01, dropout=0.5)\n    return P", ["torch"],
                        "Memisahkan fitur sendiri dari agregasi tetangga (konkatenasi); dasar untuk graf besar dengan sampling tetangga.", "Versi full-batch ini hanya cocok untuk graf kecil-menengah."),
}

# ------------------------------------------------------------------------- link prediction
IMPL = "import numpy as np\nfrom superdb.graph import pair_heuristics, link_split, sample_negatives, ppmi_embedding"
_LINK = {
    "common_neighbors": ("Common Neighbors", "", "def link_scores(A_train, pairs):\n    return pair_heuristics(A_train, pairs)['common_neighbors'].to_numpy()", [],
                         "Baseline paling sederhana: dua simpul punya banyak tetangga bersama → kemungkinan terhubung.", "Condong ke simpul berderajat tinggi."),
    "jaccard": ("Jaccard", "", "def link_scores(A_train, pairs):\n    return pair_heuristics(A_train, pairs)['jaccard'].to_numpy()", [],
                "Common neighbors yang dinormalisasi ukuran lingkungan; mengurangi bias derajat.", "Lemah untuk simpul berderajat sangat kecil."),
    "adamic_adar": ("Adamic-Adar", "", "def link_scores(A_train, pairs):\n    return pair_heuristics(A_train, pairs)['adamic_adar'].to_numpy()", [],
                    "Tetangga bersama yang langka (berderajat rendah) lebih informatif daripada hub; sering heuristik terbaik.", "Hanya melihat jarak 2 langkah."),
    "resource_allocation": ("Resource Allocation", "", "def link_scores(A_train, pairs):\n    return pair_heuristics(A_train, pairs)['resource_allocation'].to_numpy()", [],
                            "Mirip Adamic-Adar dengan penalti hub lebih kuat (1/derajat).", "Sama seperti Adamic-Adar."),
    "preferential_attachment": ("Preferential Attachment", "", "def link_scores(A_train, pairs):\n    return pair_heuristics(A_train, pairs)['preferential_attachment'].to_numpy()", [],
                                "Jaringan skala bebas: simpul populer cenderung makin populer.", "Tidak memakai struktur lokal; sering lemah di jaringan sitasi."),
    "ppmi_embedding_cosine": ("Embedding random-walk + kemiripan cosine", "",
                              "def link_scores(A_train, pairs):\n    Z = ppmi_embedding(A_train, dim=64)\n    return (Z[pairs[:, 0]] * Z[pairs[:, 1]]).sum(1)", [],
                              "Pasangan yang berada di wilayah graf yang sama (komunitas) cenderung terhubung; menangkap lebih dari 2 langkah.", "Embedding dilatih hanya pada sisi yang diketahui; simpul terisolasi tidak terwakili."),
    "supervised_lgbm_heuristics": ("Supervised: heuristik sebagai fitur + LightGBM", "from lightgbm import LGBMClassifier",
                                   "def link_scores(A_train, pairs):\n    A_obs, pos, _ = link_split(A_train, frac=0.15, seed=1)         # sisi latih disembunyikan agar fitur tidak bocor\n    neg = sample_negatives(A_train, len(pos), seed=2); P = np.vstack([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]\n    feats = lambda A, pr: pair_heuristics(A, pr).assign(du=np.asarray(A.sum(1)).ravel()[pr[:, 0]], dv=np.asarray(A.sum(1)).ravel()[pr[:, 1]])\n    m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=0, verbose=-1).fit(feats(A_obs, P), y)\n    return m.predict_proba(feats(A_train, pairs))[:, 1]", ["lightgbm"],
                                   "Menggabungkan beberapa heuristik secara otomatis; biasanya mengungguli heuristik tunggal.", "Fitur latih harus dihitung pada graf yang sisi latihnya disembunyikan (sudah ditangani); jika tidak, terjadi kebocoran."),
}

# ------------------------------------------------------------------------- komunitas
_COMM = {
    "louvain": ("Louvain", "import networkx as nx", "def communities(G, K):\n    return [set(c) for c in nx.community.louvain_communities(G, seed=42)]", [],
                "Standar de facto deteksi komunitas: memaksimalkan modularitas, cepat dan skalabel.", "Batas resolusi: komunitas kecil bisa tergabung. Jumlah komunitas tidak bisa diatur langsung."),
    "label_propagation": ("Label Propagation (asinkron)", "import networkx as nx", "def communities(G, K):\n    return [set(c) for c in nx.community.asyn_lpa_communities(G, seed=42)]", [],
                          "Sangat cepat (hampir linear); cocok graf sangat besar.", "Tidak deterministik dan hasil bervariasi antar seed."),
    "greedy_modularity": ("Greedy Modularity (Clauset-Newman-Moore)", "import networkx as nx", "def communities(G, K):\n    return [set(c) for c in nx.community.greedy_modularity_communities(G)]", [],
                          "Deterministik dan berbasis modularitas; alternatif Louvain.", "Lebih lambat dan sering memberi komunitas lebih sedikit/besar."),
    "girvan_newman": ("Girvan-Newman (hanya graf kecil)", "import networkx as nx\nimport itertools", "def communities(G, K):\n    return [set(c) for c in next(itertools.islice(nx.community.girvan_newman(G), K - 1, K))]", [],
                      "Membuang sisi dengan betweenness tertinggi; jelas secara konsep, bagus untuk graf kecil (puluhan simpul).", "O(m²n): tidak dapat dipakai pada graf besar (dilewati untuk Cora)."),
    "spectral_clustering": ("Spectral Clustering pada adjacency", "from sklearn.cluster import SpectralClustering\nimport networkx as nx\nimport scipy.sparse as sp",
                            "def communities(G, K):\n    A = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=list(G.nodes)), dtype=float); A.indices, A.indptr = A.indices.astype(np.int32), A.indptr.astype(np.int32)   # sklearn butuh indeks int32\n    lab = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=42, assign_labels='kmeans').fit_predict(A)\n    nodes = list(G.nodes); return [set(nodes[i] for i in np.where(lab == c)[0]) for c in range(K)]", [],
                            "Jumlah komunitas K diketahui; berbasis struktur spektral graf.", "Butuh K; lambat untuk graf besar; graf tak terhubung bermasalah."),
    "embedding_kmeans": ("KMeans pada embedding random-walk", "from sklearn.cluster import KMeans\nimport networkx as nx\nimport scipy.sparse as sp\nfrom superdb.graph import ppmi_embedding",
                         "def communities(G, K):\n    nodes = list(G.nodes); A = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodes)); Z = ppmi_embedding(A, dim=32)\n    lab = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(Z); return [set(nodes[i] for i in np.where(lab == c)[0]) for c in range(K)]", [],
                         "Menggabungkan struktur graf dengan algoritma clustering klasik; K bebas dipilih.", "Embedding dan KMeans acak; butuh K."),
}


def _register_all():
    for key, (title, imp, code, req, when, pit) in _NODE.items():
        register(modality="graph", task="node_classification", key=key, recipe="gr_node", title=title, imports=IMPN + ("\n" + imp if imp else ""),
                 model_code=code, when=when, pitfalls=pit, params=dict(CORA), requires=req, heavy=key.endswith("_torch"))
    for key, (title, imp, code, req, when, pit) in _LINK.items():
        register(modality="graph", task="link_prediction", key=key, recipe="gr_link", title=title, imports=IMPL + ("\n" + imp if imp else ""),
                 model_code=code, when=when, pitfalls=pit, params=dict(CORA), requires=req)
    for key, (title, imp, code, req, when, pit) in _COMM.items():
        register(modality="graph", task="community_detection", key=key, recipe="gr_comm", title=title, imports="import numpy as np\n" + imp,
                 model_code=code, when=when, pitfalls=pit, params=dict(CORA, small_only=key == "girvan_newman"), requires=req)
    register(modality="graph", task="analysis", key="network_analysis", recipe="gr_analysis", title="Analisis jaringan: sentralitas, k-core, komponen",
             when="Langkah pertama pada data graf baru: siapa simpul penting, seberapa terhubung, apakah ada hub dan komponen terpisah.",
             pitfalls="Sentralitas berbeda mengukur hal berbeda (derajat ≠ betweenness ≠ PageRank). Hitung pada graf tak berarah/terarah sesuai makna data.", params=dict(CORA))
    register_skip("graph", "graph_classification", "*", "klasifikasi graf (satu label per graf) butuh dataset banyak-graf yang tidak dibundel; gunakan fitur graf (jumlah simpul/sisi, densitas, spektrum) + model tabular")


_register_all()
