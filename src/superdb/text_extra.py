"""Teks lanjutan: multilabel, NER, topic modeling, retrieval, kata kunci/ringkasan, clustering teks.
Bagian heavy (embedding, transformer) ada di superdb/text_heavy.py.
"""
import re

import numpy as np
import pandas as pd

from .registry import register, register_skip
from .text import ID_STOPWORDS


# ================================================================== NER
def entity_spans(tags):
    """Himpunan (mulai, akhir, tipe) dari urutan tag BIO."""
    spans, start, typ = set(), None, None
    for i, t in enumerate(list(tags) + ["O"]):
        cont = t.startswith("I-") and t[2:] == typ and start is not None
        if not cont:
            if start is not None:
                spans.add((start, i, typ))
                start, typ = None, None
            if t.startswith(("B-", "I-")):
                start, typ = i, t[2:]
    return spans


def entity_f1(true_seqs, pred_seqs):
    """Precision/recall/F1 level ENTITAS (span + tipe harus persis sama), micro dan per tipe."""
    from collections import defaultdict
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for yt, yp in zip(true_seqs, pred_seqs):
        a, b = entity_spans(yt), entity_spans(yp)
        for s in a & b:
            tp[s[2]] += 1
        for s in b - a:
            fp[s[2]] += 1
        for s in a - b:
            fn[s[2]] += 1

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    rows = {k: prf(tp[k], fp[k], fn[k]) + (tp[k] + fn[k],) for k in sorted(set(tp) | set(fp) | set(fn))}
    out = pd.DataFrame(rows, index=["precision", "recall", "f1", "support"]).T
    micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return {"precision": micro[0], "recall": micro[1], "f1": micro[2]}, out


def token_features(tokens, i, mode="rich"):
    w = tokens[i]
    f = {"bias": 1.0, "w": w.lower(), "upper": w.isupper(), "title": w.istitle(), "digit": w.isdigit()}
    if mode == "rich":
        for n in (1, 2, 3, 4):
            f[f"pre{n}"] = w[:n].lower()
            f[f"suf{n}"] = w[-n:].lower()
        f["shape"] = re.sub(r"[A-Z]", "A", re.sub(r"[a-z]", "a", re.sub(r"\d", "0", w)))
        f["hyphen"] = "-" in w
        f["len"] = min(len(w), 12)
        for d in (-2, -1, 1, 2):
            j = i + d
            if 0 <= j < len(tokens):
                f[f"{d}:w"] = tokens[j].lower()
                f[f"{d}:title"] = tokens[j].istitle()
            else:
                f[f"{d}:pad"] = True
    else:
        for d in (-1, 1):
            j = i + d
            f[f"{d}:w"] = tokens[j].lower() if 0 <= j < len(tokens) else "<pad>"
    return f


def sent_features(tokens, mode="rich"):
    return [token_features(tokens, i, mode) for i in range(len(tokens))]


def load_ner_csv(path):
    d = pd.read_csv(path, keep_default_na=False)
    sents = [(g["token"].tolist(), g["tag"].tolist()) for _, g in d.groupby("sentence_id", sort=True)]
    return [s[0] for s in sents], [s[1] for s in sents]


class TokenLogReg:
    """Tagger tingkat-token (tanpa memodelkan transisi antar-tag): pembanding untuk CRF. API sama dengan CRF sklearn_crfsuite."""

    def __init__(self, C=3.0):
        self.C = C

    def fit(self, X_seqs, y_seqs):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        flat = [f for s in X_seqs for f in s]
        self.dv = DictVectorizer()
        self.lr = LogisticRegression(C=self.C, max_iter=300).fit(self.dv.fit_transform(flat), [t for s in y_seqs for t in s])
        return self

    def predict(self, X_seqs):
        out = []
        for s in X_seqs:
            out.append(list(self.lr.predict(self.dv.transform(s))) if s else [])
        return out


class MajorityTagger:
    """Baseline hafalan: tiap kata diberi tag yang paling sering dilihat; kata baru = 'O'."""

    def fit(self, X_seqs, y_seqs):
        from collections import Counter, defaultdict
        c = defaultdict(Counter)
        for xs, ys in zip(X_seqs, y_seqs):
            for f, t in zip(xs, ys):
                c[f["w"]][t] += 1
        self.map = {w: cc.most_common(1)[0][0] for w, cc in c.items()}
        return self

    def predict(self, X_seqs):
        return [[self.map.get(f["w"], "O") for f in s] for s in X_seqs]


# ================================================================== retrieval
def retrieval_metrics(scores, q_labels, d_labels, k=10):
    """scores: (n_query, n_dokumen). Relevan = label sama. Kembalikan P@k, MAP@k, MRR, nDCG@k."""
    q_labels, d_labels = np.asarray(q_labels), np.asarray(d_labels)
    order = np.argsort(-scores, axis=1)[:, :k]
    rel = (d_labels[order] == q_labels[:, None]).astype(float)
    ranks = np.arange(1, k + 1)
    n_rel = np.minimum((d_labels[None, :] == q_labels[:, None]).sum(1), k)
    ap = ((rel.cumsum(1) / ranks) * rel).sum(1) / np.maximum(n_rel, 1)
    first = np.where(rel.any(1), rel.argmax(1) + 1, np.inf)
    mrr = np.where(np.isinf(first), 0, 1 / first)
    disc = 1 / np.log2(ranks + 1)
    idcg = np.array([disc[:int(n)].sum() for n in n_rel])
    ndcg = (rel * disc).sum(1) / np.maximum(idcg, 1e-9)
    return {f"P@{k}": rel.mean(), f"MAP@{k}": ap.mean(), "MRR": mrr.mean(), f"nDCG@{k}": ndcg.mean()}


# ================================================================== topik
def umass_coherence(topic_words, X_bin, vocab, top_n=10):
    """Koherensi UMass per topik (makin mendekati 0 makin koheren). X_bin: dokumen x kata (biner, sparse), vocab: kata->indeks."""
    X = X_bin.tocsc()
    df = np.asarray(X.sum(0)).ravel()
    out = []
    for words in topic_words:
        idx = [vocab[w] for w in words[:top_n] if w in vocab]
        s = 0.0
        for a in range(1, len(idx)):
            for b in range(a):
                co = X[:, idx[a]].multiply(X[:, idx[b]]).sum()
                s += np.log((co + 1) / max(df[idx[b]], 1))
        out.append(s / max(1, len(idx) * (len(idx) - 1) / 2))
    return out


def purity(true, pred):
    ct = pd.crosstab(pd.Series(pred, name="pred"), pd.Series(true, name="true"))
    return float(ct.max(1).sum() / ct.values.sum())


# ================================================================== TextRank
def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", str(text).strip()) if len(s.strip()) > 2]


def textrank_summary(text, n=3, lang="en"):
    """Ringkasan ekstraktif: kalimat dengan skor PageRank tertinggi pada graf kemiripan kalimat (urutan asli dipertahankan)."""
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    sents = split_sentences(text)
    if len(sents) <= n:
        return sents
    try:
        M = TfidfVectorizer(stop_words=("english" if lang == "en" else ID_STOPWORDS)).fit_transform(sents)
    except ValueError:
        return sents[:n]
    S = (M @ M.T).toarray()
    np.fill_diagonal(S, 0)
    pr = nx.pagerank(nx.from_numpy_array(S), max_iter=200)
    top = sorted(sorted(pr, key=pr.get, reverse=True)[:n])
    return [sents[i] for i in top]


def textrank_keywords(text, k=8, window=4, lang="en"):
    """Kata kunci: PageRank pada graf ko-okurensi kata (jendela `window`)."""
    import networkx as nx
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = ENGLISH_STOP_WORDS if lang == "en" else set(ID_STOPWORDS)
    toks = [w for w in re.findall(r"[A-Za-zÀ-ɏ]{3,}", str(text).lower()) if w not in stop]
    g = nx.Graph()
    for i, w in enumerate(toks):
        for j in range(i + 1, min(i + window, len(toks))):
            if w != toks[j]:
                g.add_edge(w, toks[j])
    if not g.number_of_nodes():
        return []
    pr = nx.pagerank(g)
    return sorted(pr, key=pr.get, reverse=True)[:k]


# ================================================================== pendaftaran
def _add(task, key, recipe, title, imports, code, when, pit, params, requires=(), kind="other"):
    register(modality="text", task=task, key=key, recipe=recipe, title=title, imports=imports, model_code=code,
             kind=kind, when=when, pitfalls=pit, params=params, requires=list(requires))


WORD = 'TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode", stop_words=STOP)'
IMP = "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.multiclass import OneVsRestClassifier"

ML = dict(train_path="data_latihan/text/multilabel/casa_train.csv", test_path="data_latihan/text/multilabel/casa_test.csv",
          text_col="text", targets=["fuel", "machine", "others", "part", "price", "service"], id_col="id", lang="id")
NER = dict(train_path="data_latihan/text/ner/nerp_train.csv", test_path="data_latihan/text/ner/nerp_test.csv")
NEWS = dict(train_path="data_latihan/text/classification_en/news4_train.csv", test_path="data_latihan/text/classification_en/news4_test.csv",
            text_col="text", label_col="label", lang="en")


def _register_all():
    # ---- multilabel
    for key, title, imp, code, when, pit, proba in [
        ("ovr_logreg", "One-vs-Rest Logistic Regression", "from sklearn.linear_model import LogisticRegression",
         "model = OneVsRestClassifier(LogisticRegression(C=5, max_iter=2000, class_weight='balanced'))",
         "Baseline multilabel: satu classifier biner per label. Punya probabilitas sehingga threshold tiap label bisa disetel.",
         "Mengabaikan korelasi antar-label. Label langka butuh class_weight dan threshold sendiri.", True),
        ("ovr_linearsvc", "One-vs-Rest Linear SVM", "from sklearn.svm import LinearSVC",
         "model = OneVsRestClassifier(LinearSVC(C=0.5, class_weight='balanced'))",
         "Sering sedikit lebih akurat dari logreg pada teks; sangat cepat.", "Tanpa probabilitas; threshold diatur lewat skor keputusan.", False),
        ("ovr_complementnb", "One-vs-Rest Complement NB", "from sklearn.naive_bayes import ComplementNB",
         "model = OneVsRestClassifier(ComplementNB(alpha=0.3))",
         "Baseline cepat untuk data kecil dan label tidak seimbang.", "Probabilitas tidak terkalibrasi.", False),
        ("classifier_chain_logreg", "Classifier Chain (Logistic Regression)", "from sklearn.multioutput import ClassifierChain\nfrom sklearn.linear_model import LogisticRegression",
         "model = ClassifierChain(LogisticRegression(C=5, max_iter=2000, class_weight='balanced'), order='random', random_state=42)",
         "Memodelkan ketergantungan antar-label: prediksi label sebelumnya jadi fitur untuk label berikutnya.", "Urutan rantai memengaruhi hasil; kesalahan merambat. Bandingkan dengan OvR sebelum dipercaya.", True),
        ("ovr_lightgbm", "One-vs-Rest LightGBM", "from lightgbm import LGBMClassifier",
         "model = OneVsRestClassifier(LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=15, colsample_bytree=0.3, random_state=42, n_jobs=1, verbose=-1))",
         "Untuk ensemble dengan model linear.", "Data teks kecil: sering kalah dari linear dan lebih lambat.", True),
    ]:
        _add("multilabel", key, "text_multilabel", title, IMP + "\n" + imp,
             f'vec = {WORD}\n{code}', when, pit, dict(ML, has_proba=proba), requires=["lightgbm"] if "lightgbm" in imp else [])
    # ---- NER
    for key, title, imp, code, when, pit, req in [
        ("crf_basic", "CRF (fitur dasar)", "import sklearn_crfsuite",
         'MODE = "basic"\ntagger = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)',
         "Baseline NER klasik: CRF memodelkan transisi antar-tag sehingga urutan BIO konsisten.", "Fitur dasar (kata + tetangga) lemah untuk kata tak dikenal.", ["sklearn_crfsuite"]),
        ("crf_rich", "CRF (fitur kaya: afiks, bentuk, jendela ±2)", "import sklearn_crfsuite",
         'MODE = "rich"\ntagger = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=150, all_possible_transitions=True)',
         "Pilihan utama NER klasik: afiks (imbuhan Indonesia), bentuk huruf, dan konteks ±2 membantu nama yang belum pernah terlihat.", "Fitur berlebih memperlambat latihan; atur c1/c2 (regularisasi) bila overfit.", ["sklearn_crfsuite"]),
        ("logreg_token", "Logistic Regression per token", "from superdb.text_extra import TokenLogReg",
         'MODE = "rich"\ntagger = TokenLogReg(C=3.0)',
         "Pembanding: tanpa memodelkan transisi tag. Menunjukkan seberapa besar CRF diuntungkan oleh struktur urutan.", "Dapat menghasilkan urutan BIO yang tidak valid (I- tanpa B-).", []),
        ("majority_baseline", "Baseline hafalan kata", "from superdb.text_extra import MajorityTagger",
         'MODE = "basic"\ntagger = MajorityTagger()',
         "Batas bawah: jika model Anda hampir tidak mengalahkannya, ada yang salah.", "Kata baru selalu 'O'.", []),
    ]:
        _add("ner", key, "text_ner", title, imp, code, when, pit, dict(NER), requires=req)
    # ---- topic modeling
    for key, title, imp, code, when, pit in [
        ("nmf", "NMF (TF-IDF)", "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import NMF",
         'vec = TfidfVectorizer(min_df=3, max_df=0.5, stop_words=STOP, sublinear_tf=True)\nmodel = NMF(n_components=K, init="nndsvda", random_state=42, max_iter=400)',
         "Topik yang mudah dibaca dan stabil; pilihan pertama untuk topic modeling di teks pendek-menengah.", "Jumlah topik K harus dipilih; periksa topik gado-gado. Hasil bergantung stopword dan max_df."),
        ("lda", "LDA (hitungan kata)", "from sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.decomposition import LatentDirichletAllocation",
         'vec = CountVectorizer(min_df=3, max_df=0.5, stop_words=STOP)\nmodel = LatentDirichletAllocation(n_components=K, learning_method="batch", max_iter=30, random_state=42)',
         "Model probabilistik klasik: tiap dokumen campuran topik. Cocok bila perlu distribusi topik per dokumen.", "Lambat dan peka terhadap hiperparameter dan stopword; topik sering kurang tajam dibanding NMF pada teks pendek."),
        ("lsa", "LSA / TruncatedSVD", "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import TruncatedSVD",
         'vec = TfidfVectorizer(min_df=3, max_df=0.5, stop_words=STOP, sublinear_tf=True)\nmodel = TruncatedSVD(n_components=K, random_state=42)',
         "Cepat; dasar untuk reduksi dimensi sebelum clustering atau klasifikasi.", "Komponen bisa bernilai negatif sehingga 'topik' lebih sulit ditafsirkan."),
    ]:
        _add("topic_modeling", key, "text_topic", title, imp, code, when, pit,
             dict(NEWS, k=4), requires=[])
    # ---- retrieval
    for key, title, imp, code, when, pit, req in [
        ("tfidf_cosine", "TF-IDF + cosine", "from sklearn.feature_extraction.text import TfidfVectorizer",
         'vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=STOP)\nscorer = "tfidf"',
         "Baseline pencarian dokumen mirip; sulit dikalahkan untuk kecocokan kata persis.", "Tidak menangkap sinonim/parafrase.", []),
        ("tfidf_char_cosine", "TF-IDF karakter + cosine", "from sklearn.feature_extraction.text import TfidfVectorizer",
         'vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)\nscorer = "tfidf"',
         "Tahan typo/variasi ejaan dan bahasa kaya imbuhan.", "Indeks lebih besar dan lebih lambat.", []),
        ("bm25", "BM25", "from rank_bm25 import BM25Okapi",
         'scorer = "bm25"   # BM25Okapi dibuat di sel berikutnya',
         "Standar emas pencarian leksikal; biasanya mengungguli TF-IDF cosine pada query pendek.", "Query panjang (dokumen utuh) kurang cocok; tidak menangkap makna.", ["rank_bm25"]),
        ("lsa_cosine", "LSA + cosine", "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import TruncatedSVD\nfrom sklearn.pipeline import make_pipeline",
         'vec = make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True, stop_words=STOP), TruncatedSVD(n_components=150, random_state=42))\nscorer = "lsa"',
         "Menangkap kemiripan topikal dasar (kata berbeda, topik sama).", "Presisi kata persis turun. Jumlah komponen perlu disetel.", []),
    ]:
        _add("retrieval", key, "text_retrieval", title, imp, code, when, pit, dict(NEWS), requires=req)
    # ---- keyword / summary
    for key, title, when, pit, req in [
        ("tfidf_keywords", "Kata kunci TF-IDF", "Kata kunci per dokumen berdasarkan bobot TF-IDF terhadap korpus.", "Butuh korpus pembanding; frasa multikata tidak tertangkap kecuali bigram.", []),
        ("textrank_keywords", "Kata kunci TextRank", "Kata kunci dari satu dokumen saja (tanpa korpus), lewat graf ko-okurensi.", "Hasil kasar pada teks pendek; bergantung stopword.", ["networkx"]),
        ("textrank_summary", "Ringkasan ekstraktif TextRank", "Ringkasan dengan memilih kalimat penting; tanpa model besar, jalan di CPU, bahasa apa pun.", "Hanya memilih kalimat (tidak menulis ulang). Kualitas dinilai manual atau dengan ROUGE terhadap ringkasan referensi bila ada.", ["networkx"]),
        ("lead_baseline", "Baseline ringkasan: kalimat awal", "Pembanding wajib: pada berita, n kalimat pertama sering sudah kuat.", "Hanya cocok untuk teks berstruktur piramida terbalik (berita).", []),
    ]:
        _add("keywords_summary", key, "text_keysum", title, "", "", when, pit, dict(NEWS, doc_idx=[0, 5, 11]), requires=req)
    # ---- clustering teks
    for key, title, imp, code, when, pit, req in [
        ("kmeans_tfidf", "KMeans pada TF-IDF", "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.cluster import KMeans",
         'vec = TfidfVectorizer(min_df=3, max_df=0.5, stop_words=STOP, sublinear_tf=True)\nmodel = KMeans(n_clusters=K, n_init=10, random_state=42)',
         "Baseline clustering dokumen; centroid bisa dibaca sebagai kata khas cluster.", "Dimensi tinggi membuat jarak Euclid kurang bermakna; LSA biasanya lebih baik.", []),
        ("kmeans_lsa", "KMeans pada LSA (ternormalisasi)", "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import TruncatedSVD\nfrom sklearn.preprocessing import Normalizer\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.cluster import KMeans",
         'vec = make_pipeline(TfidfVectorizer(min_df=3, max_df=0.5, stop_words=STOP, sublinear_tf=True), TruncatedSVD(n_components=100, random_state=42), Normalizer(copy=False))\nmodel = KMeans(n_clusters=K, n_init=10, random_state=42)',
         "Resep standar clustering teks: LSA lalu normalisasi sehingga KMeans mendekati cosine.", "K harus dipilih; cluster bisa mengikuti panjang dokumen, bukan topik.", []),
        ("agglomerative_lsa", "Hierarchical (Ward) pada LSA", "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import TruncatedSVD\nfrom sklearn.preprocessing import Normalizer\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.cluster import AgglomerativeClustering",
         'vec = make_pipeline(TfidfVectorizer(min_df=3, max_df=0.5, stop_words=STOP, sublinear_tf=True), TruncatedSVD(n_components=100, random_state=42), Normalizer(copy=False))\nmodel = AgglomerativeClustering(n_clusters=K, linkage="ward")   # ward pada vektor ternormalisasi (cosine+average cenderung runtuh jadi 1 cluster raksasa)',
         "Ingin hierarki topik (dendrogram) tanpa centroid.", "Memori O(n²): hanya beberapa ribu dokumen. Jangan pakai linkage average/single pada teks: hampir selalu menghasilkan satu cluster raksasa.", []),
        ("gmm_lsa", "Gaussian Mixture pada LSA", "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import TruncatedSVD\nfrom sklearn.preprocessing import Normalizer\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.mixture import GaussianMixture",
         'vec = make_pipeline(TfidfVectorizer(min_df=3, max_df=0.5, stop_words=STOP, sublinear_tf=True), TruncatedSVD(n_components=30, random_state=42), Normalizer(copy=False))\nmodel = GaussianMixture(n_components=K, covariance_type="diag", random_state=42)',
         "Soft clustering: tiap dokumen punya probabilitas ke tiap cluster.", "Dimensi harus rendah (SVD kecil); covariance penuh tidak stabil untuk teks.", []),
        ("hdbscan_umap", "UMAP + HDBSCAN (jumlah cluster otomatis)", "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import TruncatedSVD\nfrom sklearn.cluster import HDBSCAN\nimport umap",
         'vec = None   # dikerjakan di sel khusus (LSA -> UMAP)\nmodel = HDBSCAN(min_cluster_size=25)',
         "Jumlah cluster tidak diketahui dan ada dokumen 'noise' yang tidak masuk cluster mana pun.", "Banyak dokumen berlabel -1 (noise) itu normal. Sensitif pada parameter UMAP dan min_cluster_size.", ["umap"]),
    ]:
        _add("clustering", key, "text_cluster", title, imp, code, when, pit, dict(NEWS, k=4), requires=req)
    register_skip("text", "regression", "*", "tidak ada dataset regresi teks publik kecil yang berlisensi jelas; gunakan embedding/TF-IDF + model regresi dari notebooks/tabular (Ridge/LightGBM) dengan fitur teks")


_register_all()
