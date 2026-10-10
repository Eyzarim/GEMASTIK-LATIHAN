"""Unsupervised: helper (clustering, anomali, reduksi dimensi, seleksi fitur, imputasi, asosiasi).
Pendaftaran notebook ada di unsupervised_reg.py."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ================================================================== clustering
def cluster_scores(Z, labels, y_true=None):
    """Silhouette / Davies-Bouldin / Calinski-Harabasz (tanpa label) dan ARI/NMI (jika label asli ada). Noise (-1) dikecualikan dari skor tanpa label."""
    from sklearn import metrics as m
    labels = np.asarray(labels)
    ok = labels != -1
    k = len(set(labels[ok]))
    out = {"n_cluster": k, "noise_persen": float(100 * (~ok).mean())}
    if k >= 2 and ok.sum() > k:
        out["silhouette"] = float(m.silhouette_score(Z[ok], labels[ok]))
        out["davies_bouldin(kecil=baik)"] = float(m.davies_bouldin_score(Z[ok], labels[ok]))
        out["calinski_harabasz"] = float(m.calinski_harabasz_score(Z[ok], labels[ok]))
    if y_true is not None:
        out["ARI"] = float(m.adjusted_rand_score(y_true, labels))
        out["NMI"] = float(m.normalized_mutual_info_score(y_true, labels))
    return out


def k_sweep(Z, make_model, ks, y_true=None):
    """Sapuan jumlah cluster: inertia (bila ada), silhouette, DB, CH, dan ARI jika ada label."""
    rows = []
    for k in ks:
        mdl = make_model(k)
        lab = mdl.fit_predict(Z) if hasattr(mdl, "fit_predict") else mdl.fit(Z).predict(Z)
        r = {"K": k, **cluster_scores(Z, lab, y_true)}
        if hasattr(mdl, "inertia_"):
            r["inertia"] = float(mdl.inertia_)
        if hasattr(mdl, "bic"):
            r["BIC"] = float(mdl.bic(Z))
        rows.append(r)
    return pd.DataFrame(rows).set_index("K")


def bootstrap_stability(Z, make_model, n=8, frac=0.8, seed=0):
    """Stabilitas cluster: ARI rata-rata antar hasil pada subsampel acak berbeda (dihitung pada baris yang beririsan). Mendekati 1 = stabil."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    N = len(Z)
    runs = []
    for _ in range(n):
        idx = np.sort(rng.choice(N, int(frac * N), replace=False))
        mdl = make_model()
        lab = mdl.fit_predict(Z[idx]) if hasattr(mdl, "fit_predict") else mdl.fit(Z[idx]).predict(Z[idx])
        runs.append((idx, lab))
    ar = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            common, ia, ib = np.intersect1d(runs[i][0], runs[j][0], return_indices=True)
            ar.append(adjusted_rand_score(runs[i][1][ia], runs[j][1][ib]))
    return float(np.mean(ar)), float(np.std(ar))


def cluster_profile(df, labels):
    """Rata-rata tiap fitur per cluster + ukuran cluster (bahan memberi nama cluster di laporan)."""
    d = df.copy()
    d["cluster"] = labels
    prof = d.groupby("cluster").mean(numeric_only=True)
    prof.insert(0, "n", d.groupby("cluster").size())
    return prof


def gower_distance(df, cat_cols=None):
    """Jarak Gower untuk data campuran: numerik = selisih/rentang, kategorikal = 0 jika sama, 1 jika beda. Rata-rata antar fitur."""
    cat_cols = list(cat_cols if cat_cols is not None else df.select_dtypes(exclude="number").columns)
    num_cols = [c for c in df.columns if c not in cat_cols]
    parts = []
    for c in num_cols:
        v = df[c].to_numpy(float)
        rng = np.ptp(v) or 1.0
        parts.append(np.abs(v[:, None] - v[None, :]) / rng)
    for c in cat_cols:
        v = df[c].astype(str).to_numpy()
        parts.append((v[:, None] != v[None, :]).astype(float))
    return np.mean(parts, axis=0)


# ================================================================== anomali
def anomaly_eval(y_true, scores, k=None):
    """ROC-AUC, PR-AUC (utama untuk anomali langka), dan precision@k (k = jumlah anomali sebenarnya)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y_true, scores = np.asarray(y_true), np.asarray(scores, float)
    k = k or int(y_true.sum())
    top = np.argsort(-scores)[:k]
    return {"roc_auc": float(roc_auc_score(y_true, scores)), "pr_auc": float(average_precision_score(y_true, scores)),
            f"precision@{k}": float(y_true[top].mean()), "baseline_pr_auc(proporsi anomali)": float(y_true.mean())}


# ================================================================== reduksi dimensi
def knn_cv_accuracy(Z, y, k=5, cv=5, seed=0):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    return float(cross_val_score(KNeighborsClassifier(k), Z, y, cv=StratifiedKFold(cv, shuffle=True, random_state=seed)).mean())


def reduction_cv_accuracy(X, y, reducer=None, k=5, cv=5, seed=0):
    """Skala dan reduksi dipelajari ulang di training tiap fold, termasuk LDA supervised.

    Reducer tanpa transform (t-SNE, MDS, SpectralEmbedding) hanya boleh dinilai
    sebagai visualisasi; jangan membuat skor prediktif dari embedding seluruh data.
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    if reducer is not None and not hasattr(reducer, "transform"):
        raise ValueError("Reducer tidak mendukung transform data baru; gunakan hanya untuk visualisasi")
    steps = [MinMaxScaler(clip=True)]
    if reducer is not None:
        steps.append(clone(reducer))
    pipe = make_pipeline(*steps, KNeighborsClassifier(k))
    folds = StratifiedKFold(cv, shuffle=True, random_state=seed)
    return float(cross_val_score(pipe, X, y, cv=folds, error_score="raise").mean())


def plot_embedding(Z, y, title="", path=None, ax=None):
    import matplotlib.pyplot as plt
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    codes, uniq = pd.factorize(pd.Series(y))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=codes, cmap="tab10", s=8, alpha=.8)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    if own:
        ax.legend(*sc.legend_elements(), title="kelas", loc="best", fontsize=7)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=130)
    return ax


# ================================================================== imputasi
def mask_mcar(X, frac=0.2, seed=0):
    """Hapus `frac` nilai secara acak (MCAR) dari matriks numerik. Kembalikan (X_ber_NaN, mask)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float).copy()
    mask = rng.random(X.shape) < frac
    X[mask] = np.nan
    return X, mask


def impute_nrmse(X_true, X_imp, mask):
    """RMSE hanya pada sel yang dihapus, dinormalisasi std tiap kolom, lalu dirata-rata (0 = sempurna, ~1 = sama buruk dengan menebak rata-rata)."""
    X_true, X_imp = np.asarray(X_true, float), np.asarray(X_imp, float)
    sd = X_true.std(0) + 1e-12
    err = []
    for j in range(X_true.shape[1]):
        m = mask[:, j]
        if m.any():
            err.append(np.sqrt(np.mean((X_true[m, j] - X_imp[m, j]) ** 2)) / sd[j])
    return float(np.mean(err))


# ================================================================== seleksi fitur
class ShadowSelector(BaseEstimator, TransformerMixin):
    """Seleksi fitur ala Boruta (disederhanakan): bandingkan importance tiap fitur dengan importance salinan acak (shadow) tertinggi.
    Fitur 'menang' bila importance > shadow terbaik; dipilih bila menang di > setengah iterasi."""

    def __init__(self, n_iter=15, random_state=0):
        self.n_iter, self.random_state = n_iter, random_state

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        X = pd.DataFrame(X)
        rng = np.random.default_rng(self.random_state)
        hits = np.zeros(X.shape[1])
        for i in range(self.n_iter):
            sh = X.apply(lambda c: rng.permutation(c.to_numpy()))
            both = np.hstack([X.to_numpy(), sh.to_numpy()])
            rf = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=self.random_state + i).fit(both, y)
            imp = rf.feature_importances_
            hits += imp[:X.shape[1]] > imp[X.shape[1]:].max()
        self.hits_ = hits
        self.support_ = hits > self.n_iter / 2
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


# ================================================================== asosiasi
def load_basket(path):
    """Satu transaksi per baris, item dipisah koma -> list of list."""
    with open(path, encoding="utf-8") as f:
        return [[i.strip() for i in line.strip().split(",") if i.strip()] for line in f if line.strip()]
