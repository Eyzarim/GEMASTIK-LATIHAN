"""Recommender: loader MovieLens, split yang benar (temporal / leave-last-out), metrik rating & ranking, model baseline.
Pendaftaran notebook ada di recsys_reg.py."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .paths import ROOT


# ================================================================== data & split
def load_ml100k(folder="data_latihan/recsys/ml-100k"):
    d = ROOT / folder
    r = pd.read_csv(d / "ratings.csv"); mv = pd.read_csv(d / "movies.csv"); us = pd.read_csv(d / "users.csv")
    # indeks 0..n-1 yang rapat
    r["u"] = r["user"].astype("category").cat.codes; r["i"] = r["item"].astype("category").cat.codes
    umap = dict(zip(r["user"], r["u"])); imap = dict(zip(r["item"], r["i"]))
    mv = mv[mv["item"].isin(imap)].assign(i=lambda d: d["item"].map(imap)).sort_values("i").reset_index(drop=True)
    us = us[us["user"].isin(umap)].assign(u=lambda d: d["user"].map(umap)).sort_values("u").reset_index(drop=True)
    return r, mv, us, int(r.u.max() + 1), int(r.i.max() + 1)


def temporal_split(r, frac_test=0.2):
    """Pemisahan global menurut waktu: rating terbaru sebagai uji (meniru memprediksi masa depan)."""
    cut = r["timestamp"].quantile(1 - frac_test)
    return r[r["timestamp"] <= cut].copy(), r[r["timestamp"] > cut].copy()


def leave_last_out(r):
    """Per pengguna: interaksi TERAKHIR (menurut waktu) jadi uji, sisanya latih. Standar evaluasi top-N."""
    r = r.sort_values(["u", "timestamp", "i"])
    last = r.groupby("u").tail(1)
    return r.drop(last.index), last


def interaction_matrix(df, n_users, n_items, binary=True):
    v = np.ones(len(df), np.float32) if binary else df["rating"].to_numpy(np.float32)
    return sp.csr_matrix((v, (df["u"], df["i"])), shape=(n_users, n_items))


# ================================================================== metrik
def rating_metrics(y, p, lo=1, hi=5):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), lo, hi)
    return {"rmse": float(np.sqrt(np.mean((y - p) ** 2))), "mae": float(np.mean(np.abs(y - p)))}


def rank_metrics(scores, train_csr, test_items, k=10):
    """scores: (n_user, n_item) skor tiap item; item yang sudah dilihat di train dikecualikan. test_items: (n_user,) satu item uji tiap pengguna.
    Kembalikan HR@k, NDCG@k, MRR, serta cakupan katalog (variasi rekomendasi)."""
    S = np.array(scores, dtype=np.float32, copy=True)
    if not 1 <= k <= S.shape[1]:
        raise ValueError("k harus di antara 1 dan jumlah item")
    S[train_csr.nonzero()] = -np.inf
    test_items = np.asarray(test_items)
    pos = S[np.arange(len(S)), test_items]
    if not np.isfinite(pos).all():
        raise ValueError("Item uji harus belum dilihat di train dan memiliki skor finite")
    # Ties diselesaikan dengan ID item naik, sama pada rank target dan top-k.
    # Menghitung hanya S > pos memberi rank 0 untuk semua kandidat dengan skor sama.
    item_ids = np.arange(S.shape[1])
    rank = ((S > pos[:, None]) | ((S == pos[:, None]) & (item_ids < test_items[:, None]))).sum(1)
    top = np.argsort(-S, axis=1, kind="stable")[:, :k]
    hit = rank < k
    return {f"HR@{k}": float(hit.mean()), f"NDCG@{k}": float(np.mean(np.where(hit, 1 / np.log2(rank + 2), 0))),
            "MRR": float(np.mean(1 / (rank + 1))), "coverage_katalog": float(len(np.unique(top)) / S.shape[1]), "peringkat_median": float(np.median(rank + 1))}


# ================================================================== model
class BiasModel:
    """Baseline bias: rating = mu + b_user + b_item, dengan regularisasi (hasil terbaik di antara baseline sederhana)."""

    def __init__(self, reg_u=10.0, reg_i=25.0, iters=10):
        self.reg_u, self.reg_i, self.iters = reg_u, reg_i, iters

    def fit(self, df, n_users, n_items):
        self.mu = df["rating"].mean(); self.bu = np.zeros(n_users); self.bi = np.zeros(n_items)
        u, i, r = df["u"].to_numpy(), df["i"].to_numpy(), df["rating"].to_numpy(float)
        for _ in range(self.iters):
            self.bi = np.bincount(i, weights=r - self.mu - self.bu[u], minlength=n_items) / (np.bincount(i, minlength=n_items) + self.reg_i)
            self.bu = np.bincount(u, weights=r - self.mu - self.bi[i], minlength=n_users) / (np.bincount(u, minlength=n_users) + self.reg_u)
        return self

    def predict(self, df):
        return self.mu + self.bu[df["u"].to_numpy()] + self.bi[df["i"].to_numpy()]


def item_knn_scores(R, k=50, shrink=10.0):
    """Skor item-KNN: S = cosine antar-item (dipotong k tetangga), skor = R @ S. R: pengguna x item (biner)."""
    R = R.tocsc().astype(np.float32)
    co = (R.T @ R).toarray()
    norm = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel()) + 1e-9
    S = co / (np.outer(norm, norm) + shrink)
    np.fill_diagonal(S, 0)
    idx = np.argpartition(-S, k, axis=1)[:, k:]
    np.put_along_axis(S, idx, 0, axis=1)
    return np.asarray(R.tocsr() @ S)


def user_knn_scores(R, k=50, shrink=10.0):
    R = R.tocsr().astype(np.float32)
    co = (R @ R.T).toarray()
    norm = np.sqrt(np.asarray(R.multiply(R).sum(1)).ravel()) + 1e-9
    S = co / (np.outer(norm, norm) + shrink)
    np.fill_diagonal(S, 0)
    idx = np.argpartition(-S, k, axis=1)[:, k:]
    np.put_along_axis(S, idx, 0, axis=1)
    return S @ R.toarray()


def surprise_fit_predict(algo, train, test):
    """Latih algoritma scikit-surprise pada df train (kolom u,i,rating) dan prediksi baris df test."""
    from surprise import Dataset, Reader
    ds = Dataset.load_from_df(train[["u", "i", "rating"]], Reader(rating_scale=(1, 5)))
    algo.fit(ds.build_full_trainset())
    return np.array([algo.predict(int(u), int(i)).est for u, i in zip(test["u"], test["i"])])


def torch_mf(train, test, n_users, n_items, dim=32, epochs=40, lr=0.01, wd=1e-4, seed=42):
    """Faktorisasi matriks eksplisit dengan bias di PyTorch (GPU). Kembalikan prediksi untuk baris test."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu = float(train["rating"].mean())

    class MF(nn.Module):
        def __init__(self):
            super().__init__()
            self.P, self.Q = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
            self.bu, self.bi = nn.Embedding(n_users, 1), nn.Embedding(n_items, 1)
            for e in (self.P, self.Q):
                nn.init.normal_(e.weight, 0, 0.05)
            for e in (self.bu, self.bi):
                nn.init.zeros_(e.weight)

        def forward(self, u, i):
            return mu + self.bu(u).squeeze(1) + self.bi(i).squeeze(1) + (self.P(u) * self.Q(i)).sum(1)
    net = MF().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    u, i = torch.tensor(train.u.values).long().to(dev), torch.tensor(train.i.values).long().to(dev); r = torch.tensor(train.rating.values).float().to(dev)
    for ep in range(epochs):
        perm = torch.randperm(len(u), device=dev)
        for s in range(0, len(u), 4096):
            b = perm[s:s + 4096]
            loss = nn.functional.mse_loss(net(u[b], i[b]), r[b]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(test.u.values).long().to(dev), torch.tensor(test.i.values).long().to(dev)).cpu().numpy()
