"""Graph: loader, fitur struktur, propagasi, embedding PPMI (ala DeepWalk), label propagation, heuristik link prediction, GCN/GraphSAGE murni PyTorch.
Pendaftaran notebook ada di graph_reg.py."""
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .paths import ROOT


# ================================================================== data
def load_graph(folder, label_col="label", feature_prefix="w"):
    """Muat nodes.csv (id,label,fitur...) dan edges.csv (source,target) -> namespace g (A simetris, A_hat ternormalisasi, X, y, classes)."""
    d = ROOT / folder
    nodes = pd.read_csv(d / "nodes.csv"); edges = pd.read_csv(d / "edges.csv")
    n = len(nodes)
    A = sp.coo_matrix((np.ones(len(edges)), (edges.source, edges.target)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32).tocsr(); A.setdiag(0); A.eliminate_zeros()
    fcols = [c for c in nodes.columns if c.startswith(feature_prefix) and c[len(feature_prefix):].isdigit()]
    X = nodes[fcols].to_numpy(np.float32) if fcols else None
    classes, y = np.unique(nodes[label_col].astype(str), return_inverse=True)
    return SimpleNamespace(A=A, A_hat=normalize_adj(A), X=X, y=y, classes=list(classes), n=n, C=len(classes), nodes=nodes, edges=edges)


def normalize_adj(A, self_loops=True):
    """D^-1/2 (A + I) D^-1/2 (normalisasi simetris seperti pada GCN)."""
    A = A + sp.eye(A.shape[0]) if self_loops else A
    d = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(np.power(d, -0.5, where=d > 0, out=np.zeros_like(d)))
    return (dinv @ A @ dinv).tocsr().astype(np.float32)


def split_per_class(y, per_class=20, seed=0):
    """Pembagian semi-terawasi: `per_class` simpul berlabel per kelas untuk latih, sisanya uji."""
    rng = np.random.default_rng(seed)
    tr = np.concatenate([rng.choice(np.where(y == c)[0], per_class, replace=False) for c in np.unique(y)])
    return np.sort(tr), np.setdiff1d(np.arange(len(y)), tr)


# ================================================================== fitur
def structural_features(A):
    """Fitur struktur per simpul: derajat, koefisien klaster, PageRank, core number, derajat rata-rata tetangga, jumlah segitiga, betweenness (aproksimasi)."""
    import networkx as nx
    G = nx.from_scipy_sparse_array(A)
    deg = np.asarray(A.sum(1)).ravel()
    pr = nx.pagerank(G); cc = nx.clustering(G); core = nx.core_number(G); tri = nx.triangles(G); knn = nx.average_neighbor_degree(G)
    bt = nx.betweenness_centrality(G, k=min(200, G.number_of_nodes()), seed=0)
    f = lambda d: np.array([d[i] for i in range(A.shape[0])])
    return np.column_stack([deg, np.log1p(deg), f(cc), f(pr) * A.shape[0], f(core), f(knn), f(tri), f(bt)])


def propagate(A_hat, X, k=2):
    """Rata-ratakan fitur dengan tetangga k kali (A_hat^k X), inti dari SGC."""
    Z = X
    for _ in range(k):
        Z = A_hat @ Z
    return np.asarray(Z)


def ppmi_embedding(A, dim=64, walk_len=10, n_walks=10, window=5, seed=0):
    """Embedding simpul tanpa pustaka tambahan: random walk -> matriks ko-okurensi -> PPMI -> SVD terpotong (setara dengan DeepWalk secara implisit)."""
    from sklearn.decomposition import TruncatedSVD
    rng = np.random.default_rng(seed)
    n = A.shape[0]; indptr, indices = A.indptr, A.indices
    rows, cols = [], []
    for _ in range(n_walks):
        cur = np.arange(n); walk = [cur]
        for _ in range(walk_len - 1):
            nxt = cur.copy()
            for i, v in enumerate(cur):
                nb = indices[indptr[v]:indptr[v + 1]]
                if len(nb):
                    nxt[i] = nb[rng.integers(len(nb))]
            walk.append(nxt); cur = nxt
        W = np.stack(walk, 1)
        for off in range(1, window + 1):
            rows.append(W[:, :-off].ravel()); cols.append(W[:, off:].ravel())
    r, c = np.concatenate(rows), np.concatenate(cols)
    M = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr(); M = M + M.T
    tot = M.sum(); rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel()
    M = M.tocoo(); pmi = np.log(M.data * tot / (rs[M.row] * cs[M.col] + 1e-12) + 1e-12)
    P = sp.coo_matrix((np.maximum(pmi, 0), (M.row, M.col)), shape=(n, n)).tocsr()
    Z = TruncatedSVD(n_components=min(dim, n - 1), random_state=seed).fit_transform(P)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def label_propagation(A, y, train_idx, alpha=0.9, iters=60):
    """Label propagation (Zhou dkk.): skor kelas menyebar lewat graf; label latih ditanam terus-menerus. Kembalikan skor (n, C)."""
    C = int(y.max()) + 1
    Y0 = np.zeros((A.shape[0], C), np.float32); Y0[train_idx, y[train_idx]] = 1
    S = normalize_adj(A, self_loops=False); F = Y0.copy()
    for _ in range(iters):
        F = alpha * (S @ F) + (1 - alpha) * Y0
    return F / (F.sum(1, keepdims=True) + 1e-9)


# ================================================================== link prediction
def link_split(A, frac=0.1, seed=0):
    """Pisahkan `frac` sisi sebagai uji (positif) + jumlah sama pasangan non-sisi (negatif). Kembalikan (A_train, pos_test, neg_test)."""
    rng = np.random.default_rng(seed)
    U = sp.triu(A, 1).tocoo(); m = len(U.row); k = int(frac * m)
    sel = rng.choice(m, k, replace=False)
    pos = np.column_stack([U.row[sel], U.col[sel]])
    keep = np.ones(m, bool); keep[sel] = False
    n = A.shape[0]
    At = sp.coo_matrix((np.ones(keep.sum()), (U.row[keep], U.col[keep])), shape=(n, n)); At = (At + At.T).tocsr()
    neg = sample_negatives(A, k, seed=seed)
    return At.astype(np.float32), pos, neg


def pair_heuristics(A, pairs):
    """Heuristik link prediction untuk daftar pasangan: common neighbors, Jaccard, Adamic-Adar, Resource Allocation, Preferential Attachment."""
    A = A.tocsr(); deg = np.asarray(A.sum(1)).ravel()
    inv_log = sp.diags(1 / np.log(np.maximum(deg, 2))); inv = sp.diags(1 / np.maximum(deg, 1))
    u, v = pairs[:, 0], pairs[:, 1]
    cn = np.asarray(A[u].multiply(A[v]).sum(1)).ravel()
    aa = np.asarray((A[u] @ inv_log).multiply(A[v]).sum(1)).ravel()
    ra = np.asarray((A[u] @ inv).multiply(A[v]).sum(1)).ravel()
    jac = cn / np.maximum(deg[u] + deg[v] - cn, 1)
    pa = deg[u] * deg[v]
    return pd.DataFrame({"common_neighbors": cn, "jaccard": jac, "adamic_adar": aa, "resource_allocation": ra, "preferential_attachment": pa})


def sample_negatives(A, k, seed=0, avoid=None):
    rng = np.random.default_rng(seed); n = A.shape[0]; out = set()
    forbidden = A if avoid is None else ((A != 0) + (avoid != 0))
    available = n * (n - 1) // 2 - sp.triu(forbidden, 1).nnz
    if k < 0 or k > available:
        raise ValueError(f"Diminta {k} non-sisi unik, tersedia {available}")
    while len(out) < k:
        u, v = rng.integers(n, size=2)
        if u != v and A[u, v] == 0 and (avoid is None or avoid[u, v] == 0):
            out.add((min(u, v), max(u, v)))
    return np.asarray(sorted(out), dtype=int).reshape(-1, 2)


# ================================================================== GNN (PyTorch murni)
def _sparse_to_torch(S, dev):
    import torch
    S = S.tocoo()
    return torch.sparse_coo_tensor(np.vstack([S.row, S.col]), S.data, S.shape, dtype=torch.float32).coalesce().to(dev)


def train_gnn(kind, g, train_idx, val_idx, test_idx, hidden=64, epochs=200, lr=0.01, dropout=0.5, wd=5e-4, seed=0):
    """GCN ('gcn') atau GraphSAGE-mean ('sage'), full-batch, 2 lapisan. Pemilihan epoch memakai val_idx; kembalikan (proba uji, riwayat)."""
    import torch
    import torch.nn as nn
    if (np.intersect1d(train_idx, val_idx).size or np.intersect1d(train_idx, test_idx).size
            or np.intersect1d(val_idx, test_idx).size):
        raise ValueError("Simpul train, validation, dan test harus saling terpisah")
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(g.X).to(dev); X = X / X.sum(1, keepdim=True).clamp(min=1)
    y = torch.tensor(g.y).to(dev)
    S = _sparse_to_torch(g.A_hat if kind == "gcn" else _row_norm(g.A), dev)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            m = 2 if kind == "sage" else 1
            self.l1, self.l2 = nn.Linear(X.shape[1] * m, hidden), nn.Linear(hidden * m, g.C)
            self.dp = nn.Dropout(dropout)

        def layer(self, lin, h):
            h = self.dp(h)
            if kind == "gcn":                                    # A_hat (H W) + b
                return torch.sparse.mm(S, h @ lin.weight.T) + lin.bias
            return lin(torch.cat([h, torch.sparse.mm(S, h)], 1))   # GraphSAGE-mean: [h, rata-rata tetangga] W

        def forward(self, x):
            return self.layer(self.l2, torch.relu(self.layer(self.l1, x)))
    net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    tr, va = torch.tensor(train_idx).to(dev), torch.tensor(val_idx).to(dev)
    best, best_state, hist, t0 = -1, None, [], time.time()
    for ep in range(epochs):
        net.train(); out = net(X); loss = nn.functional.cross_entropy(out[tr], y[tr]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            out = net(X); acc = (out[va].argmax(1) == y[va]).float().mean().item()
        if acc >= best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1:
            hist.append({"epoch": ep + 1, "loss": loss.item(), "val_acc": acc, "detik": round(time.time() - t0, 1)})
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        P = torch.softmax(net(X), -1).cpu().numpy()
    return P[test_idx], hist


def _row_norm(A):
    d = np.asarray(A.sum(1)).ravel()
    return (sp.diags(1 / np.maximum(d, 1)) @ A).tocsr().astype(np.float32)
