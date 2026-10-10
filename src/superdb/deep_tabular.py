"""Deep learning untuk data tabular (PyTorch): MLP dengan embedding kategori, ResNet-MLP, FT-Transformer.
Estimator gaya sklearn (fit/predict/predict_proba) menerima DataFrame mentah, jadi bisa dipakai di CV yang sama dengan LightGBM.
Pendaftaran notebook di bawah file ini."""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator

from .registry import register
from .tabular import DEMO


class _Tower(nn.Module):
    def __init__(self, arch, n_num, cards, out_dim, hidden, depth, dropout, d_token, heads, layers):
        super().__init__()
        self.arch, self.n_num = arch, n_num
        if arch == "ftt":
            self.num_w = nn.Parameter(torch.randn(n_num, d_token) * 0.02); self.num_b = nn.Parameter(torch.zeros(n_num, d_token))
            self.embs = nn.ModuleList([nn.Embedding(c + 1, d_token) for c in cards])
            self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
            enc = nn.TransformerEncoderLayer(d_token, heads, dim_feedforward=d_token * 2, dropout=dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, out_dim))
            return
        e_dim = [min(16, (c + 2) // 2) for c in cards]
        self.embs = nn.ModuleList([nn.Embedding(c + 1, d) for c, d in zip(cards, e_dim)])
        d_in = n_num + sum(e_dim)
        if arch == "mlp":
            blocks, d = [], d_in
            for _ in range(depth):
                blocks += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]; d = hidden
            self.body, self.head = nn.Sequential(*blocks), nn.Linear(hidden, out_dim)
        else:                                           # resnet
            self.inp = nn.Linear(d_in, hidden)
            self.blocks = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(hidden), nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.Dropout(dropout)) for _ in range(depth)])
            self.head = nn.Sequential(nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, xn, xc):
        if self.arch == "ftt":
            tok = xn.unsqueeze(-1) * self.num_w + self.num_b                       # (B, n_num, d)
            if len(self.embs):
                tok = torch.cat([tok, torch.stack([e(xc[:, j]) for j, e in enumerate(self.embs)], 1)], 1)
            h = self.enc(torch.cat([self.cls.expand(len(xn), -1, -1), tok], 1))[:, 0]
            return self.head(h)
        z = torch.cat([xn] + [e(xc[:, j]) for j, e in enumerate(self.embs)], 1) if len(self.embs) else xn
        if self.arch == "mlp":
            return self.head(self.body(z))
        h = self.inp(z)
        for b in self.blocks:
            h = h + b(h)
        return self.head(h)


class TabNN(BaseEstimator):
    """arch: 'mlp' | 'resnet' | 'ftt'. task: 'classification' | 'regression'. Memakai GPU bila ada; berhenti dini pada 10% data validasi internal."""

    def __init__(self, arch="mlp", task="classification", num_cols=(), cat_cols=(), hidden=128, depth=3, dropout=0.2, epochs=80,
                 lr=2e-3, wd=1e-4, batch=256, patience=12, seed=42, d_token=32, heads=4, layers=2):
        self.arch, self.task, self.num_cols, self.cat_cols = arch, task, num_cols, cat_cols
        self.hidden, self.depth, self.dropout, self.epochs, self.lr, self.wd = hidden, depth, dropout, epochs, lr, wd
        self.batch, self.patience, self.seed, self.d_token, self.heads, self.layers = batch, patience, seed, d_token, heads, layers

    def __sklearn_tags__(self):
        from sklearn.utils import ClassifierTags, RegressorTags
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier" if self.task == "classification" else "regressor"
        tags.target_tags.required = True
        if self.task == "classification":
            tags.classifier_tags = ClassifierTags()
        else:
            tags.regressor_tags = RegressorTags()
        return tags

    def _prep_fit(self, X):
        Xn = X[list(self.num_cols)].astype(float)
        self.med_ = Xn.median().fillna(0); Xn = Xn.fillna(self.med_)
        self.mu_, self.sd_ = Xn.mean(), Xn.std().replace(0, 1)
        self.maps_ = {c: {v: i + 1 for i, v in enumerate(sorted(X[c].astype(str).dropna().unique()))} for c in self.cat_cols}

    def _tensors(self, X):
        Xn = ((X[list(self.num_cols)].astype(float).fillna(self.med_) - self.mu_) / self.sd_).to_numpy(np.float32)
        Xc = np.column_stack([X[c].astype(str).map(self.maps_[c]).fillna(0).astype(int).to_numpy() for c in self.cat_cols]) if len(self.cat_cols) else np.zeros((len(X), 0), int)
        return torch.tensor(Xn), torch.tensor(Xc, dtype=torch.long)

    def fit(self, X, y):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        X = pd.DataFrame(X).reset_index(drop=True); y = np.asarray(y)
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(np.arange(len(X)), test_size=0.1, random_state=self.seed,
                                  stratify=y if self.task == "classification" else None)
        self._prep_fit(X.iloc[tr])
        if self.task == "classification":
            self.classes_ = np.unique(y); yt = torch.tensor(np.searchsorted(self.classes_, y))
            out = len(self.classes_) if len(self.classes_) > 2 else 1
        else:
            self.ym_, self.ys_ = float(y[tr].mean()), float(y[tr].std() + 1e-9); yt = torch.tensor(((y - self.ym_) / self.ys_).astype(np.float32)); out = 1
        self.binary_ = self.task == "classification" and out == 1
        xn, xc = self._tensors(X)
        cards = [len(self.maps_[c]) for c in self.cat_cols]
        net = _Tower(self.arch, xn.shape[1], cards, out, self.hidden, self.depth, self.dropout, self.d_token, self.heads, self.layers).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        xn, xc, yt = xn.to(dev), xc.to(dev), yt.to(dev)

        def loss_fn(o, t):
            if self.task == "regression":
                return nn.functional.mse_loss(o.squeeze(1), t)
            return nn.functional.binary_cross_entropy_with_logits(o.squeeze(1), t.float()) if self.binary_ else nn.functional.cross_entropy(o, t)
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            net.train(); perm = rng.permutation(tr)
            for i in range(0, len(perm), self.batch):
                b = torch.tensor(perm[i:i + self.batch], device=dev)
                if len(b) < 2:
                    continue
                loss = loss_fn(net(xn[b], xc[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(xn[va], xc[va]), yt[va]).item()
            if v < best - 1e-5:
                best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state); self.net_ = net.cpu(); self.best_epoch_ = ep + 1 - bad
        return self

    def _raw(self, X):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        xn, xc = self._tensors(pd.DataFrame(X).reset_index(drop=True))
        net = self.net_.to(dev).eval()
        with torch.no_grad():
            o = torch.cat([net(xn[i:i+self.batch].to(dev), xc[i:i+self.batch].to(dev)).cpu()
                           for i in range(0, len(xn), self.batch)])
        self.net_ = net.cpu()
        return o

    def predict_proba(self, X):
        o = self._raw(X)
        if self.binary_:
            p = torch.sigmoid(o).numpy(); return np.column_stack([1 - p, p])
        return torch.softmax(o, -1).numpy()

    def predict(self, X):
        if self.task == "regression":
            return self._raw(X).numpy().ravel() * self.ys_ + self.ym_
        return self.classes_[self.predict_proba(X).argmax(1)]


_DEEP = {
    "mlp": ("MLP + embedding kategori", "mlp",
            "model = TabNN(arch='mlp', task=TASK, num_cols=NUM, cat_cols=CAT, hidden=128, depth=3, dropout=0.2, epochs=80)",
            "Titik awal deep learning di tabular: kategori diubah jadi embedding yang dipelajari, numerik distandarkan.",
            "Pada data tabular kecil-menengah umumnya kalah dari boosting. Wajib standarisasi (sudah di dalam). Butuh dropout/weight decay untuk data kecil."),
    "resnet": ("ResNet-MLP", "resnet",
               "model = TabNN(arch='resnet', task=TASK, num_cols=NUM, cat_cols=CAT, hidden=128, depth=3, dropout=0.2, epochs=80)",
               "Varian MLP dengan koneksi residual dan BatchNorm; lebih stabil saat lebih dalam.", "Sama seperti MLP; sensitif learning rate."),
    "ft_transformer": ("FT-Transformer (ringkas)", "ftt",
                       "model = TabNN(arch='ftt', task=TASK, num_cols=NUM, cat_cols=CAT, d_token=32, heads=4, layers=2, dropout=0.1, epochs=80, lr=1e-3)",
                       "Tiap fitur jadi token lalu di-attention-kan; arsitektur deep tabular modern yang kompetitif dengan boosting pada data besar.",
                       "Lebih lambat dan butuh data lebih banyak; pada ribuan baris biasanya belum mengungguli LightGBM."),
}


def _register_all():
    for task, base in (("deep_classification", "classification"), ("deep_regression", "regression")):
        for key, (title, arch, code, when, pit) in _DEEP.items():
            register(modality="tabular", task=task, key=key, recipe="tab_deep", title=title, imports="from superdb.deep_tabular import TabNN", model_code=code,
                     kind="nn", when=when, pitfalls=pit, params=dict(DEMO[base], base_task=base, group_col=None), requires=["torch"], heavy=True)


_register_all()
