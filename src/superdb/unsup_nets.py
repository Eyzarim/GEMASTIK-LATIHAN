"""Autoencoder PyTorch untuk reduksi dimensi dan deteksi anomali (modul sendiri agar bisa di-pickle)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin


class _AENet(nn.Module):
    def __init__(self, d, latent, hidden):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class TorchAutoencoder(BaseEstimator, TransformerMixin):
    """Autoencoder sederhana (API sklearn). transform() = representasi laten; reconstruction_error() = skor anomali. Data sebaiknya sudah diskalakan."""

    def __init__(self, n_components=2, hidden=64, epochs=200, lr=2e-3, batch_size=128, seed=0):
        self.n_components, self.hidden, self.epochs, self.lr, self.batch_size, self.seed = n_components, hidden, epochs, lr, batch_size, seed

    def fit(self, X, y=None):
        torch.manual_seed(self.seed)
        self.dev_ = "cuda" if torch.cuda.is_available() else "cpu"
        X = torch.tensor(np.asarray(X, np.float32)).to(self.dev_)
        self.net_ = _AENet(X.shape[1], self.n_components, self.hidden).to(self.dev_)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), device=self.dev_)
            for i in range(0, len(X), self.batch_size):
                b = X[perm[i:i + self.batch_size]]
                loss = nn.functional.mse_loss(self.net_(b)[0], b)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def _run(self, X):
        self.net_.eval()
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X, np.float32)).to(self.dev_)
            rec, z = self.net_(Xt)
        return Xt.cpu().numpy(), rec.cpu().numpy(), z.cpu().numpy()

    def transform(self, X):
        return self._run(X)[2]

    def reconstruction_error(self, X):
        x, rec, _ = self._run(X)
        return ((x - rec) ** 2).mean(1)
