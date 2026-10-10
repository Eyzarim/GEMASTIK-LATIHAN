"""Fitur klasifikasi time series tambahan: FFT dan tsfresh."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .timeseries import ZNorm


def fft_features(X, k=30):
    """Magnitudo k koefisien FFT pertama dari deret yang dinormalisasi (n, T) -> (n, k)."""
    Z = ZNorm().transform(X)
    return np.abs(np.fft.rfft(Z, axis=1))[:, :k]


class TsfreshFeatures(BaseEstimator, TransformerMixin):
    """Ekstraksi fitur tsfresh (set minimal/efisien) untuk matriks (n, T). Lebih lambat; fitur sangat banyak."""

    def __init__(self, params="minimal"):
        self.params = params

    def fit(self, X, y=None):
        self.cols_ = self._extract(X).columns
        return self

    def _extract(self, X):
        from tsfresh import extract_features
        from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
        X = np.asarray(X, float)
        n, T = X.shape
        long = pd.DataFrame({"id": np.repeat(np.arange(n), T), "t": np.tile(np.arange(T), n), "v": X.ravel()})
        fc = MinimalFCParameters() if self.params == "minimal" else EfficientFCParameters()
        F = extract_features(long, column_id="id", column_sort="t", default_fc_parameters=fc, n_jobs=0, disable_progressbar=True)
        return F.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def transform(self, X):
        return self._extract(X).reindex(columns=self.cols_, fill_value=0.0).to_numpy()


from sklearn.base import ClassifierMixin  # noqa: E402


class RocketTransform(BaseEstimator, TransformerMixin):
    """Pembungkus ROCKET/MiniRocket sktime agar kompatibel dengan Pipeline sklearn (input (n, T))."""

    def __init__(self, kind="minirocket", num_kernels=2000, random_state=42):
        self.kind, self.num_kernels, self.random_state = kind, num_kernels, random_state

    def fit(self, X, y=None):
        from sktime.transformations.panel.rocket import MiniRocket, Rocket
        X3 = np.asarray(X, float)[:, None, :]
        self.t_ = (MiniRocket(random_state=self.random_state) if self.kind == "minirocket"
                   else Rocket(num_kernels=self.num_kernels, random_state=self.random_state)).fit(X3)
        return self

    def transform(self, X):
        return np.asarray(self.t_.transform(np.asarray(X, float)[:, None, :]))


class DTWKNN(BaseEstimator, ClassifierMixin):
    """k-NN dengan jarak DTW (sktime) di balik API sklearn. Input (n, T); tiap deret dinormalisasi z."""

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
        self.clf_ = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors, distance="dtw").fit(ZNorm().transform(X)[:, None, :], np.asarray(y))
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(ZNorm().transform(X)[:, None, :])
