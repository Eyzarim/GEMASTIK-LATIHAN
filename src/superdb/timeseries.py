"""Time series: helper umum (forecasting, klasifikasi, anomali). Pendaftaran notebook ada di timeseries_reg.py.

Prinsip: split SELALU kronologis; evaluasi dengan rolling-origin; fitur lag hanya memakai masa lalu (tanpa kebocoran).
"""
import numpy as np
import pandas as pd


# ================================================================== data & metrik
def load_series(path, date_col="date", value_col="value", freq="MS"):
    df = pd.read_csv(path, parse_dates=[date_col]).sort_values(date_col)
    s = df.set_index(date_col)[value_col].astype(float)
    s.index = pd.DatetimeIndex(s.index)
    s = s.asfreq(freq)
    if s.isna().any():
        print(f"[peringatan] {int(s.isna().sum())} titik hilang setelah asfreq('{freq}'); diinterpolasi linear")
        s = s.interpolate()
    return s


def forecast_metrics(y_true, y_pred, y_train=None, m=1):
    """MAE, RMSE, MAPE, sMAPE, dan MASE (skala = galat naive musiman pada training; <1 berarti lebih baik dari naive musiman)."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = yt - yp
    out = {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2)))}
    nz = yt != 0
    out["mape"] = float(np.mean(np.abs(e[nz] / yt[nz]))) if nz.any() else float("nan")
    out["smape"] = float(np.mean(2 * np.abs(e) / (np.abs(yt) + np.abs(yp) + 1e-12)))
    if y_train is not None and len(y_train) > m:
        yt_ = np.asarray(y_train, float)
        scale = np.mean(np.abs(yt_[m:] - yt_[:-m]))
        out["mase"] = float(np.mean(np.abs(e)) / scale) if scale > 0 else float("nan")
    return out


def rolling_origin(series, fit_predict, h, n_origins=3, step=None, m=1, **kw):
    """Evaluasi rolling-origin (expanding window): latih sampai titik asal, ramal h langkah, geser asal sebesar `step`."""
    step = step or h
    rows = []
    for i in range(n_origins):
        cut = len(series) - h - step * (n_origins - 1 - i)
        if cut < 2 * m + 5:
            continue
        tr, te = series.iloc[:cut], series.iloc[cut:cut + h]
        pred = np.asarray(fit_predict(tr, h, **kw), float)
        rows.append({"asal_akhir_latih": str(tr.index[-1].date()), **forecast_metrics(te.to_numpy(), pred, tr.to_numpy(), m)})
    df = pd.DataFrame(rows)
    return df, df.drop(columns="asal_akhir_latih").mean().to_dict()


def future_index(series, h):
    off = series.index.freq or pd.infer_freq(series.index)
    return pd.date_range(series.index[-1], periods=h + 1, freq=off)[1:]


def plot_forecast(train, test, pred, lo=None, hi=None, title="Forecast", last=None, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.8))
    tr = train.iloc[-last:] if last else train
    ax.plot(tr.index, tr.values, color="0.4", label="latih")
    if test is not None:
        ax.plot(test.index, test.values, "k", label="aktual")
    idx = test.index if test is not None else None
    if idx is not None and pred is not None:
        ax.plot(idx, pred, "r--", label="ramalan")
        if lo is not None:
            ax.fill_between(idx, lo, hi, color="r", alpha=.15)
    ax.set_title(title); ax.legend(); fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


# ================================================================== fitur lag (ML)
def _row(zs, date, lags, windows, exog=None):
    f = [zs[-l] for l in lags]
    for w in windows:
        seg = np.asarray(zs[-w:], float)
        f += [seg.mean(), seg.std()]
    f += [date.month, date.quarter, date.dayofweek, date.hour]
    if exog is not None:
        f += list(exog)
    return f


def _names(lags, windows, exog_cols=()):
    n = [f"lag{l}" for l in lags]
    for w in windows:
        n += [f"rmean{w}", f"rstd{w}"]
    return n + ["month", "quarter", "dow", "hour"] + list(exog_cols)


def recursive_forecast(model, train, h, lags=(1, 2, 3, 12), season=12, mode="seasonal_diff", windows=(3,),
                       exog_train=None, exog_future=None):
    """Peramal rekursif: model dilatih memprediksi 1 langkah; prediksi dipakai sebagai input langkah berikutnya.
    mode: 'level' (nilai asli), 'diff1' (selisih langkah), 'seasonal_diff' (selisih musiman: menangani tren+musiman, disarankan untuk pohon)."""
    from sklearn.base import clone
    y = train.to_numpy(float)
    d0 = {"level": 0, "diff1": 1, "seasonal_diff": season}[mode]
    z = np.full(len(y), np.nan)
    if mode == "level":
        z = y.copy()
    elif mode == "diff1":
        z[1:] = np.diff(y)
    else:
        z[season:] = y[season:] - y[:-season]
    mx = max(lags)
    t0 = d0 + mx
    X, T = [], []
    for t in range(t0, len(y)):
        ex = None if exog_train is None else np.asarray(exog_train)[t]
        X.append(_row(list(z[:t]), train.index[t], lags, windows, ex)); T.append(z[t])
    m = clone(model).fit(np.asarray(X), np.asarray(T))
    ys, zs, out = list(y), list(z), []
    fidx = future_index(train, h)
    for k in range(h):
        ex = None if exog_future is None else np.asarray(exog_future)[k]
        zh = float(m.predict(np.asarray([_row(zs, fidx[k], lags, windows, ex)]))[0])
        yn = zh if mode == "level" else (ys[-1] + zh if mode == "diff1" else ys[-season] + zh)
        ys.append(yn); zs.append(zh); out.append(yn)
    return np.asarray(out)


def direct_forecast(make_model, train, h, lags=(1, 2, 3, 12), windows=(3,), exog_train=None, exog_future=None):
    """Strategi direct: satu model per langkah ramalan k=1..h. Target = y[t+k] - y[t] (selisih terhadap nilai terakhir), jadi tren tidak jadi masalah."""
    y = train.to_numpy(float)
    mx = max(max(lags), max(windows))
    out = []
    last = y[-1]
    base_feat = lambda yy, t, date: [yy[t] - yy[t - l] for l in lags] + [yy[t - w + 1:t + 1].mean() - yy[t] for w in windows]
    for k in range(1, h + 1):
        X, T = [], []
        for t in range(mx, len(y) - k):
            tgt_date = train.index[t + k]
            X.append(base_feat(y, t, train.index[t]) + [train.index[t].month, tgt_date.month, tgt_date.dayofweek])
            T.append(y[t + k] - y[t])
        m = make_model().fit(np.asarray(X), np.asarray(T))
        fd = future_index(train, h)[k - 1]
        xr = base_feat(y, len(y) - 1, train.index[-1]) + [train.index[-1].month, fd.month, fd.dayofweek]
        out.append(last + float(m.predict(np.asarray([xr]))[0]))
    return np.asarray(out)


# ================================================================== peramal neural (PyTorch)
def torch_forecast(kind, train, h, window=24, epochs=150, hidden=64, lr=2e-3, batch=64, seed=42):
    """Peramal multi-output langsung: jendela masa lalu -> h titik ke depan. Normalisasi relatif terhadap nilai terakhir jendela
    (menangani tren). kind: 'lstm' | 'gru' | 'cnn1d' | 'mlp'. Memakai GPU bila tersedia."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    v = train.to_numpy(float)
    mu, sd = v.mean(), v.std() + 1e-9
    z = (v - mu) / sd
    X, Y = [], []
    for i in range(window, len(z) - h + 1):
        base = z[i - 1]
        X.append(z[i - window:i] - base); Y.append(z[i:i + h] - base)
    X, Y = torch.tensor(np.asarray(X), dtype=torch.float32), torch.tensor(np.asarray(Y), dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind in ("lstm", "gru"):
                self.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(1, hidden, batch_first=True)
                self.head = nn.Linear(hidden, h)
            elif kind == "cnn1d":
                self.conv = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
                self.head = nn.Linear(32 * window, h)
            else:
                self.mlp = nn.Sequential(nn.Linear(window, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, h))

        def forward(self, x):
            if kind in ("lstm", "gru"):
                o, _ = self.rnn(x.unsqueeze(-1)); return self.head(o[:, -1])
            if kind == "cnn1d":
                return self.head(self.conv(x.unsqueeze(1)).flatten(1))
            return self.mlp(x)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xd, Yd = X.to(dev), Y.to(dev)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xd), device=dev)
        for i in range(0, len(Xd), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.mse_loss(net(Xd[idx]), Yd[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        last = z[-window:]
        base = last[-1]
        yh = net(torch.tensor(last - base, dtype=torch.float32, device=dev).unsqueeze(0)).cpu().numpy().ravel() + base
    return yh * sd + mu


# ================================================================== klasifikasi time series
def ts_stat_features(X):
    """Fitur ringkasan tiap deret (n, T) -> (n, d): statistik, bentuk, autokorelasi, spektrum."""
    from scipy.stats import kurtosis, skew
    X = np.asarray(X, float)
    t = np.arange(X.shape[1])
    feats = []
    for x in X:
        d = np.diff(x)
        fft = np.abs(np.fft.rfft(x - x.mean()))[:6]
        feats.append([x.mean(), x.std(), x.min(), x.max(), np.median(x), np.percentile(x, 25), np.percentile(x, 75),
                      skew(x), kurtosis(x), np.mean(np.abs(d)), d.std(), np.sum(x ** 2) / len(x),
                      np.polyfit(t, x, 1)[0], np.argmax(x) / len(x), np.argmin(x) / len(x),
                      np.corrcoef(x[:-1], x[1:])[0, 1] if x.std() > 0 else 0.0,
                      int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))), *fft])
    return np.nan_to_num(np.asarray(feats))


from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin  # noqa: E402


class To3D(BaseEstimator, TransformerMixin):
    """(n, T) -> (n, 1, T), format yang diminta transformer sktime (Rocket, MiniRocket)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, None, :]


class ZNorm(BaseEstimator, TransformerMixin):
    """Normalisasi z tiap deret secara terpisah (umum untuk klasifikasi time series)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)


class TorchTSClassifier(BaseEstimator, ClassifierMixin):
    """Klasifier time series PyTorch dengan API sklearn. arch: 'cnn' (FCN: 3 blok Conv1D + global average pooling) atau 'lstm'."""

    def __init__(self, arch="cnn", epochs=80, lr=1e-3, batch_size=32, hidden=64, seed=42):
        self.arch, self.epochs, self.lr, self.batch_size, self.hidden, self.seed = arch, epochs, lr, batch_size, hidden, seed

    def _net(self, n_classes):
        import torch.nn as nn
        h = self.hidden
        if self.arch == "lstm":
            from .tsnet import LSTMNet
            return LSTMNet(h, n_classes)
        return nn.Sequential(nn.Conv1d(1, h, 8, padding=4), nn.BatchNorm1d(h), nn.ReLU(),
                             nn.Conv1d(h, 2 * h, 5, padding=2), nn.BatchNorm1d(2 * h), nn.ReLU(),
                             nn.Conv1d(2 * h, h, 3, padding=1), nn.BatchNorm1d(h), nn.ReLU(),
                             nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(h, n_classes))

    def fit(self, X, y):
        import torch
        from sklearn.preprocessing import LabelEncoder
        torch.manual_seed(self.seed)
        self.dev_ = "cuda" if torch.cuda.is_available() else "cpu"
        self.le_ = LabelEncoder().fit(y); self.classes_ = self.le_.classes_
        Xn = torch.tensor(ZNorm().transform(X), dtype=torch.float32).unsqueeze(1).to(self.dev_)
        yt = torch.tensor(self.le_.transform(y)).to(self.dev_)
        self.net_ = self._net(len(self.classes_)).to(self.dev_)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        for ep in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(Xn), device=self.dev_)
            for i in range(0, len(Xn), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:
                    continue
                loss = torch.nn.functional.cross_entropy(self.net_(Xn[idx]), yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        import torch
        self.net_.eval()
        Xn = torch.tensor(ZNorm().transform(X), dtype=torch.float32).unsqueeze(1).to(self.dev_)
        with torch.no_grad():
            return torch.softmax(self.net_(Xn), -1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ================================================================== anomali
def window_matrix(x, w):
    """Matriks jendela geser (n-w+1, w)."""
    x = np.asarray(x, float)
    idx = np.arange(w)[None, :] + np.arange(len(x) - w + 1)[:, None]
    return x[idx]


def to_point_scores(win_scores, w, n):
    """Skor per jendela -> skor per titik (diberikan ke titik terakhir jendela; titik awal diisi skor minimum)."""
    out = np.full(n, np.nanmin(win_scores))
    out[w - 1:] = win_scores
    return out


def event_metrics(scores, labels, windows, q=0.995):
    """Evaluasi anomali berbasis jendela label: ambang = kuantil q dari skor.
    - recall_jendela: berapa jendela anomali yang memuat >= 1 alarm.
    - alarm_di_luar: jumlah KLASTER alarm di luar semua jendela (false alarm, titik berurutan digabung).
    - pr_auc_titik: PR-AUC per titik terhadap label jendela (kasar karena jendela panjang)."""
    from sklearn.metrics import average_precision_score
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    thr = np.quantile(scores, q)
    alarm = scores >= thr
    det = [bool(alarm[a:b + 1].any()) for a, b in windows]
    outside = alarm & (labels == 0)
    clusters = int(np.sum(outside & ~np.r_[False, outside[:-1]]))
    return {"recall_jendela": float(np.mean(det)), "jendela_terdeteksi": int(sum(det)), "jumlah_jendela": len(windows),
            "alarm_klaster_di_luar": clusters, "pr_auc_titik": float(average_precision_score(labels, scores)), "ambang": float(thr)}


def label_windows(labels):
    """Indeks (awal, akhir) tiap rentetan label 1."""
    lab = np.asarray(labels).astype(int)
    d = np.diff(np.r_[0, lab, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))
