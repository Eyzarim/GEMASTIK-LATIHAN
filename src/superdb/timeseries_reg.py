"""Pendaftaran notebook time series: forecasting, forecasting eksogen, klasifikasi, anomali, diagnostik."""
from .registry import register, register_skip

FC = dict(path="data_latihan/timeseries/forecasting/co2_monthly.csv", date_col="date", value_col="value", freq="MS",
          season=12, horizon=36, n_origins=3, step=12, lags=[1, 2, 3, 6, 12, 24], exog=None, title_data="CO2 bulanan (Mauna Loa)")
EXO = dict(path="data_latihan/timeseries/forecasting/macro_quarterly.csv", date_col="date", value_col="realgdp", freq="QS",
           season=4, horizon=8, n_origins=3, step=4, lags=[1, 2, 3, 4], exog=["unemp", "tbilrate", "realinv"], title_data="PDB riil AS kuartalan + eksogen")
IMP = "import numpy as np, pandas as pd\nfrom superdb.timeseries import recursive_forecast, direct_forecast, torch_forecast"

# --------------------------------------------------------------------------- forecasting univariat
# (judul, imports, kode fit_predict, requires, kapan, jebakan)
_UNI = {
    "naive": ("Naive (nilai terakhir)", "", "def fit_predict(train, h):\n    return np.repeat(train.iloc[-1], h)", [],
              "Batas bawah wajib: ramalan = nilai terakhir. Model yang tidak mengalahkannya tidak berguna.", "Mengabaikan musiman dan tren."),
    "seasonal_naive": ("Naive musiman", "", "def fit_predict(train, h):\n    return np.tile(train.iloc[-M:].to_numpy(), h // M + 1)[:h]", [],
                       "Baseline wajib untuk data musiman: ulangi satu siklus terakhir. MASE membandingkan model terhadap ini.", "Meneruskan noise siklus terakhir; tren diabaikan."),
    "drift": ("Drift (garis dari awal ke akhir)", "", "def fit_predict(train, h):\n    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)\n    return train.iloc[-1] + slope * np.arange(1, h + 1)", [],
              "Baseline untuk data bertren tanpa musiman.", "Tidak menangkap musiman."),
    "moving_average": ("Rata-rata bergerak", "", "def fit_predict(train, h):\n    return np.repeat(train.iloc[-M:].mean(), h)", [],
                       "Seri datar berfluktuasi di sekitar rata-rata (tanpa tren).", "Terlambat mengikuti tren; datar ke depan."),
    "ses": ("Exponential smoothing sederhana (SES)", "from statsmodels.tsa.holtwinters import SimpleExpSmoothing",
            "def fit_predict(train, h):\n    return SimpleExpSmoothing(train, initialization_method='estimated').fit().forecast(h).to_numpy()", [],
            "Tanpa tren dan musiman, level berubah perlahan.", "Ramalan datar."),
    "holt_damped": ("Holt (tren teredam)", "from statsmodels.tsa.holtwinters import ExponentialSmoothing",
                    "def fit_predict(train, h):\n    return ExponentialSmoothing(train, trend='add', damped_trend=True, initialization_method='estimated').fit().forecast(h).to_numpy()", [],
                    "Ada tren tapi tidak musiman; redaman mencegah tren menanjak tanpa batas.", "Tanpa musiman."),
    "holt_winters_add": ("Holt-Winters aditif", "from statsmodels.tsa.holtwinters import ExponentialSmoothing",
                         "def fit_predict(train, h):\n    return ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=M, initialization_method='estimated').fit().forecast(h).to_numpy()", [],
                         "Tren + musiman dengan amplitudo musiman konstan. Baseline statistik klasik yang kuat.", "Butuh minimal 2 siklus penuh. Musiman multiplikatif lebih cocok bila amplitudo ikut naik dengan level."),
    "holt_winters_mul": ("Holt-Winters multiplikatif", "from statsmodels.tsa.holtwinters import ExponentialSmoothing",
                         "def fit_predict(train, h):\n    return ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=M, initialization_method='estimated').fit().forecast(h).to_numpy()", [],
                         "Amplitudo musiman membesar seiring level (penjualan, penumpang).", "Data harus positif."),
    "ets_auto": ("ETS dengan pemilihan AIC", "from statsmodels.tsa.exponential_smoothing.ets import ETSModel",
                 "def fit_predict(train, h):\n    best = None\n    for trend in (None, 'add'):\n        for seas in (None, 'add', 'mul'):\n            try:\n                r = ETSModel(train, error='add', trend=trend, seasonal=seas, seasonal_periods=M if seas else None).fit(disp=False)\n            except Exception:\n                continue\n            if best is None or r.aic < best.aic:\n                best = r\n    return best.forecast(h).to_numpy()", [],
                 "Tidak yakin tren/musiman apa yang ada; biarkan AIC memilih di antara beberapa konfigurasi.", "Pemilihan AIC tidak menjamin akurasi ramalan; tetap validasi rolling-origin."),
    "theta": ("Theta", "from statsmodels.tsa.forecasting.theta import ThetaModel",
              "def fit_predict(train, h):\n    return ThetaModel(train, period=M).fit().forecast(h).to_numpy()", [],
              "Metode sederhana yang sangat kuat di kompetisi M3; baseline statistik bagus.", "Kurang fleksibel untuk pola kompleks."),
    "arima": ("ARIMA", "from statsmodels.tsa.arima.model import ARIMA",
              "def fit_predict(train, h):\n    return ARIMA(train, order=(2, 1, 2)).fit().forecast(h).to_numpy()", [],
              "Seri tak musiman atau sudah dimusimankan; urutan (p,d,q) dipilih dari ACF/PACF (lihat notebook diagnostics).", "Mengabaikan musiman. Urutan perlu dipilih; atau pakai auto_arima."),
    "sarima": ("SARIMA", "from statsmodels.tsa.statespace.sarimax import SARIMAX",
               "def fit_predict(train, h):\n    return SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, M)).fit(disp=False).forecast(h).to_numpy()", [],
               "Musiman + tren dengan interval prediksi yang jelas; model klasik standar.", "Lambat untuk musiman panjang (m besar); konvergensi bisa gagal; urutan perlu disetel."),
    "auto_arima": ("auto_arima (pmdarima)", "import pmdarima as pm",
                   "def fit_predict(train, h):\n    m = pm.auto_arima(train, seasonal=True, m=M, stepwise=True, max_p=2, max_q=2, max_P=1, max_Q=1, max_order=5, suppress_warnings=True, error_action='ignore')\n    return m.predict(h)", ["pmdarima"],
                   "Pemilihan otomatis orde ARIMA/SARIMA; hemat waktu saat belum tahu ordenya.", "Lambat pada musiman panjang (ruang pencarian dipersempit di sini; lebarkan max_p/max_q bila waktu ada). Terbaik menurut AIC belum tentu terbaik untuk ramalan."),
    "stl_arima": ("STL + ARIMA", "from statsmodels.tsa.forecasting.stl import STLForecast\nfrom statsmodels.tsa.arima.model import ARIMA",
                  "def fit_predict(train, h):\n    return STLForecast(train, ARIMA, model_kwargs=dict(order=(1, 1, 0), trend='t'), period=M).fit().forecast(h).to_numpy()", [],
                  "Musiman kuat/berubah perlahan: STL memisahkan musiman, ARIMA meramal sisanya.", "Musiman diasumsikan tetap ke depan."),
    "ridge_recursive": ("Ridge + fitur lag (rekursif)", "from sklearn.linear_model import Ridge\n" + IMP,
                        "def fit_predict(train, h):\n    return recursive_forecast(Ridge(alpha=1.0), train, h, lags=LAGS, season=M, mode='seasonal_diff')", [],
                        "Mengubah forecasting jadi regresi tabular dengan fitur lag; titik awal ML yang stabil.", "Galat menumpuk pada ramalan jauh (rekursif). Perhatikan mode target (selisih musiman menangani tren)."),
    "lightgbm_recursive": ("LightGBM + fitur lag (rekursif)", "from lightgbm import LGBMRegressor\n" + IMP,
                           "def fit_predict(train, h):\n    m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1)\n    return recursive_forecast(m, train, h, lags=LAGS, season=M, mode='seasonal_diff')", ["lightgbm"],
                           "Pola non-linear dan banyak fitur tambahan (kalender, eksogen); pilihan utama ML di lomba forecasting.", "Pohon tidak bisa ekstrapolasi tren: wajib memakai mode selisih. Data pendek membuat overfit."),
    "xgboost_recursive": ("XGBoost + fitur lag (rekursif)", "from xgboost import XGBRegressor\n" + IMP,
                          "def fit_predict(train, h):\n    m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)\n    return recursive_forecast(m, train, h, lags=LAGS, season=M, mode='seasonal_diff')", ["xgboost"],
                          "Alternatif LightGBM untuk ensemble.", "Sama seperti LightGBM."),
    "random_forest_recursive": ("Random Forest + fitur lag (rekursif)", "from sklearn.ensemble import RandomForestRegressor\n" + IMP,
                                "def fit_predict(train, h):\n    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)\n    return recursive_forecast(m, train, h, lags=LAGS, season=M, mode='seasonal_diff')", [],
                                "Baseline ML kokoh tanpa banyak tuning.", "Tidak bisa ekstrapolasi."),
    "lightgbm_direct": ("LightGBM (strategi direct)", "from lightgbm import LGBMRegressor\n" + IMP,
                        "def fit_predict(train, h):\n    mk = lambda: LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1)\n    return direct_forecast(mk, train, h, lags=LAGS)", ["lightgbm"],
                        "Horizon jauh: satu model per langkah menghindari galat yang menumpuk pada rekursif.", "h model dilatih (lebih lambat); ramalan antar langkah bisa tidak mulus."),
    "ridge_direct": ("Ridge (strategi direct)", "from sklearn.linear_model import Ridge\n" + IMP,
                     "def fit_predict(train, h):\n    return direct_forecast(lambda: Ridge(alpha=1.0), train, h, lags=LAGS)", [],
                     "Direct paling sederhana dan cepat.", "Linear: tidak menangkap musiman non-linear."),
    "lstm_torch": ("LSTM (PyTorch, multi-output)", "from superdb.timeseries import torch_forecast",
                   "def fit_predict(train, h):\n    return torch_forecast('lstm', train, h, window=WINDOW, epochs=EPOCHS)", ["torch"],
                   "Data panjang/banyak seri dan pola kompleks. Di data kecil biasanya kalah dari model statistik.", "Butuh banyak data; sensitif seed. Selalu bandingkan dengan seasonal naive dan ETS."),
    "gru_torch": ("GRU (PyTorch, multi-output)", "from superdb.timeseries import torch_forecast",
                  "def fit_predict(train, h):\n    return torch_forecast('gru', train, h, window=WINDOW, epochs=EPOCHS)", ["torch"],
                  "Lebih ringan dari LSTM dengan hasil mirip.", "Sama seperti LSTM."),
    "cnn1d_torch": ("CNN 1D (PyTorch, multi-output)", "from superdb.timeseries import torch_forecast",
                    "def fit_predict(train, h):\n    return torch_forecast('cnn1d', train, h, window=WINDOW, epochs=EPOCHS)", ["torch"],
                    "Pola lokal berulang; cepat dilatih.", "Sama seperti LSTM."),
    "mlp_torch": ("MLP (PyTorch, multi-output)", "from superdb.timeseries import torch_forecast",
                  "def fit_predict(train, h):\n    return torch_forecast('mlp', train, h, window=WINDOW, epochs=EPOCHS)", ["torch"],
                  "Pembanding jaringan paling sederhana untuk forecasting.", "Sama seperti LSTM."),
}

# --------------------------------------------------------------------------- forecasting dengan eksogen
_EXO = {
    "sarimax_exog": ("SARIMAX dengan regresor eksogen", "from statsmodels.tsa.statespace.sarimax import SARIMAX",
                     "def fit_predict(train, h, X_train, X_future):\n    return SARIMAX(train, exog=X_train, order=(1, 1, 1)).fit(disp=False).forecast(h, exog=X_future).to_numpy()", [],
                     "Ada variabel penjelas yang nilai masa depannya DIKETAHUI (kalender, harga terencana, kebijakan).", "Nilai eksogen masa depan harus tersedia saat prediksi. Jika harus diramal juga, galatnya menular."),
    "ridge_exog": ("Ridge lag + eksogen (rekursif)", "from sklearn.linear_model import Ridge\n" + IMP,
                   "def fit_predict(train, h, X_train, X_future):\n    dX = np.diff(X_train, axis=0, prepend=X_train[:1]); dXf = np.diff(np.vstack([X_train[-1:], X_future]), axis=0)   # eksogen sebagai selisih (pertumbuhan)\n    return recursive_forecast(Ridge(alpha=1.0), train, h, lags=LAGS, season=M, mode='diff1', exog_train=dX, exog_future=dXf)", [],
                   "Baseline ML dengan fitur eksogen.", "Eksogen masa depan harus diketahui."),
    "lightgbm_exog": ("LightGBM lag + eksogen (rekursif)", "from lightgbm import LGBMRegressor\n" + IMP,
                      "def fit_predict(train, h, X_train, X_future):\n    m = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1)\n    dX = np.diff(X_train, axis=0, prepend=X_train[:1]); dXf = np.diff(np.vstack([X_train[-1:], X_future]), axis=0)   # eksogen sebagai selisih (pertumbuhan)\n    return recursive_forecast(m, train, h, lags=LAGS, season=M, mode='diff1', exog_train=dX, exog_future=dXf)", ["lightgbm"],
                      "Hubungan non-linear antara eksogen dan target.", "Data kuartalan pendek: mudah overfit."),
    "auto_arima_exog": ("auto_arima dengan eksogen", "import pmdarima as pm",
                        "def fit_predict(train, h, X_train, X_future):\n    m = pm.auto_arima(train, X=X_train, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')\n    return m.predict(h, X=X_future)", ["pmdarima"],
                        "ARIMAX dengan orde dipilih otomatis.", "Eksogen masa depan harus diketahui."),
}

# --------------------------------------------------------------------------- klasifikasi time series
IMPC = ("import numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import FunctionTransformer, StandardScaler\n"
        "from superdb.timeseries import ZNorm, ts_stat_features, TorchTSClassifier\nfrom superdb.tsfeatures import fft_features, TsfreshFeatures, RocketTransform, DTWKNN")
_CLS = {
    "knn_euclid": ("1-NN jarak Euclid", "from sklearn.neighbors import KNeighborsClassifier",
                   "model = make_pipeline(ZNorm(), KNeighborsClassifier(n_neighbors=1))", [],
                   "Baseline wajib klasifikasi time series: mirip tetangga terdekat.", "Peka terhadap pergeseran fase; DTW sering lebih baik."),
    "knn_dtw": ("1-NN DTW", "",
                "model = DTWKNN(n_neighbors=1)   # pembungkus KNeighborsTimeSeriesClassifier(distance='dtw') sktime", ["sktime"],
                "Pola sama tetapi bergeser/melar dalam waktu; standar emas klasik.", "Lambat (O(T²) per pasangan) untuk data besar."),
    "stats_lgbm": ("Fitur statistik + LightGBM", "from lightgbm import LGBMClassifier",
                   "model = make_pipeline(FunctionTransformer(ts_stat_features), LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=3, random_state=42, verbose=-1))", ["lightgbm"],
                   "Ringkasan statistik membuat deret panjang jadi tabel kecil yang mudah diinterpretasi (bahan laporan).", "Membuang informasi urutan; kalah dari ROCKET pada pola bentuk halus."),
    "stats_logreg": ("Fitur statistik + Logistic Regression", "from sklearn.linear_model import LogisticRegression",
                     "model = make_pipeline(FunctionTransformer(ts_stat_features), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))", [],
                     "Baseline interpretabel di atas fitur ringkasan.", "Sama seperti di atas."),
    "fft_logreg": ("Spektrum FFT + Logistic Regression", "from sklearn.linear_model import LogisticRegression",
                   "model = make_pipeline(FunctionTransformer(fft_features), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))", [],
                   "Pola berbeda pada frekuensi (getaran, sinyal sensor, EKG).", "Mengabaikan fase; kurang cocok untuk pola yang dibedakan oleh bentuk."),
    "minirocket_ridge": ("MiniRocket + Ridge", "from sklearn.linear_model import RidgeClassifierCV",
                         "model = make_pipeline(RocketTransform('minirocket'), StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))", ["sktime"],
                         "Akurasi tertinggi di klasifikasi time series klasik dengan waktu latih sangat singkat; pilihan pertama setelah baseline.", "Tanpa probabilitas. Hasil acak bila random_state tidak diset."),
    "rocket_ridge": ("ROCKET + Ridge", "from sklearn.linear_model import RidgeClassifierCV",
                     "model = make_pipeline(RocketTransform('rocket', num_kernels=2000), StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))", ["sktime"],
                     "Pendahulu MiniRocket; akurasi tinggi dengan kernel konvolusi acak.", "Lebih lambat dari MiniRocket dengan hasil serupa."),
    "tsfresh_lgbm": ("tsfresh + LightGBM", "from lightgbm import LGBMClassifier",
                     "model = make_pipeline(TsfreshFeatures('minimal'), LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=3, random_state=42, verbose=-1))", ["tsfresh", "lightgbm"],
                     "Ekstraksi fitur otomatis banyak statistik; berguna bila belum tahu fitur apa yang membedakan kelas.", "Lambat untuk data besar; set 'efficient' sangat lambat; fitur banyak berisiko overfit pada data kecil."),
    "random_forest_raw": ("Random Forest pada nilai mentah", "from sklearn.ensemble import RandomForestClassifier",
                          "model = make_pipeline(ZNorm(), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))", [],
                          "Pembanding cepat tanpa rekayasa fitur.", "Tiap titik waktu diperlakukan fitur terpisah; tidak memahami struktur urutan."),
    "cnn1d_torch": ("CNN 1D / FCN (PyTorch)", "from superdb.timeseries import TorchTSClassifier",
                    "model = TorchTSClassifier(arch='cnn', epochs=80, hidden=64, seed=42)", ["torch"],
                    "Data cukup banyak; belajar fitur bentuk sendiri. Kuat bila data ribuan deret.", "Data kecil (puluhan deret) membuatnya tidak stabil; GPU mempercepat."),
    "lstm_torch": ("LSTM (PyTorch)", "from superdb.timeseries import TorchTSClassifier",
                   "model = TorchTSClassifier(arch='lstm', epochs=80, hidden=64, seed=42)", ["torch"],
                   "Pola berurutan panjang; pembanding RNN.", "Biasanya kalah dari ROCKET/CNN di klasifikasi time series."),
}
CLS_DATA = {
    "classification_gunpoint": dict(train="data_latihan/timeseries/classification/gunpoint_train.csv",
                                    test="data_latihan/timeseries/classification/gunpoint_test.csv", name="GunPoint"),
    "classification_italy_power": dict(train="data_latihan/timeseries/classification/italy_power_train.csv",
                                       test="data_latihan/timeseries/classification/italy_power_test.csv", name="ItalyPowerDemand"),
}

# --------------------------------------------------------------------------- anomali
ANOM = dict(path="data_latihan/timeseries/anomaly/nyc_taxi.csv", period=48)
IMPA = "import numpy as np, pandas as pd\nfrom superdb.timeseries import window_matrix, to_point_scores"
_ANO = {
    "zscore_rolling": ("Z-score bergulir", "",
                       "def score(s):\n    mu = s.rolling(P * 7, min_periods=P).mean(); sd = s.rolling(P * 7, min_periods=P).std()\n    return ((s - mu).abs() / sd).fillna(0).to_numpy()", [],
                       "Baseline paling sederhana: seberapa jauh titik dari rata-rata lokal.", "Pola musiman harian dianggap anomali; gagal pada data musiman kuat."),
    "weekly_profile_residual": ("Residual terhadap profil mingguan", "",
                                "def score(s):\n    key = s.index.dayofweek * 48 + s.index.hour * 2 + s.index.minute // 30\n    prof = s.groupby(key).transform('median'); res = s - prof\n    mad = (res - res.median()).abs().median() * 1.4826\n    return (res.abs() / mad).to_numpy()", [],
                                "Data punya pola harian/mingguan tetap: bandingkan terhadap kebiasaan di jam yang sama.", "Profil dari seluruh data termasuk anomali (gunakan median agar robust)."),
    "stl_residual": ("Residual STL (robust)", "from statsmodels.tsa.seasonal import STL",
                     "def score(s):\n    res = STL(s, period=P, robust=True).fit().resid\n    mad = (res - res.median()).abs().median() * 1.4826\n    return (res.abs() / mad).to_numpy()", [],
                     "Musiman kuat: pisahkan tren dan musiman, lalu cari sisa yang aneh.", "Satu periode musiman saja (di sini harian); musiman mingguan masih tersisa."),
    "isolation_forest_window": ("Isolation Forest pada jendela", "from sklearn.ensemble import IsolationForest",
                                "def score(s):\n    Wm = window_matrix(s.to_numpy(), W)\n    m = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1).fit(Wm)\n    return to_point_scores(-m.score_samples(Wm), W, len(s))", [],
                                "Bentuk jendela yang tidak biasa, bukan hanya nilai tunggal.", "Perlu memilih lebar jendela; tidak mengerti urutan antar-jendela."),
    "lof_window": ("LOF pada jendela", "from sklearn.neighbors import LocalOutlierFactor",
                   "def score(s):\n    Wm = window_matrix(s.to_numpy(), W)\n    m = LocalOutlierFactor(n_neighbors=50).fit(Wm)\n    return to_point_scores(-m.negative_outlier_factor_, W, len(s))", [],
                   "Densitas lokal: anomali = jendela yang jauh lebih jarang dari tetangganya.", "In-sample (fit_predict); untuk data baru gunakan novelty=True. Lambat di data besar."),
    "ocsvm_window": ("One-Class SVM pada jendela (PCA)", "from sklearn.svm import OneClassSVM\nfrom sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline",
                     "def score(s):\n    Wm = window_matrix(s.to_numpy(), W)\n    m = make_pipeline(StandardScaler(), PCA(n_components=8, random_state=42), OneClassSVM(nu=0.01, gamma='scale')).fit(Wm)\n    return to_point_scores(-m.decision_function(Wm), W, len(s))", [],
                     "Batas 'normal' non-linear di ruang jendela terkompresi.", "Sensitif terhadap nu dan gamma; lambat untuk data besar."),
    "pca_reconstruction": ("Galat rekonstruksi PCA", "from sklearn.decomposition import PCA",
                           "def score(s):\n    Wm = window_matrix(s.to_numpy(), W); mu = Wm.mean(0)\n    p = PCA(n_components=6, random_state=42).fit(Wm - mu)\n    rec = p.inverse_transform(p.transform(Wm - mu)) + mu\n    return to_point_scores(((Wm - rec) ** 2).mean(1), W, len(s))", [],
                           "Pola normal hidup di subruang berdimensi rendah; anomali sulit direkonstruksi.", "Anomali yang sering muncul ikut tertangkap sebagai 'normal'."),
    "autoencoder_torch": ("Autoencoder jendela (PyTorch)", "import torch, torch.nn as nn",
                          "def score(s):\n    Wm = window_matrix(s.to_numpy(), W); mu, sd = Wm.mean(), Wm.std()\n    X = torch.tensor((Wm - mu) / sd, dtype=torch.float32); dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n    torch.manual_seed(42)\n    net = nn.Sequential(nn.Linear(W, 24), nn.ReLU(), nn.Linear(24, 8), nn.ReLU(), nn.Linear(8, 24), nn.ReLU(), nn.Linear(24, W)).to(dev)\n    opt = torch.optim.Adam(net.parameters(), lr=2e-3); Xd = X.to(dev)\n    for ep in range(40):\n        perm = torch.randperm(len(Xd), device=dev)\n        for i in range(0, len(Xd), 256):\n            b = Xd[perm[i:i + 256]]; loss = nn.functional.mse_loss(net(b), b); opt.zero_grad(); loss.backward(); opt.step()\n    with torch.no_grad():\n        err = ((net(Xd) - Xd) ** 2).mean(1).cpu().numpy()\n    return to_point_scores(err, W, len(s))", ["torch"],
                          "Data besar dan pola normal kompleks; galat rekonstruksi sebagai skor anomali.", "Terlatih juga pada anomali (tanpa label); butuh tuning dan GPU membantu."),
}


def _register_all():
    for key, (title, imp, code, req, when, pit) in _UNI.items():
        register(modality="timeseries", task="forecasting", key=key, recipe="ts_forecast", title=title, imports=imp, model_code=code,
                 when=when, pitfalls=pit, params=dict(FC), requires=req, heavy=key.endswith("_torch"))
    for key, (title, imp, code, req, when, pit) in _EXO.items():
        register(modality="timeseries", task="forecasting_exog", key=key, recipe="ts_forecast", title=title, imports=imp, model_code=code,
                 when=when, pitfalls=pit, params=dict(EXO), requires=req)
    for task, d in CLS_DATA.items():
        for key, (title, imp, code, req, when, pit) in _CLS.items():
            register(modality="timeseries", task=task, key=key, recipe="ts_classify", title=title, imports=IMPC + "\n" + imp, model_code=code,
                     when=when, pitfalls=pit, params=dict(d), requires=req, heavy=key.endswith("_torch"))
    for key, (title, imp, code, req, when, pit) in _ANO.items():
        register(modality="timeseries", task="anomaly", key=key, recipe="ts_anomaly", title=title, imports=IMPA + ("\n" + imp if imp else ""),
                 model_code=code, when=when, pitfalls=pit, params=dict(ANOM), requires=req, heavy=key.endswith("_torch"))
    register(modality="timeseries", task="diagnostics", key="decompose_stationarity", recipe="ts_diag",
             title="Diagnostik: dekomposisi, stasioneritas, ACF/PACF",
             when="Langkah pertama pada data time series baru: lihat tren/musiman, perlu diferensiasi atau tidak, dan orde ARIMA kasar.",
             pitfalls="Uji ADF dan KPSS bisa bertentangan; baca keduanya dan lihat grafik. Jangan menyimpulkan dari satu uji.", params=dict(FC))
    register_skip("timeseries", "forecasting_panel", "*",
                  "data banyak seri (panel/hierarkis) berlisensi jelas dan kecil tidak tersedia offline; gunakan lightgbm_recursive/direct dengan fitur id-seri dan agregat (lihat cookbook)")


_register_all()
