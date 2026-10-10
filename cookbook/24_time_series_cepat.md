# 24 · Time series: split, fitur lag, baseline

**Kapan:** data berurutan waktu dan perlu meramal. **Kata kunci:** time series, deret waktu, forecasting, lag, rolling, TimeSeriesSplit, MASE, musiman, kebocoran waktu.

```python
import numpy as np, pandas as pd
from superdb.timeseries import load_series, forecast_metrics, rolling_origin, recursive_forecast

y = load_series("data_latihan/timeseries/forecasting/co2_monthly.csv", "date", "value", "MS")   # indeks tanggal dengan frekuensi
M, H = 12, 24                                                       # musiman bulanan; horizon ramalan 24 bulan

# 1. SPLIT KRONOLOGIS (jangan acak!): uji = H titik terakhir
train, test = y.iloc[:-H], y.iloc[-H:]

# 2. BASELINE wajib: naive musiman. Model yang tidak mengalahkannya tidak berguna.
snaive = np.tile(train.iloc[-M:].to_numpy(), H // M + 1)[:H]
print("naive musiman:", {k: round(v, 3) for k, v in forecast_metrics(test, snaive, train, M).items()})

# 3. Fitur lag TANPA kebocoran: semua fitur hanya dari masa lalu (shift)
d = pd.DataFrame({"y": y})
for l in (1, 2, 3, 12): d[f"lag{l}"] = d["y"].shift(l)
d["rmean3"] = d["y"].shift(1).rolling(3).mean()                     # shift(1) DULU lalu rolling, jika tidak jendela memuat nilai saat ini
d["bulan"] = d.index.month
print(d.dropna().head(2).round(2).to_string())

# 4. Model ML dengan target selisih musiman (menangani tren + musiman; pohon tidak bisa ekstrapolasi tren)
from lightgbm import LGBMRegressor
fc = recursive_forecast(LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5, verbose=-1, random_state=0), train, H, lags=[1, 2, 3, 6, 12], season=M, mode="seasonal_diff")
print("LightGBM lag:", {k: round(v, 3) for k, v in forecast_metrics(test, fc, train, M).items()})

# 5. Statistik klasik: sering mengalahkan ML pada seri pendek
from statsmodels.tsa.holtwinters import ExponentialSmoothing
hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=M, initialization_method="estimated").fit().forecast(H)
print("Holt-Winters:", {k: round(v, 3) for k, v in forecast_metrics(test, hw, train, M).items()})

# 6. Evaluasi rolling-origin: lebih jujur daripada satu holdout
_, rata = rolling_origin(y, lambda tr, h: recursive_forecast(LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=0), tr, h, lags=[1, 2, 3, 12], season=M), H, n_origins=3, step=12, m=M)
print("rolling-origin LightGBM:", {k: round(v, 3) for k, v in rata.items()})

# 7. Validasi CV untuk model tabular pada data berurutan: TimeSeriesSplit (tidak pernah memakai masa depan untuk melatih)
from sklearn.model_selection import TimeSeriesSplit
dd = d.dropna(); print("TimeSeriesSplit:", [(len(a), len(b)) for a, b in TimeSeriesSplit(3).split(dd)])
```

## Jebakan
- `train_test_split` acak pada data waktu = kebocoran: skor palsu tinggi. Selalu kronologis.
- Rolling/lag tanpa `shift(1)` membocorkan nilai saat ini ke fitur.
- Ramalan jauh dengan model rekursif menumpuk galat; coba strategi *direct* (`direct_forecast`).
- Data lomba biasanya punya tanggal tak lengkap: `asfreq()` lalu interpolasi/isi sesuai makna data.
- Lengkap: `notebooks/timeseries/forecasting/` (22 metode), `forecasting_exog/`, `anomaly/`, `classification_*`, dan `diagnostics/`.
