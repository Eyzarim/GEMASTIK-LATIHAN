# 17 · Simpan/muat model dan membuat submission yang valid

**Kapan:** akan mengirim hasil, atau perlu memakai model lagi tanpa melatih ulang. **Kata kunci:** submission, simpan model, joblib, pickle, save, load, format, sample_submission, to_csv.

```python
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

tr = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv"); te = pd.read_csv("data_latihan/tabular/classification/titanic_test.csv")
feats = ["pclass", "age", "sibsp", "parch", "fare"]
model = make_pipeline(__import__("sklearn.impute", fromlist=["x"]).SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(tr[feats], tr["survived"])

# 1. Simpan SELURUH pipeline (imputer+scaler+model), bukan model saja
out = Path("outputs/_cookbook"); out.mkdir(parents=True, exist_ok=True)
joblib.dump({"pipe": model, "features": feats, "versi_sklearn": __import__("sklearn").__version__}, out / "model.joblib")
b = joblib.load(out / "model.joblib")
assert (b["pipe"].predict(te[b["features"]]) == model.predict(te[feats])).all()      # uji: hasil muat-ulang HARUS sama

# 2. Buat submission: ID + prediksi, urutan baris sama dengan file test, tanpa index
pred = model.predict(te[feats])
sub = pd.DataFrame({"id": te["id"], "survived": pred})
sub.to_csv(out / "submission.csv", index=False)

# 3. VALIDASI submission sebelum mengirim (lakukan SELALU)
sample = pd.DataFrame({"id": te["id"], "survived": 0})                  # di lomba: pd.read_csv("sample_submission.csv")
assert list(sub.columns) == list(sample.columns), "nama/urutan kolom berbeda dari sample_submission"
assert len(sub) == len(sample), "jumlah baris berbeda"
assert sub["id"].is_unique and set(sub["id"]) == set(sample["id"]), "ID tidak lengkap/duplikat"
assert sub["survived"].notna().all(), "ada NaN"
assert sub["survived"].dtype == sample["survived"].dtype or True          # tipe: cocokkan dengan aturan lomba (int vs float vs string)
print(sub.head(3).to_string(index=False)); print("submission valid:", sub.shape)

# 4. Probabilitas atau label? Baca aturan lomba. Contoh keduanya:
proba = model.predict_proba(te[feats])[:, 1]
pd.DataFrame({"id": te["id"], "survived": proba}).round(6).to_csv(out / "submission_proba.csv", index=False)

# 5. Menyelaraskan urutan dengan sample_submission (jika urutan ID di sample berbeda)
sample_shuffled = sample.sample(frac=1, random_state=0)
aligned = sample_shuffled[["id"]].merge(sub, on="id", how="left"); assert aligned["survived"].notna().all(); print("selaras dengan urutan sample:", aligned.shape)
```

## Jebakan
- `to_csv(..., index=False)` — tanpa itu muncul kolom `Unnamed: 0` yang sering membuat submission ditolak.
- Label kembali ke bentuk asli (`LabelEncoder.inverse_transform`) kalau lomba meminta nama kelas, bukan angka.
- Kirim **model terlatih pada seluruh data** (refit setelah memilih konfigurasi), tetapi catat skor CV-nya.
- Simpan versi pustaka (`pip freeze > requirements_lomba.txt`): `joblib` hasil versi lain bisa gagal dimuat.
- Model PyTorch: simpan `state_dict` (`torch.save(net.state_dict(), "model.pt")`), bukan seluruh objek; muat dengan arsitektur yang sama.
- Simpan submission dengan nama bernomor (`sub_01_lgbm_cv0.812.csv`) agar mudah kembali ke versi terbaik.
