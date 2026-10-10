# 22 · Reprodusibilitas: seed, versi, konfigurasi

**Kapan:** hasil berubah tiap dijalankan, atau perlu mengulang hasil terbaik. **Kata kunci:** seed, random_state, reproducible, reprodusibilitas, deterministik, versi, requirements, konfigurasi.

```python
import random, json, platform, sys
import numpy as np, pandas as pd

# 1. Satu fungsi seed untuk semuanya (sudah ada: superdb.common.set_seed)
from superdb.common import set_seed
set_seed(42)

# 2. random_state di SETIAP komponen acak sklearn/boosting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")
X = df[["pclass", "age", "sibsp", "parch", "fare"]].fillna(0); y = df["survived"]
skor = [cross_val_score(RandomForestClassifier(100, random_state=42, n_jobs=-1), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42)).mean() for _ in range(2)]
print("dua kali dijalankan:", [round(s, 6) for s in skor], "-> identik:", skor[0] == skor[1])

# 3. Variasi antar seed = ukuran 'noise' eksperimen. Selisih model < variasi ini tidak bermakna.
seeds = [cross_val_score(RandomForestClassifier(100, random_state=s, n_jobs=-1), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean() for s in range(5)]
print("rata-rata & std antar 5 seed:", round(np.mean(seeds), 4), round(np.std(seeds), 4))

# 4. Catat konfigurasi + versi di samping hasil
import sklearn, lightgbm
meta = {"seed": 42, "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__, "fitur": list(X.columns), "cv_mean": float(np.mean(skor))}
json.dump(meta, open("outputs/_cookbook_config.json", "w"), indent=2); print(json.dumps(meta, indent=1)[:260], "...")

# 5. Deep learning: seed + mode deterministik (lebih lambat)
import torch
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
# torch.use_deterministic_algorithms(True)   # aktifkan bila perlu hasil identik persis di GPU (bisa memerlukan CUBLAS_WORKSPACE_CONFIG=:4096:8)
```

## Jebakan
- `n_jobs=-1` pada beberapa model dapat membuat urutan operasi berbeda; hasil tetap sama untuk RandomForest dengan `random_state`, tetapi GPU dan beberapa boosting bisa berbeda tipis.
- Urutan baris data harus sama (hasil `groupby`/`set` bisa berubah urutannya): `sort_values` eksplisit sebelum split.
- Jalankan ulang dari awal (Restart & Run All) sebelum mengirim; sel yang dijalankan tidak berurutan membuat hasil tidak dapat diulang.
- Simpan `pip freeze > requirements_lomba.txt` saat hasil terbaik didapat.
