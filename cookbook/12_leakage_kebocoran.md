# 12 · Leakage / kebocoran data

**Kapan:** skor CV sangat tinggi (>0.98) tapi curiga, atau skor CV jauh di atas leaderboard. **Kata kunci:** leakage, bocor, data leakage, target leakage, duplikat, CV tidak cocok leaderboard, terlalu bagus.

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")
y = df["survived"]

# 1. TARGET LEAKAGE: kolom turunan target. Di Titanic, 'alive' (yes/no) = salinan 'survived'.
auc = {c: roc_auc_score(y, pd.factorize(df[c])[0]) for c in df.columns if c not in ("survived", "id") and df[c].nunique() > 1}
auc = pd.Series({c: max(a, 1 - a) for c, a in auc.items()}).sort_values(ascending=False)   # AUC satu fitur (arah diabaikan)
print("AUC satu fitur (curigai yang mendekati 1.0):\n", auc.head(4).round(3))
# aturan: fitur tunggal dengan AUC > ~0.95 hampir selalu bocor, kecuali memang sebab langsung

# 2. Uji tegas: buang kolom bocor, skor harus turun ke tingkat wajar
X = df.drop(columns=["survived", "id"]).select_dtypes("number").fillna(0)
rf = RandomForestClassifier(100, random_state=0, n_jobs=-1)
bocor = X.assign(alive=(df["alive"] == "yes").astype(int))
print("AUC dengan kolom bocor  :", cross_val_score(rf, bocor, y, cv=5, scoring="roc_auc").mean().round(3))
print("AUC tanpa kolom bocor   :", cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean().round(3))

# 3. DUPLIKAT antar train dan test
tr, te = train_test_split(df.drop(columns=["survived", "id"]), test_size=0.3, random_state=0)
dup = tr.merge(te, how="inner").shape[0]
print("baris test yang identik dengan baris train:", dup, "(banyak = split tidak independen)")

# 4. Label shuffle test: acak label, skor HARUS ~ acak (0.5). Kalau tinggi, pipeline Anda bocor.
ys = y.sample(frac=1, random_state=0).to_numpy()
print("AUC label diacak:", cross_val_score(rf, X, ys, cv=5, scoring="roc_auc").mean().round(3), "(harus ~0.5)")
```

## Jenis kebocoran dan cara mencegahnya
| Jenis | Contoh | Pencegahan |
|---|---|---|
| Target leakage | kolom 'alive' untuk memprediksi 'survived'; 'tanggal pelunasan' untuk memprediksi gagal bayar | buang fitur yang tidak tersedia saat prediksi nyata |
| Preprocessing sebelum split | scaler/imputer/encoder dilatih di seluruh data | taruh di `Pipeline` (resep 09, 27) |
| Duplikat/hampir duplikat | baris sama di train dan valid | deduplikasi; `GroupKFold` bila ada grup |
| Grup | pasien/pelanggan yang sama di train dan valid | `GroupKFold` (lihat `notebooks/tabular/validation/group_vs_random_cv`) |
| Waktu | memakai data masa depan untuk memprediksi masa lalu | split kronologis, fitur lag hanya dari masa lalu |
| Target encoding | rata-rata target dihitung dengan baris itu sendiri | cross-fit (`sklearn.preprocessing.TargetEncoder`) |
| Pemilihan fitur/tuning pada seluruh data | seleksi fitur di luar CV | seleksi di dalam fold |

## Jebakan
- Skor yang "terlalu bagus" hampir selalu berarti kebocoran, bukan kecerdasan model.
- Cek **adversarial validation** (`notebooks/tabular/validation/adversarial_validation`) jika test berbeda dari train.
- ID dan urutan baris kadang membocorkan target (data diurutkan menurut kelas): periksa korelasi `id` dengan target.
