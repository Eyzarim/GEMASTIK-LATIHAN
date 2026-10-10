# 02 · Pandas: groupby, pivot, merge, kolom baru

**Kapan:** meringkas per kelompok, menggabungkan tabel, membuat fitur agregat. **Kata kunci:** groupby, agg, pivot_table, merge, join, concat, apply, map, duplicate.

```python
import pandas as pd, numpy as np
df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")

# --- groupby: ringkasan per kelompok
print(df.groupby("pclass")["survived"].mean())                      # rata-rata survived per kelas
print(df.groupby(["pclass", "sex"]).agg(n=("id", "count"), umur=("age", "mean"), tarif=("fare", "median")))

# --- fitur agregat per grup, ditempel balik ke tiap baris (transform)
df["umur_rata_kelas"] = df.groupby("pclass")["age"].transform("mean")
df["fare_vs_kelas"] = df["fare"] / df.groupby("pclass")["fare"].transform("median")

# --- pivot / crosstab
print(pd.pivot_table(df, values="survived", index="pclass", columns="sex", aggfunc="mean"))
print(pd.crosstab(df["pclass"], df["survived"], normalize="index").round(2))

# --- kolom baru
df["keluarga"] = df["sibsp"] + df["parch"] + 1
df["anak"] = (df["age"] < 12).astype(int)
df["kelompok_umur"] = pd.cut(df["age"], [0, 12, 18, 40, 60, 100], labels=["anak", "remaja", "dewasa", "paruh", "lansia"])
df["tarif_kuartil"] = pd.qcut(df["fare"], 4, labels=False, duplicates="drop")
df["sapaan"] = df["sex"].map({"male": "Tn", "female": "Ny"})        # map: kamus nilai -> nilai
df["jenis"] = np.where(df["fare"] > 30, "mahal", "murah")             # if-else vektor

# --- merge (seperti JOIN SQL)
info = pd.DataFrame({"pclass": [1, 2, 3], "nama_kelas": ["Eksekutif", "Bisnis", "Ekonomi"]})
m = df.merge(info, on="pclass", how="left")      # left: semua baris df dipertahankan
print(m[["pclass", "nama_kelas"]].drop_duplicates())
assert len(m) == len(df), "jumlah baris berubah -> kunci merge tidak unik di tabel kanan"

# --- concat: tumpuk baris / sambung kolom
atas, bawah = df.iloc[:5], df.iloc[5:10]
print(pd.concat([atas, bawah], axis=0, ignore_index=True).shape)

# --- duplikat
print("duplikat:", df.duplicated().sum(), " duplikat berdasar kolom:", df.duplicated(subset=["sex", "age", "fare"]).sum())
df = df.drop_duplicates()

# --- urut & ambil teratas
print(df.sort_values("fare", ascending=False).head(3)[["id", "fare"]])
print(df.nlargest(3, "fare")[["id", "fare"]])

# --- apply: fungsi sendiri per baris (LAMBAT, hindari kalau bisa vektor)
df["inisial"] = df["embark_town"].fillna("?").apply(lambda s: s[:1])
```

## Jebakan
- `merge` dengan kunci duplikat di tabel kanan menggandakan baris. Selalu cek `len` sebelum dan sesudah.
- `groupby(...).mean()` mengabaikan NaN; jumlah baris per kelompok bisa berbeda dari jumlah nilai valid.
- **Fitur agregat target (mean target per grup) harus dihitung hanya dari data train** (lihat resep 12 leakage).
- `apply(axis=1)` lambat pada jutaan baris; pakai operasi kolom (`np.where`, `.str`, `.dt`).
