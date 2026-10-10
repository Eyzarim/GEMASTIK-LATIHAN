# 01 · Pandas: baca, tulis, lihat data

**Kapan:** langkah pertama setiap soal. **Kata kunci:** read_csv, excel, encoding, separator, kolom, dtype, head, info.

```python
import pandas as pd
df = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv")

print(df.shape)            # (baris, kolom)
print(df.dtypes)           # tipe tiap kolom
df.info()                  # tipe + jumlah non-null + memori
print(df.head(3))          # 3 baris pertama
print(df.sample(3, random_state=0))   # 3 baris acak
print(df.columns.tolist())
```

## Variasi baca file

```python
# separator titik-koma (CSV Eropa/Indonesia sering begitu) + desimal koma
# df = pd.read_csv("file.csv", sep=";", decimal=",")

# error 'UnicodeDecodeError' -> coba encoding lain
# df = pd.read_csv("file.csv", encoding="latin-1")      # atau "utf-8-sig", "cp1252"

# Excel (butuh openpyxl, sudah terpasang)
# df = pd.read_excel("file.xlsx", sheet_name="Sheet1")

# JSON / parquet
# df = pd.read_json("file.json"); df = pd.read_parquet("file.parquet")

# hanya kolom tertentu / sebagian baris (hemat memori, lihat resep 19)
part = pd.read_csv("data_latihan/tabular/classification/titanic_train.csv", usecols=["survived", "sex", "age"], nrows=100)
print(part.shape)

# tanggal langsung diparse, ID tetap string (jaga nol di depan)
# df = pd.read_csv("file.csv", parse_dates=["tanggal"], dtype={"id": str})

# nilai kosong yang ditulis aneh ('-', 'NA', '?') -> NaN
# df = pd.read_csv("file.csv", na_values=["-", "?", "N/A", "tidak ada"])
```

## Memilih baris & kolom

```python
print(df["age"].head(3))                       # satu kolom -> Series
print(df[["sex", "age"]].head(3))              # beberapa kolom -> DataFrame
print(df[df["age"] > 60].shape)                # filter baris
print(df[(df["sex"] == "female") & (df["pclass"] == 1)].shape)   # gabung syarat dengan & | dan kurung
print(df.query("age > 60 and fare > 50").shape)
print(df.loc[0:2, ["sex", "age"]])             # berdasar label
print(df.iloc[0:2, 0:3])                       # berdasar posisi
print(df[df["embarked"].isin(["S", "C"])].shape)
```

## Menulis

```python
df.head(5).to_csv("outputs/contoh_tulis.csv", index=False)   # index=False: jangan tulis nomor baris
```

## Jebakan
- Lupa `index=False` → submission punya kolom tambahan `Unnamed: 0` dan ditolak.
- ID dibaca sebagai angka lalu nol depan hilang (`007` → `7`): pakai `dtype={"id": str}`.
- `df[df.a > 1 & df.b < 2]` salah; harus `df[(df.a > 1) & (df.b < 2)]`.
