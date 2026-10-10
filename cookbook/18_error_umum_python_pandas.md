# 18 · Error umum Python / pandas: pesan → penyebab → perbaikan

**Kapan:** kode berhenti dengan error. **Kata kunci:** error, KeyError, ValueError, SettingWithCopyWarning, UnicodeDecodeError, UnicodeEncodeError, MemoryError, could not convert string to float, ModuleNotFoundError, FileNotFoundError.

```python
import numpy as np, pandas as pd

def tampil(judul, fn):
    try: fn(); print("OK        ", judul)
    except Exception as ex: print(f"{type(ex).__name__:18s}", judul, "->", str(ex)[:70])

df = pd.DataFrame({"nama": ["a", "b", "c"], "harga": ["1.000", "2.500", "x"], "qty": [1, 2, 3]})

# KeyError: 'Harga'  -> nama kolom salah huruf besar/spasi. Periksa df.columns; bersihkan: df.columns = df.columns.str.strip().str.lower()
tampil("KeyError (nama kolom salah)", lambda: df["Harga"])

# ValueError: could not convert string to float: 'x' -> kolom angka berisi teks. Gunakan to_numeric(errors="coerce") lalu lihat yang NaN
tampil("ValueError string->float", lambda: df["harga"].astype(float))
angka = pd.to_numeric(df["harga"].str.replace(".", "", regex=False), errors="coerce"); print("   perbaikan:", angka.tolist(), "| yang gagal:", df.loc[angka.isna(), "harga"].tolist())

# SettingWithCopyWarning -> mengubah 'potongan' DataFrame. Gunakan .copy() atau .loc
sub = df[df["qty"] > 1].copy(); sub["qty"] = 0                      # benar: .copy() sebelum diubah
df.loc[df["qty"] > 1, "qty"] = 99                                   # benar: .loc[baris, kolom] = nilai

# ValueError: cannot reindex on an axis with duplicate labels / merge menggandakan baris -> kunci merge tidak unik
a = pd.DataFrame({"k": [1, 1, 2], "x": [1, 2, 3]}); b = pd.DataFrame({"k": [1, 1, 2], "y": [4, 5, 6]})
print("   merge dengan kunci duplikat ->", len(a.merge(b, on="k")), "baris (seharusnya 3?). Periksa a['k'].is_unique;  validate='one_to_one' menghentikan kesalahan ini:")
tampil("merge validate", lambda: a.merge(b, on="k", validate="one_to_one"))

# ValueError: Input contains NaN -> model sklearn tidak menerima NaN. Imputasi dulu (SimpleImputer) atau pakai model yang menangani NaN (LightGBM, HistGB)
from sklearn.linear_model import LinearRegression
tampil("sklearn + NaN", lambda: LinearRegression().fit(pd.DataFrame({"x": [1, np.nan, 3]}), [1, 2, 3]))

# UnicodeDecodeError saat read_csv -> berkas bukan UTF-8. Coba encoding lain
import io
data = "nama;nilai\nJosé;1".encode("latin-1")
tampil("UnicodeDecodeError", lambda: pd.read_csv(io.BytesIO(data), sep=";"))
print("   perbaikan:", pd.read_csv(io.BytesIO(data), sep=";", encoding="latin-1").iloc[0, 0], "| delimiter ; (Excel Indonesia) -> sep=';' dan decimal=','")

# UnicodeEncodeError saat print di Windows (karakter spesial di konsol cp1252) -> set variabel lingkungan PYTHONIOENCODING=utf-8
# atau di awal skrip:  import sys; sys.stdout.reconfigure(encoding="utf-8")

# MemoryError / kernel mati -> data terlalu besar. Lihat resep 20 (dtype hemat, usecols, chunksize)
# ModuleNotFoundError: No module named 'x' -> paket belum terpasang di lingkungan yang dipakai notebook.
#   Pastikan notebook memakai kernel dari venv ini. Pasang offline:  pip install --no-index --find-links=env/wheelhouse nama_paket
# FileNotFoundError -> path relatif salah. Cetak os.getcwd(); pakai path absolut atau Path(__file__)/ROOT seperti di notebook-notebook ini
import os; print("folder kerja:", os.getcwd())

# TypeError: unhashable type / 'NoneType' ... -> fungsi mengembalikan None (mis. df = df.sort_values(inplace=True) menghasilkan None)
x = df.sort_values("qty", inplace=True); print("sort_values(inplace=True) mengembalikan:", x, "<- jangan ditimpakan ke df")

# IndexError: single positional indexer is out-of-bounds -> .iloc di luar jangkauan, atau DataFrame kosong setelah filter (cek len(df))
tampil("iloc di luar jangkauan", lambda: df.iloc[10])
```

## Kebiasaan yang mencegah banyak error
- Setelah memuat data: `df.shape`, `df.dtypes`, `df.head()`, `df.isna().sum()`.
- Jangan menimpa `df` di sel yang dijalankan berulang kali (hasil berbeda tiap run); beri nama baru atau jalankan ulang dari atas.
- Baca pesan error dari **baris paling bawah** ke atas: baris terakhir menyebut penyebabnya, baris atasnya menunjukkan letaknya.
- Restart kernel + Run All sebelum menyimpan hasil agar tahu notebook benar-benar berjalan dari nol.
