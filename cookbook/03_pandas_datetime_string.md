# 03 · Pandas: tanggal & string

**Kapan:** ada kolom tanggal/waktu atau teks pendek yang perlu dibersihkan. **Kata kunci:** datetime, tanggal, to_datetime, dt, resample, rolling, lag, str.contains, regex, strip, lower.

```python
import pandas as pd, numpy as np

# ---------- TANGGAL ----------
s = pd.Series(["2024-01-15", "15/02/2024", "Mar 3, 2024", "tidak valid"])
d = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)   # gagal parse -> NaT (bukan error)
print(d)

# format khusus Indonesia: dd-mm-yyyy
d2 = pd.to_datetime(pd.Series(["31-12-2023", "01-01-2024"]), format="%d-%m-%Y")

tgl = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=400, freq="D")})
tgl["tahun"] = tgl.t.dt.year
tgl["bulan"] = tgl.t.dt.month
tgl["hari_minggu"] = tgl.t.dt.dayofweek          # 0=Senin
tgl["akhir_pekan"] = (tgl.t.dt.dayofweek >= 5).astype(int)
tgl["minggu_ke"] = tgl.t.dt.isocalendar().week.astype(int)
tgl["hari_ke_dalam_tahun"] = tgl.t.dt.dayofyear
tgl["kuartal"] = tgl.t.dt.quarter
tgl["selisih_hari"] = (tgl.t - tgl.t.min()).dt.days
print(tgl.tail(2))

# fitur siklik: bulan 12 dan 1 itu bertetangga, jangan dianggap jauh
tgl["bulan_sin"] = np.sin(2 * np.pi * tgl.bulan / 12)
tgl["bulan_cos"] = np.cos(2 * np.pi * tgl.bulan / 12)

# agregasi per periode (butuh index tanggal)
seri = pd.Series(np.arange(400.0), index=pd.date_range("2024-01-01", periods=400, freq="D"))
print(seri.resample("MS").sum().head(3))         # per bulan; "W" mingguan, "QS" kuartal

# lag & rolling (time series / urutan waktu)
x = seri.to_frame("y")
x["lag1"] = x.y.shift(1)
x["lag7"] = x.y.shift(7)
x["rata7_sebelumnya"] = x.y.shift(1).rolling(7).mean()      # shift(1) dulu agar TIDAK mengintip hari ini
print(x.tail(2))

# ---------- STRING ----------
t = pd.Series(["  Andi Pratama ", "BUDI santoso", "Cici-Rahma", None, "Dedi 123"])
print(t.str.strip().str.lower())
print(t.str.contains("budi", case=False, na=False))
print(t.str.extract(r"(\d+)"))                    # ambil angka pertama
print(t.str.replace(r"[^a-zA-Z ]", "", regex=True).str.strip())
print(t.str.split(" ", expand=True))
print(t.str.len())
print(t.fillna("").str.title())
```

## Jebakan
- Format tanggal ambigu `03/04/2024`: tentukan `dayfirst=True/False`. Salah satu = bulan/hari tertukar diam-diam.
- Time series: **jangan split acak**; pakai urutan waktu (data uji = periode setelah train). Lihat resep 09.
- Rolling/lag tanpa `shift(1)` membocorkan nilai hari yang sedang diprediksi.
- `.str` hanya untuk kolom teks; kolom campur angka/teks beri `.astype(str)` dulu. Di pandas 3 NaN tetap NaN setelah `astype(str)` (bukan teks 'nan'); isi dulu dengan `.fillna('')` bila perlu.
