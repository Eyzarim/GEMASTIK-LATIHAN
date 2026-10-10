# 30 · Strategi lomba: waktu, prioritas, laporan

**Kapan:** hari lomba (2 soal, 5 jam, tanpa internet). **Kata kunci:** strategi, waktu, prioritas, lomba, checklist, laporan, presentasi, submission, GEMASTIK.

```python
# Pembagian waktu 5 jam / 2 soal (contoh; sesuaikan). Jalankan sel ini untuk mencetak jadwal.
import pandas as pd
jadwal = pd.DataFrame([
    ("0:00-0:20", "Baca KEDUA soal: metrik, format submission, data, aturan, kolom ID/waktu/grup", "pilih urutan: yang paling jelas dulu"),
    ("0:20-0:50", "Soal A: EDA singkat (resep 04) + skema validasi (09/12) + baseline (27)", "submission pertama VALID (resep 17)"),
    ("0:50-1:30", "Soal A: model kuat (LightGBM/notebooks per tipe data) + catat CV", "submission ke-2"),
    ("1:30-2:00", "Soal B: EDA + validasi + baseline", "submission pertama B"),
    ("2:00-2:40", "Soal B: model kuat", "submission ke-2 B"),
    ("2:40-4:00", "Perbaikan terarah di soal dengan potensi naik terbesar: fitur (16), tuning (14), ensemble (15)", "jangan membuka eksperimen baru setelah 3:40"),
    ("4:00-4:40", "Laporan/slide: tempel grafik dan tabel dari outputs/ (laporan/make_figures.py)", "analisis kesalahan + 3 temuan utama"),
    ("4:40-5:00", "Cek akhir: format submission, file tersimpan di 2 tempat, kode dapat dijalankan ulang", "JANGAN mengubah apa pun lagi"),
], columns=["waktu", "kegiatan", "keluaran"])
print(jadwal.to_string(index=False))
```

## Peta cepat: data apa → mulai dari mana
Lihat `00_MULAI_DI_SINI.md`. Ringkasnya: tabular → `notebooks/tabular/classification|regression` (mulai `lightgbm`); teks → `notebooks/text/` (mulai `tfidf_linearsvc_word`, lalu `finetune_*`/`minilm_*`); gambar → `notebooks/vision/classification` (mulai `clip_linear_probe` atau `finetune_resnet18`); time series → `notebooks/timeseries/forecasting` (mulai `seasonal_naive`, `holt_winters_*`, `lightgbm_recursive`); audio → `notebooks/audio/classification` (mulai `whisper_encoder_logreg`).

## Aturan yang menyelamatkan
1. **Submission pertama dalam 45 menit**, walau jelek: memastikan format dan alur benar.
2. **Satu skema validasi untuk semua eksperimen**; catat setiap eksperimen (model, fitur, CV, std) di tabel kecil. Skor CV tanpa catatan tidak dapat dibandingkan.
3. Perubahan **satu per satu**, ukur dengan CV; yang tidak naik melebihi `std` dibuang.
4. Skor CV dan leaderboard berbeda jauh → hentikan eksperimen, periksa kebocoran/skema validasi (resep 12).
5. Waktu terbatas: model kuat standar (LightGBM / transfer learning) + fitur bagus mengalahkan model eksotis yang belum sempat dituning.
6. Simpan semua submission bernomor dengan skor CV di nama berkas; jangan menimpa.
7. **Laporan dinilai**: tulis sambil bekerja (catatan 2-3 kalimat per eksperimen), jangan menunggu jam terakhir.

## Kerangka laporan
Latar belakang & tujuan → data & EDA (temuan, bukan daftar) → metodologi (pra-proses, validasi, model, alasan) → eksperimen (tabel perbandingan + ketidakpastian) → analisis kesalahan & interpretasi (SHAP/importance) → kesimpulan & keterbatasan. Template: `laporan/`.
