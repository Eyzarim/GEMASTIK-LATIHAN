# 26 · Audio: memuat, fitur MFCC, klasifikasi cepat

**Kapan:** data berupa suara (WAV/MP3). **Kata kunci:** audio, suara, MFCC, spektrogram, librosa, wav, klasifikasi audio, speech, whisper.

```python
import numpy as np, pandas as pd, soundfile as sf, librosa
from pathlib import Path

ROOT = Path("data_latihan/audio/classification/speech_commands_mini")
tr = pd.read_csv(ROOT / "train.csv").sample(300, random_state=0)               # kolom: path, label

# 1. Muat satu klip: gelombang + laju sampel. Samakan SEMUA ke 16 kHz mono dan panjang tetap.
x, sr = sf.read(ROOT / tr["path"].iloc[0], dtype="float32")
print("laju sampel:", sr, "| durasi:", round(len(x) / sr, 2), "dtk | label:", tr["label"].iloc[0])
x = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x        # (MP3: librosa.load(path, sr=16000))
x = x[:16000] if len(x) >= 16000 else np.pad(x, (0, 16000 - len(x)))               # potong/pad ke 1 detik

# 2. Mel-spektrogram dan MFCC: 'gambar' dari suara
mel = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=16000, n_fft=512, hop_length=160, n_mels=64), ref=np.max)
mfcc = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=20, n_fft=512, hop_length=160)
print("mel:", mel.shape, "| mfcc:", mfcc.shape)

# 3. Pipeline klasik: ringkas MFCC per klip -> classifier
from superdb.audio import load_wavs, mfcc_stats
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
W = load_wavs(tr["path"], ROOT, 16000); F = mfcc_stats(W)
print("akurasi CV MFCC + SVM:", cross_val_score(make_pipeline(StandardScaler(), SVC(C=10)), F, tr["label"], cv=3).mean().round(3))

# 4. Jauh lebih kuat: embedding pretrained (Whisper encoder / wav2vec2) + classifier linear
from superdb.audio import WhisperEmbedder
E = WhisperEmbedder().encode(W)
print("akurasi CV embedding Whisper + LogReg:", cross_val_score(make_pipeline(StandardScaler(), __import__("sklearn.linear_model", fromlist=["x"]).LogisticRegression(max_iter=2000)), E, tr["label"], cv=3).mean().round(3))
```

## Jebakan
- Laju sampel berbeda antar berkas: selalu `resample` ke satu nilai (16 kHz untuk ucapan).
- Klip berbeda panjang: potong/pad, atau pakai statistik (rata-rata/std) agar panjang tetap.
- Pembicara yang sama di train dan valid membuat skor terlalu bagus untuk pengenalan kata; pisahkan menurut pembicara (`GroupKFold`).
- Transkripsi ucapan: `superdb.audio.whisper_transcribe` (offline); evaluasi dengan WER/CER (`notebooks/audio/asr/whisper_tiny`).
- Pilihan lengkap: `notebooks/audio/` (MFCC klasik, CNN log-mel, wav2vec2, Whisper, speaker ID, retrieval).
