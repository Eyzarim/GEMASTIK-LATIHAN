"""Pendaftaran notebook audio: klasifikasi kata, identifikasi pembicara, ASR, retrieval."""
from .registry import register, register_skip

A = "data_latihan/audio/"
AC = dict(root=A + "classification/speech_commands_mini", train_csv="train.csv", test_csv="test.csv", test_labels="test_labels.csv",
          name_data="Speech Commands mini (8 kata: down/go/left/no/right/stop/up/yes; 1200 latih / 320 uji; 1 detik, 16 kHz)")
SP = dict(root=A + "speaker/speech_commands_speakers", train_csv="train.csv", test_csv="test.csv", test_labels="test_labels.csv",
          name_data="Identifikasi 10 pembicara (150 latih / 60 uji, kata campuran; data kecil dan sulit)")
IMP = "import numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom superdb.audio import mfcc_stats"

_CLASSIC = {
    "mfcc_logreg": ("MFCC + Logistic Regression", "from sklearn.linear_model import LogisticRegression",
                    "def featurize(X):\n    return mfcc_stats(X)\nmodel = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))", [],
                    "Baseline audio klasik: ringkasan MFCC (mirip 'sidik jari' spektral) + model linear. Cepat, tanpa GPU.", "Ringkasan statistik membuang urutan waktu; kata yang beda urutan fonem bisa tertukar."),
    "mfcc_svm": ("MFCC + SVM (RBF)", "from sklearn.svm import SVC",
                 "def featurize(X):\n    return mfcc_stats(X)\nmodel = make_pipeline(StandardScaler(), SVC(C=10, kernel='rbf', probability=True))", [],
                 "Biasanya terbaik di antara model klasik pada fitur MFCC dengan data sedang.", "probability=True menambah waktu; skala fitur wajib (sudah di pipeline)."),
    "mfcc_random_forest": ("MFCC + Random Forest", "from sklearn.ensemble import RandomForestClassifier",
                           "def featurize(X):\n    return mfcc_stats(X)\nmodel = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)", [],
                           "Pembanding non-linear tanpa tuning.", "Biasanya kalah dari SVM pada fitur akustik."),
    "mfcc_lightgbm": ("MFCC + LightGBM", "from lightgbm import LGBMClassifier",
                      "def featurize(X):\n    return mfcc_stats(X)\nmodel = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, colsample_bytree=0.5, random_state=42, n_jobs=-1, verbose=-1)", ["lightgbm"],
                      "Fitur audio digabung metadata tabular lain, atau untuk ensemble.", "Data kecil: mudah overfit."),
    "mfcc_knn": ("MFCC + KNN", "from sklearn.neighbors import KNeighborsClassifier",
                 "def featurize(X):\n    return mfcc_stats(X)\nmodel = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance'))", [],
                 "Pembanding berbasis kemiripan.", "Peka skala dan dimensi."),
}
_EMB = {
    "wav2vec2_logreg": ("wav2vec2 (beku) + Logistic Regression", "wav2vec2",
                        "Embedding ucapan self-supervised (768 dim) + classifier linear: transfer learning audio paling praktis. Kuat bila data berlabel sedikit.",
                        "wav2vec2 dilatih pada ucapan Inggris; untuk non-ucapan (bunyi lingkungan) hasilnya lebih lemah. Butuh GPU untuk cepat (CPU tetap bisa)."),
    "whisper_encoder_logreg": ("Encoder Whisper (beku) + Logistic Regression", "whisper",
                               "Embedding dari encoder Whisper-tiny (384 dim) yang dilatih multibahasa pada audio sangat banyak; sering sangat kuat untuk kata/bahasa/pembicara.",
                               "Whisper memproses jendela 30 detik: klip pendek di-pad, hanya frame awal yang dirata-rata (sudah ditangani)."),
}


def _register_all():
    for task, params, labcol in (("classification", AC, "kata"), ("speaker_id", SP, "pembicara")):
        for key, (title, imp, code, req, when, pit) in _CLASSIC.items():
            if task == "speaker_id" and key not in ("mfcc_logreg", "mfcc_svm", "mfcc_lightgbm"):
                continue
            register(modality="audio", task=task, key=key, recipe="aud_classic", title=title, imports=IMP + "\n" + imp, model_code=code,
                     when=when, pitfalls=pit, params=dict(params), requires=["librosa"] + req)
        for key, (title, emb, when, pit) in _EMB.items():
            register(modality="audio", task=task, key=key, recipe="aud_embed", title=title, imports="from sklearn.linear_model import LogisticRegression",
                     model_code="model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))", when=when, pitfalls=pit,
                     params=dict(params, emb=emb), requires=["torch", "transformers"], heavy=True)
    for key, aug, title, when, pit in (("logmel_cnn", False, "CNN log-mel (dari nol)", "Data audio cukup banyak; belajar fitur spektrogram sendiri di GPU.", "Data kecil: butuh puluhan epoch (20 epoch hanya ~0.57 akurasi, 60 epoch ~0.81 pada demo) dan tetap kalah dari embedding pretrained."),
                                       ("logmel_cnn_aug", True, "CNN log-mel + augmentasi", "Sama dengan di atas dengan geser waktu, derau, dan gain acak untuk ketahanan.", "Pada data sangat kecil (1200 klip) augmentasi memperlambat konvergensi: di demo ini belum mengungguli tanpa augmentasi bahkan setelah 120 epoch. Ukur sendiri; manfaatnya muncul pada data lebih banyak/kondisi uji berderau. Augmentasi harus mencerminkan kondisi nyata.")):
        register(modality="audio", task="classification", key=key, recipe="aud_cnn", title=title, when=when, pitfalls=pit,
                 params=dict(AC, aug=aug, epochs=120 if aug else 60), requires=["torch", "torchaudio"], heavy=True)
    register(modality="audio", task="asr", key="whisper_tiny", recipe="aud_asr", title="ASR Whisper-tiny (tanpa melatih)",
             when="Transkripsi ucapan menjadi teks tanpa data latih, multibahasa termasuk Indonesia (language='id'). Evaluasi dengan WER/CER.",
             pitfalls="Klip sangat pendek bisa menghasilkan halusinasi; batasi max_new_tokens. Model tiny tidak sebaik base/small; untuk lomba pertimbangkan membawa Whisper lebih besar.",
             params=dict(AC), requires=["torch", "transformers"], heavy=True)
    for key, feat, title, when, pit in (("mfcc_cosine", "mfcc", "Pencarian audio: MFCC + cosine", "Baseline pencarian klip audio mirip.", "Hanya kemiripan akustik kasar."),
                                        ("wav2vec2_cosine", "wav2vec2", "Pencarian audio: wav2vec2 + cosine", "Kemiripan fonetik/semantik ucapan lebih baik dari MFCC.", "Butuh embedding sekali (di-cache)."),
                                        ("whisper_encoder_cosine", "whisper", "Pencarian audio: encoder Whisper + cosine", "Embedding terkuat untuk ucapan di sini.", "Sama seperti di atas.")):
        register(modality="audio", task="retrieval", key=key, recipe="aud_retrieval", title=title, when=when, pitfalls=pit,
                 params=dict(AC, feat=feat), requires=["librosa"] + (["torch", "transformers"] if feat != "mfcc" else []), heavy=feat != "mfcc")
    register_skip("audio", "enhancement_separation", "*", "model pemisahan sumber/peningkatan suara membutuhkan bobot dan data tambahan; di luar cakupan data mining dasar")
    register_skip("audio", "tts_generation", "*", "sintesis suara butuh model besar; tidak relevan untuk lomba data mining")


_register_all()
