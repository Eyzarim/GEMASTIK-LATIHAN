"""Pendaftaran notebook gambar (klasifikasi, retrieval, anomali/denoising, deteksi, segmentasi, OCR)."""
from .registry import register, register_skip

V = "data_latihan/vision/"
CIF = dict(root=V + "classification/cifar10_small", train_csv="train.csv", test_csv="test.csv", test_labels="test_labels.csv",
           size=64, name_data="CIFAR-10 subset (4000 latih / 1000 uji, 10 kelas, 32x32)")
IMPC = ("import numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
        "from superdb.vision import hog_features, color_hist_features, lbp_features, pixel_features")

# --------------------------------------------------------------------------- klasik
_CLASSIC = {
    "hog_svm": ("HOG + SVM (RBF)", "from sklearn.svm import SVC",
                "def featurize(X):\n    return hog_features(X)\nmodel = make_pipeline(StandardScaler(), SVC(C=10, kernel='rbf'))", [],
                "Baseline klasik gambar: bentuk/tepi (HOG) + SVM. Cepat, tanpa GPU, kuat untuk gambar kecil dengan pola bentuk jelas.", "Tidak menangkap warna/tekstur; lemah untuk objek alami beragam (CIFAR-10 sulit untuk fitur manual)."),
    "hog_logreg": ("HOG + Logistic Regression", "from sklearn.linear_model import LogisticRegression",
                   "def featurize(X):\n    return hog_features(X)\nmodel = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))", [],
                   "Baseline linear cepat di atas HOG.", "Sama seperti HOG + SVM."),
    "hog_lightgbm": ("HOG + LightGBM", "from lightgbm import LGBMClassifier",
                     "def featurize(X):\n    return hog_features(X)\nmodel = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3, random_state=42, n_jobs=-1, verbose=-1)", ["lightgbm"],
                     "HOG digabung fitur tabular lain di model boosting.", "Dimensi HOG tinggi; boosting lambat dan biasanya tidak lebih baik dari SVM."),
    "color_hist_lightgbm": ("Histogram warna + LightGBM", "from lightgbm import LGBMClassifier",
                            "def featurize(X):\n    return color_hist_features(X)\nmodel = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=42, n_jobs=-1, verbose=-1)", ["lightgbm"],
                            "Kelas dibedakan oleh warna dominan (buah, langit/laut, tanah).", "Membuang semua informasi spasial; gagal bila warna tidak membedakan kelas."),
    "lbp_random_forest": ("LBP (tekstur) + Random Forest", "from sklearn.ensemble import RandomForestClassifier",
                          "def featurize(X):\n    return lbp_features(X)\nmodel = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)", [],
                          "Klasifikasi tekstur (kain, permukaan, jaringan).", "Hanya tekstur lokal; lemah untuk objek."),
    "handcrafted_lightgbm": ("HOG + warna + LBP + LightGBM", "from lightgbm import LGBMClassifier",
                             "def featurize(X):\n    return np.hstack([hog_features(X), color_hist_features(X), lbp_features(X)])\nmodel = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3, random_state=42, n_jobs=-1, verbose=-1)", ["lightgbm"],
                             "Gabungan semua fitur manual: batas atas pendekatan klasik tanpa deep learning.", "Tetap kalah jauh dari transfer learning pada gambar alami."),
    "pixels_pca_svm": ("Piksel mentah + PCA + SVM", "from sklearn.decomposition import PCA\nfrom sklearn.svm import SVC",
                       "def featurize(X):\n    return pixel_features(X, size=16)\nmodel = make_pipeline(StandardScaler(), PCA(n_components=100, random_state=42), SVC(C=5))", [],
                       "Baseline paling sederhana: gambar kecil dihamparkan jadi vektor.", "Tidak invarian terhadap pergeseran; lemah untuk gambar alami."),
}

# --------------------------------------------------------------------------- CNN scratch & fine-tuning
_CNN = {
    "cnn_scratch": ("CNN kecil dari nol", "small_cnn", dict(epochs=20, lr=2e-3, size=32, aug=False, pretrained=False),
                    "Baseline deep learning tanpa bobot pretrained; dataset cukup besar atau domain sangat berbeda dari ImageNet.", "Data kecil (ribuan) membuatnya overfit; transfer learning hampir selalu lebih baik."),
    "cnn_scratch_aug": ("CNN kecil dari nol + augmentasi", "small_cnn", dict(epochs=30, lr=2e-3, size=32, aug=True, pretrained=False),
                        "Sama seperti di atas dengan flip/geser/rotasi acak: mengurangi overfit pada data kecil.", "Augmentasi harus sesuai domain (flip horizontal salah untuk teks/huruf)."),
    "finetune_resnet18": ("Fine-tuning ResNet18", "resnet18", dict(epochs=6, lr=1e-3, size=64, aug=True, pretrained=True),
                          "Pilihan utama transfer learning ringan: seluruh jaringan di-fine-tune dari bobot ImageNet.", "Gambar kecil di-upsample (64px): naikkan size bila waktu ada. lr terlalu besar merusak bobot pretrained."),
    "finetune_mobilenet_v3_small": ("Fine-tuning MobileNetV3-small", "mobilenet_v3_small", dict(epochs=8, lr=2e-3, size=64, aug=True, pretrained=True),
                                    "Laptop tanpa GPU kuat atau batas waktu: model sangat kecil (2.5 juta parameter).", "Akurasi di bawah model besar; BatchNorm butuh batch cukup."),
    "finetune_efficientnet_b0": ("Fine-tuning EfficientNet-B0", "efficientnet_b0", dict(epochs=6, lr=1e-3, size=64, aug=True, pretrained=True),
                                 "Akurasi/parameter terbaik di kelas ringan.", "Lebih lambat dari ResNet18 per epoch; peka terhadap lr."),
    "finetune_convnext_tiny": ("Fine-tuning ConvNeXt-Tiny", "convnext_tiny", dict(epochs=5, lr=3e-4, size=64, aug=True, pretrained=True),
                               "CNN modern berakurasi tinggi; sering terbaik untuk transfer bila GPU memadai.", "Berat (28 juta parameter): kurangi batch bila memori GPU habis."),
}

# --------------------------------------------------------------------------- transfer: fitur beku + logreg
_EMB = {
    "resnet18": (96, "ResNet18"), "mobilenet_v3_small": (96, "MobileNetV3-small"), "efficientnet_b0": (96, "EfficientNet-B0"),
    "resnet50": (96, "ResNet50"), "convnext_tiny": (96, "ConvNeXt-Tiny"), "vit_b_16": (224, "ViT-B/16"),
}

# --------------------------------------------------------------------------- anomali / denoising
# (didaftarkan di vision_reg2 bila sempat)


def _register_all():
    for key, (title, imp, code, req, when, pit) in _CLASSIC.items():
        register(modality="vision", task="classification", key=key, recipe="vis_classic", title=title, imports=IMPC + "\n" + imp,
                 model_code=code, when=when, pitfalls=pit, params=dict(CIF), requires=["skimage"] + req)
    for key, (title, arch, kw, when, pit) in _CNN.items():
        register(modality="vision", task="classification", key=key, recipe="vis_cnn", title=title, imports="", model_code=arch,
                 when=when, pitfalls=pit, params=dict(CIF, arch=arch, **{**kw, "size": kw["size"]}), requires=["torch"], heavy=True)
    for bb, (size, nm) in _EMB.items():
        register(modality="vision", task="classification", key=f"transfer_{bb}_logreg", recipe="vis_embed", title=f"Fitur beku {nm} + Logistic Regression",
                 imports="from sklearn.linear_model import LogisticRegression", model_code="model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))",
                 when=f"Transfer learning tercepat: ekstrak fitur {nm} (bobot ImageNet beku) sekali, latih classifier linear. Kuat dengan data berlabel sedikit, tanpa fine-tuning.",
                 pitfalls="Fitur ImageNet kurang cocok untuk domain sangat berbeda (medis, satelit): fine-tuning lebih baik. Gambar kecil di-upsample menurunkan kualitas fitur.",
                 params=dict(CIF, backbone=bb, size=size), requires=["torch"], heavy=True)
    register(modality="vision", task="classification", key="clip_zeroshot", recipe="vis_clip", title="CLIP zero-shot (tanpa data latih)", imports="", model_code="zeroshot",
             when="Tidak ada/sangat sedikit data berlabel, kelas bisa dinyatakan dengan kata ('a photo of a cat'). Cukup menulis nama kelas.",
             pitfalls="Akurasi bergantung kualitas prompt dan kecocokan domain; kalah dari model yang dilatih dengan data berlabel cukup.",
             params=dict(CIF, mode="zeroshot"), requires=["torch", "transformers"], heavy=True)
    register(modality="vision", task="classification", key="clip_linear_probe", recipe="vis_clip", title="CLIP + Logistic Regression (linear probe)", imports="", model_code="probe",
             when="Embedding CLIP sangat kuat untuk gambar umum; classifier linear di atasnya sering menyamai fine-tuning dengan data sedikit.",
             pitfalls="Butuh embedding sekali (cepat di GPU). Domain khusus (medis) kurang cocok.",
             params=dict(CIF, mode="probe"), requires=["torch", "transformers"], heavy=True)
    register_skip("vision", "multilabel_classification", "*", "tidak ada dataset multilabel gambar kecil berlisensi jelas yang ikut dibundel; gunakan pola klasifikasi + sigmoid/BCE (lihat cookbook)")


_register_all()
