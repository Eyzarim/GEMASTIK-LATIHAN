"""Pendaftaran notebook gambar bagian 2: deteksi, segmentasi, retrieval, anomali/denoising, OCR."""
from .registry import register, register_skip
from .vision_reg import CIF

PEN = dict(name_data="Penn-Fudan (170 gambar pejalan kaki, kotak + mask); 120 latih / 50 uji tetap")

# ------------------------------------------------------------------------- deteksi
_DET = {
    "fasterrcnn_coco_zeroshot": ("Faster R-CNN COCO (tanpa melatih)", "pretrained", "Objek yang kelasnya ada di COCO (orang, mobil, hewan): pakai model jadi, tanpa data latih.",
                                 "Hanya 80 kelas COCO; domain khusus (satelit, medis) gagal. Ambang skor menentukan presisi vs recall."),
    "fasterrcnn_finetune": ("Faster R-CNN fine-tuning", "finetune", "Kelas atau domain yang TIDAK ada di COCO dan ada sedikit data berkotak: fine-tune dari bobot COCO.",
                            "Bila kelasnya sudah ada di COCO (mis. orang), fine-tuning hampir tidak menambah skor dibanding model bawaan. Anotasi kotak harus konsisten; lr terlalu besar merusak bobot pretrained."),
    "yolov8n_coco_zeroshot": ("YOLOv8n COCO (tanpa melatih)", "yolo_zero", "Deteksi cepat real-time untuk kelas COCO dengan pustaka ultralytics.",
                              "Lisensi AGPL-3.0 (periksa bila dipakai di luar riset). Hanya kelas COCO."),
    "yolov8n_finetune": ("YOLOv8n fine-tuning", "yolo_ft", "Alur paling praktis untuk lomba deteksi: data format YOLO, latih beberapa menit di GPU.",
                         "Butuh konversi anotasi ke format YOLO (dibuat otomatis di notebook). Lisensi AGPL-3.0."),
}
# ------------------------------------------------------------------------- segmentasi
_SEG = {
    "lraspp_voc_zeroshot": ("LRASPP VOC (tanpa melatih): kelas 'person'", "zeroshot_lraspp", "Objek termasuk 21 kelas VOC: gunakan model jadi tanpa melatih.", "Hanya kelas VOC; batas objek kasar pada resolusi rendah."),
    "lraspp_finetune": ("LRASPP-MobileNetV3 fine-tuning", "lraspp", "Segmentasi ringan dan cepat dengan transfer learning; titik awal segmentasi di laptop.", "Kepala diganti 2 kelas; resolusi 256px menghilangkan detail halus."),
    "deeplabv3_finetune": ("DeepLabV3-MobileNetV3 fine-tuning", "deeplabv3", "Segmentasi dengan konteks multi-skala (ASPP); biasanya lebih akurat dari LRASPP.", "Lebih lambat dan butuh memori lebih."),
    "unet_scratch": ("U-Net kecil dari nol", "unet", "Citra domain khusus (medis, mikroskop) yang berbeda dari ImageNet; arsitektur segmentasi klasik.", "Dari nol butuh data lebih banyak; pada 120 gambar jauh kalah dari model pretrained."),
}
# ------------------------------------------------------------------------- retrieval
_RET = {
    "hog_cosine": ("HOG + cosine", "hog", "Baseline pencarian gambar mirip berbasis bentuk.", "Tidak menangkap makna semantik."),
    "resnet50_cosine": ("Embedding ResNet50 + cosine", "resnet50", "Pencarian gambar mirip dengan fitur ImageNet; sederhana dan kuat.", "Embedding klasifikasi, bukan dilatih untuk similarity; fine-tuning metric learning lebih baik."),
    "convnext_tiny_cosine": ("Embedding ConvNeXt-Tiny + cosine", "convnext_tiny", "Embedding CNN modern; kualitas lebih tinggi dari ResNet.", "Sama seperti di atas."),
    "vit_b_16_cosine": ("Embedding ViT-B/16 + cosine", "vit_b_16", "Embedding transformer; sangat kuat pada kemiripan semantik.", "Butuh 224px (lebih lambat)."),
    "clip_cosine": ("Embedding CLIP + cosine", "clip", "Embedding semantik terbaik untuk gambar umum; juga memungkinkan pencarian dengan teks.", "Domain khusus kurang cocok."),
}


def _register_all():
    for key, (title, mode, when, pit) in _DET.items():
        register(modality="vision", task="detection", key=key, recipe="vis_det", title=title, when=when, pitfalls=pit,
                 params=dict(PEN, mode=mode), requires=["torch"] + (["ultralytics"] if mode.startswith("yolo") else []), heavy=True)
    for key, (title, mode, when, pit) in _SEG.items():
        register(modality="vision", task="segmentation", key=key, recipe="vis_seg", title=title, when=when, pitfalls=pit,
                 params=dict(PEN, mode=mode, size=256, epochs=12 if mode != "unet" else 40, lr=1e-3 if mode != "unet" else 3e-3), requires=["torch"], heavy=True)
    for key, (title, feat, when, pit) in _RET.items():
        register(modality="vision", task="retrieval", key=key, recipe="vis_retrieval", title=title, when=when, pitfalls=pit,
                 params=dict(CIF, feat=feat, size=224 if feat == "vit_b_16" else 96), requires=["torch"] + (["transformers"] if feat == "clip" else []), heavy=feat != "hog")
    register(modality="vision", task="anomaly", key="conv_autoencoder", recipe="vis_ae", title="Autoencoder konvolusi (galat rekonstruksi)",
             when="Hanya contoh 'normal' tersedia: latih autoencoder pada data normal; gambar aneh direkonstruksi buruk.",
             pitfalls="Autoencoder sering merekonstruksi gambar sederhana apa pun dengan baik (termasuk anomali); fitur pretrained + detektor klasik biasanya lebih kuat.",
             params=dict(CIF, mode="ae_anomaly", normal_class="automobile"), requires=["torch"], heavy=True)
    register(modality="vision", task="anomaly", key="resnet_features_isolation_forest", recipe="vis_ae", title="Fitur ResNet50 + Isolation Forest",
             when="Pendekatan praktis untuk anomali visual: fitur pretrained ditambah detektor tabular klasik.",
             pitfalls="Bergantung pada kecocokan fitur ImageNet dengan domain; anomali halus (retakan kecil) butuh fitur lokal/patch.",
             params=dict(CIF, mode="feat_iforest", normal_class="automobile"), requires=["torch"], heavy=True)
    register(modality="vision", task="denoising", key="denoising_autoencoder", recipe="vis_ae", title="Denoising autoencoder vs filter klasik",
             when="Gambar ber-noise dan tersedia pasangan bersih untuk pelatihan; bandingkan dengan Gaussian/median filter.",
             pitfalls="Noise saat latih harus mirip noise saat uji. Autoencoder kecil bisa mengaburkan detail; ukur dengan PSNR/SSIM dan lihat gambarnya.",
             params=dict(CIF, mode="denoise", sigma=0.1), requires=["torch", "skimage"], heavy=True)
    register(modality="vision", task="ocr", key="easyocr_en_id", recipe="vis_ocr", title="EasyOCR (Inggris + Indonesia)",
             when="Membaca teks dari gambar tanpa melatih; model deteksi + pengenalan siap pakai, offline setelah bobot diunduh.",
             pitfalls="Teks miring/kecil/berlatar ramai menurunkan akurasi; praproses (grayscale, perbesar) membantu. Evaluasi di sini memakai gambar teks SINTETIS buatan sendiri.",
             params=dict(), requires=["easyocr"], heavy=True)
    register_skip("vision", "pose", "*", "estimasi pose butuh model keypoint tambahan dan data berlisensi; belum dibundel")
    register_skip("vision", "image_generation", "*", "model difusi terlalu besar untuk flashdisk/laptop lomba; di luar cakupan data mining")


_register_all()
