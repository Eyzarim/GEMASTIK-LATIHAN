"""Teks berat: embedding kalimat dan fine-tuning transformer (PyTorch). Bobot dibaca dari pretrained/<nama>/ (offline)."""
import hashlib
import math
import time

import numpy as np

from .paths import ROOT
from .registry import register
from .cachekeys import weights_signature


def pretrained_dir(name):
    d = ROOT / "pretrained" / name
    if not (d / "config.json").exists():
        raise FileNotFoundError(f"Bobot '{name}' belum ada di {d}. Saat persiapan (ada internet) jalankan: python tools/download_pretrained.py {name}")
    return d


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class SentenceEmbedder:
    """Embedding kalimat dengan cache ke disk (outputs/_cache) agar tidak menghitung ulang untuk teks yang sama."""

    def __init__(self, name="minilm", batch_size=64, normalize=True):
        self.name, self.batch_size, self.normalize = name, batch_size, normalize
        self._model = None

    def encode(self, texts):
        texts = [str(t) for t in texts]
        import json
        key = hashlib.sha256(json.dumps([texts, self.name, self.normalize, weights_signature(pretrained_dir(self.name))],
                                       ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
        cache = ROOT / "outputs" / "_cache" / f"emb_{self.name}_{key}.npy"
        if cache.exists():
            return np.load(cache)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(str(pretrained_dir(self.name)), device=get_device(), local_files_only=True)
        t0 = time.time()
        emb = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                 normalize_embeddings=self.normalize, convert_to_numpy=True)
        print(f"embedding {len(texts)} teks selesai dalam {time.time() - t0:.0f} dtk -> {emb.shape}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return emb


def predict_proba_text(model, tok, texts, max_len=64, batch_size=64):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = tok(list(texts[i:i + batch_size]), truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
            out.append(torch.softmax(model(**b).logits, -1).cpu().numpy())
    return np.vstack(out)


def finetune_text_classifier(model_name, Xtr, ytr, Xva, yva, num_labels, epochs=1, max_len=64, lr=3e-5,
                             batch_size=16, seed=42, log=print):
    """Fine-tuning penuh dengan loop PyTorch biasa. Kembalikan (model, tokenizer, riwayat per epoch)."""
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = get_device()
    path = str(pretrained_dir(model_name))
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels,
                                                              ignore_mismatched_sizes=True, local_files_only=True).to(dev)
    Xtr, ytr = list(Xtr), np.asarray(ytr)
    n = len(Xtr)
    steps = epochs * math.ceil(n / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        perm, losses = rng.permutation(n), []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            b = tok([Xtr[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
            loss = model(**b, labels=torch.tensor(ytr[idx]).to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item())
        pv = predict_proba_text(model, tok, list(Xva), max_len).argmax(1)
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "valid_f1_macro": float(f1_score(yva, pv, average="macro")),
                     "detik": round(time.time() - t0)})
        log(hist[-1])
    return model, tok, hist


# ------------------------------------------------------------------ pendaftaran
EMB_MODELS = {
    "minilm_logreg": ("Embedding MiniLM + Logistic Regression", "from sklearn.linear_model import LogisticRegression",
                      "model = LogisticRegression(C=10, max_iter=3000)", [],
                      "Cara tercepat memakai makna kalimat (bukan hanya kata): embedding pretrained + classifier linear. Kuat saat data berlabel sedikit.",
                      "Embedding multibahasa umum kadang kalah dari TF-IDF pada tugas bergantung kata kunci. Selalu bandingkan."),
    "minilm_linearsvc": ("Embedding MiniLM + Linear SVM", "from sklearn.svm import LinearSVC",
                         "model = LinearSVC(C=0.5)", [],
                         "Sama seperti di atas dengan SVM linear; cepat dan sering setara.", "Tanpa probabilitas."),
    "minilm_lightgbm": ("Embedding MiniLM + LightGBM", "from lightgbm import LGBMClassifier",
                        "model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.5, random_state=42, n_jobs=-1, verbose=-1)", ["lightgbm"],
                        "Embedding digabung fitur lain atau untuk ensemble dengan model linear.", "Pada embedding padat biasanya kalah dari model linear; lebih lambat."),
    "minilm_mlp": ("Embedding MiniLM + MLP", "from sklearn.neural_network import MLPClassifier",
                   "model = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=300, random_state=42)", [],
                   "Jaringan kecil di atas embedding; kadang lebih baik dari linear bila data cukup.", "Hasil bervariasi antar seed."),
    "minilm_knn": ("Embedding MiniLM + KNN (cosine)", "from sklearn.neighbors import KNeighborsClassifier",
                   "model = KNeighborsClassifier(n_neighbors=15, metric='cosine', weights='distance')", [],
                   "Klasifikasi berbasis kemiripan makna; juga alat bantu menemukan contoh mirip.", "Lambat saat prediksi untuk data besar."),
}


def _register_all():
    from .text import TASKS
    for task, params in TASKS.items():
        for key, (title, imp, code, req, when, pit) in EMB_MODELS.items():
            register(modality="text", task=task, key=key, recipe="text_embed_clf", title=title, imports=imp, model_code=code,
                     kind="linear", when=when, pitfalls=pit, params=dict(params, embedder="minilm"),
                     requires=["sentence_transformers", "torch"] + req, heavy=True)
    for task in ("sentiment", "classification"):
        for key, mdl, title in (("finetune_distilbert_id", "distilbert_id", "Fine-tuning DistilBERT Indonesia"),
                                ("finetune_indobert", "indobert", "Fine-tuning IndoBERT base")):
            register(modality="text", task=task, key=key, recipe="text_finetune", title=title, imports="", model_code=mdl,
                     kind="nn", params=dict(TASKS[task], model_dir=mdl), requires=["torch", "transformers"], heavy=True,
                     when="Akurasi tertinggi pada teks Indonesia bila ada waktu dan (idealnya) GPU. Mengungguli TF-IDF terutama pada tugas yang butuh pemahaman konteks.",
                     pitfalls="Di CPU hanya demo (subset kecil, 1 epoch): naikkan N_TRAIN/EPOCHS saat dipakai sungguhan. Sensitif learning rate (2e-5 sampai 5e-5). Selalu bandingkan dengan baseline TF-IDF.")


_register_all()
