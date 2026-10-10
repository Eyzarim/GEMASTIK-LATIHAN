"""Fusion teks + gambar + tabel. Encoder beku lokal, preprocessing hanya pada train.

Setiap blok (N,D+1) diakhiri indikator hadir. CSV adalah satu baris per objek;
kolom gambar berisi path lokal relatif terhadap image_root, bukan URL.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from .paths import ROOT
from .registry import register


def split_indices(frame, y, task, valid_size=0.2, seed=42, group_col=None, time_col=None):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    idx = np.arange(len(frame))
    if group_col and time_col:
        raise ValueError("Pilih satu skema split: grup atau waktu")
    if group_col:
        groups = frame[group_col]
        if groups.isna().any():
            raise ValueError("Kolom grup tidak boleh kosong")
        tr, va = next(GroupShuffleSplit(1, test_size=valid_size, random_state=seed).split(idx, y, groups))
        return tr, va
    if time_col:
        times = pd.to_datetime(frame[time_col], errors="raise")
        if times.isna().any():
            raise ValueError("Kolom waktu tidak boleh kosong")
        unique = np.sort(times.unique())
        cut = int(len(unique) * (1 - valid_size))
        if cut < 1 or cut >= len(unique):
            raise ValueError("Butuh minimal dua waktu berbeda untuk split temporal")
        return idx[times < unique[cut]], idx[times >= unique[cut]]
    return train_test_split(idx, test_size=valid_size, random_state=seed,
                            stratify=y if task == "classification" else None)


class FrozenFeatures:
    """Embedding MiniLM + CLIP opsional + numerik/kategori. Tidak menebak kolom fitur.

    Bobot selalu lokal. Tidak ada fine-tuning encoder di kelas ini. Model encoder
    tidak ikut joblib; saat reload dibaca lagi dari pretrained lokal.
    """
    def __init__(self, text_col=None, image_col=None, num_cols=(), cat_cols=(), image_root=None,
                 text_model="minilm", image_model="clip_b32", batch_size=32):
        self.text_col, self.image_col = text_col, image_col
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        self.image_root = str(image_root or ROOT)
        self.text_model, self.image_model, self.batch_size = text_model, image_model, batch_size
        self.table_ = None
        self._text_encoder = self._image_encoder = None

    def fit(self, frame):
        steps = []
        if self.num_cols:
            steps.append(("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                                StandardScaler()), self.num_cols))
        if self.cat_cols:
            steps.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                                                OneHotEncoder(handle_unknown="ignore", sparse_output=False)), self.cat_cols))
        if steps:
            self.table_ = ColumnTransformer(steps).fit(self._table_frame(frame))
        if not steps and not self.text_col and not self.image_col:
            raise ValueError("Pilih setidaknya satu modalitas")
        return self

    def _table_frame(self, frame):
        out = frame[self.num_cols + self.cat_cols].copy()
        for c in self.num_cols:
            out[c] = pd.to_numeric(out[c], errors="raise")
        for c in self.cat_cols:
            out[c] = out[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
        return out

    def transform(self, frame):
        from pathlib import Path
        blocks = {}
        if self.text_col:
            from .text_heavy import SentenceEmbedder
            if self._text_encoder is None:
                self._text_encoder = SentenceEmbedder(self.text_model, self.batch_size)
            texts = frame[self.text_col].fillna("").astype(str)
            present = texts.str.strip().ne("").to_numpy()
            emb = self._text_encoder.encode(texts.tolist()).astype(np.float32)
            emb[~present] = 0
            blocks["text"] = np.column_stack([emb, present]).astype(np.float32)
        if self.image_col:
            from .vision import ClipEmbedder, load_images
            from .cachekeys import file_key, weights_signature
            values = frame[self.image_col].fillna("").astype(str)
            present = values.str.strip().ne("").to_numpy()
            idx = np.flatnonzero(present)
            emb = np.zeros((len(frame), 512), dtype=np.float32)
            if len(idx):
                paths = values.iloc[idx].tolist()
                key = file_key(paths, self.image_root, self.image_model,
                               weights_signature(ROOT / "pretrained" / self.image_model))
                dest = ROOT / "outputs/_cache" / f"fusion_img_{key}.npy"
                if dest.exists():
                    features = np.load(dest)
                else:
                    if self._image_encoder is None:
                        self._image_encoder = ClipEmbedder(self.image_model)
                    parts = []
                    for start in range(0, len(paths), self.batch_size):
                        images = load_images(paths[start:start + self.batch_size], Path(self.image_root), size=224, cache=False)
                        parts.append(self._image_encoder.images(images, self.batch_size))
                    features = np.vstack(parts).astype(np.float32)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    np.save(dest, features)
                if features.shape[1] != 512:
                    raise ValueError("Adapter image ini memakai CLIP-B/32 (512 dimensi)")
                emb[idx] = features
            blocks["image"] = np.column_stack([emb, present]).astype(np.float32)
        if self.table_ is not None:
            data = self._table_frame(frame)
            present = data.notna().any(axis=1).to_numpy()
            emb = np.asarray(self.table_.transform(data), dtype=np.float32)
            blocks["table"] = np.column_stack([emb, present]).astype(np.float32)
        return blocks

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_text_encoder"] = state["_image_encoder"] = None
        return state


class FusionModel:
    """linear | lightgbm: early fusion; late: rata-rata prediksi; torch: gated fusion.

    PyTorch memakai validation internal dari data training, AMP, clipping,
    accumulation, early stopping, dan checkpoint terbaik. Prediksi dalam batch.
    """
    def __init__(self, method="linear", task="classification", seed=42, epochs=30, batch_size=64,
                 hidden=64, lr=1e-3, patience=5, accumulation=1, modality_dropout=0.1):
        self.method, self.task, self.seed = method, task, seed
        self.epochs, self.batch_size, self.hidden, self.lr = epochs, batch_size, hidden, lr
        self.patience, self.accumulation, self.modality_dropout = patience, accumulation, modality_dropout

    def _estimator(self, boosting=False):
        if boosting:
            from lightgbm import LGBMClassifier, LGBMRegressor
            cls = LGBMClassifier if self.task == "classification" else LGBMRegressor
            return cls(n_estimators=200, num_leaves=15, learning_rate=0.05, random_state=self.seed, n_jobs=2, verbosity=-1)
        from sklearn.linear_model import LogisticRegression, Ridge
        return LogisticRegression(C=1, max_iter=2000, random_state=self.seed) if self.task == "classification" else Ridge(alpha=10)

    def _scaled(self, blocks):
        return [self.scalers_[k].transform(blocks[k]).astype(np.float32) for k in self.keys_]

    def fit(self, blocks, y, inner_train=None, inner_valid=None):
        self.keys_ = list(blocks)
        y = np.asarray(y)
        if self.method not in ("linear", "lightgbm", "late", "torch"):
            raise ValueError("method harus linear, lightgbm, late, atau torch")
        if self.task == "classification":
            self.classes_, encoded = np.unique(y, return_inverse=True)
            if len(self.classes_) < 2:
                raise ValueError("Training harus mengandung minimal dua kelas")
        if self.method == "torch":
            if self.epochs < 1 or self.accumulation < 1:
                raise ValueError("epochs dan accumulation harus >= 1")
            if inner_train is None or inner_valid is None:
                a, b = split_indices(pd.DataFrame(index=range(len(y))), y, self.task, 0.15, self.seed)
            else:
                a, b = np.asarray(inner_train), np.asarray(inner_valid)
            if not len(a) or not len(b) or np.intersect1d(a, b).size:
                raise ValueError("Train/validation internal harus tidak kosong dan saling terpisah")
        else:
            a = np.arange(len(y))
        self.scalers_ = {k: StandardScaler().fit(blocks[k][a]) for k in self.keys_}
        features = self._scaled(blocks)
        if self.method == "late":
            self.models_ = [self._estimator().fit(x, y) for x in features]
        elif self.method in ("linear", "lightgbm"):
            self.model_ = self._estimator(self.method == "lightgbm").fit(np.column_stack(features), y)
        else:
            import torch
            import torch.nn.functional as F
            from .fusion_net import GatedFusionNet
            torch.manual_seed(self.seed)
            rng = np.random.default_rng(self.seed)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            out_dim = len(self.classes_) if self.task == "classification" else 1
            self.net_ = GatedFusionNet([x.shape[1] for x in features], out_dim, self.hidden).to(dev)
            if self.task == "regression":
                self.target_mean_, self.target_std_ = float(y[a].mean()), float(y[a].std() + 1e-8)
                target = torch.tensor((y - self.target_mean_) / self.target_std_, dtype=torch.float32)
            else:
                if len(np.unique(encoded[a])) != out_dim:
                    raise ValueError("Split internal tidak memuat semua kelas di train")
                target = torch.tensor(encoded, dtype=torch.long)
            tensors = [torch.from_numpy(x) for x in features]  # tetap di RAM; transfer per batch
            opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
            def loss_fn(pred, truth):
                return F.cross_entropy(pred, truth) if self.task == "classification" else F.mse_loss(pred.squeeze(1), truth)
            best, bad, best_state = np.inf, 0, None
            self.history_ = []
            for epoch in range(self.epochs):
                self.net_.train()
                order = rng.permutation(a)
                train_loss, seen = 0.0, 0
                # Satu accumulation group dibobot menurut jumlah sampel, termasuk batch terakhir.
                group_size = self.batch_size * self.accumulation
                for start in range(0, len(order), group_size):
                    group = order[start:start + group_size]
                    opt.zero_grad(set_to_none=True)
                    for j in range(0, len(group), self.batch_size):
                        idx = group[j:j + self.batch_size]
                        xb = [x[idx].to(dev) for x in tensors]
                        mask = torch.rand(len(idx), len(xb), device=dev) >= self.modality_dropout
                        mask[:, 0] |= ~mask.any(1)  # selalu sisakan satu blok
                        xb = [x * mask[:, k:k+1] for k, x in enumerate(xb)]
                        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                            loss = loss_fn(self.net_(xb), target[idx].to(dev))
                        scaler.scale(loss * len(idx) / len(group)).backward()
                        train_loss += float(loss.item()) * len(idx); seen += len(idx)
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                self.net_.eval()
                total = 0.0
                with torch.no_grad():
                    for start in range(0, len(b), self.batch_size):
                        idx = b[start:start + self.batch_size]
                        total += float(loss_fn(self.net_([x[idx].to(dev) for x in tensors]), target[idx].to(dev)).item()) * len(idx)
                val_loss = total / len(b)
                self.history_.append(dict(epoch=epoch+1, train_loss=train_loss/seen, valid_loss=val_loss))
                if val_loss < best:
                    best, bad = val_loss, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()}
                    self.best_epoch_ = epoch + 1
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            if best_state is None:
                raise RuntimeError("Loss training/validation tidak finite; periksa input")
            self.net_.load_state_dict(best_state)
            self.net_.cpu()
        return self

    def _raw(self, blocks):
        features = self._scaled(blocks)
        if self.method == "late":
            fn = "predict_proba" if self.task == "classification" else "predict"
            return np.mean([getattr(m, fn)(x) for m, x in zip(self.models_, features)], axis=0)
        if self.method != "torch":
            fn = "predict_proba" if self.task == "classification" else "predict"
            return getattr(self.model_, fn)(np.column_stack(features))
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net_.to(dev).eval()
        result = []
        with torch.no_grad():
            for start in range(0, len(features[0]), self.batch_size):
                out = self.net_([torch.from_numpy(x[start:start+self.batch_size]).to(dev) for x in features])
                result.append((out.softmax(1) if self.task == "classification" else out.squeeze(1)).cpu().numpy())
        self.net_.cpu()
        pred = np.concatenate(result)
        return pred if self.task == "classification" else pred * self.target_std_ + self.target_mean_

    def predict_proba(self, blocks):
        if self.task != "classification":
            raise ValueError("Regresi tidak menghasilkan probabilitas kelas")
        return self._raw(blocks)

    def predict(self, blocks):
        raw = self._raw(blocks)
        return self.classes_[raw.argmax(1)] if self.task == "classification" else raw


for task in ("classification", "regression"):
    for method, title in (("linear", "Early fusion + model linear"), ("lightgbm", "Early fusion + LightGBM"),
                          ("late", "Late fusion: rata-rata prediksi modalitas"), ("torch", "Gated fusion PyTorch")):
        register(modality="multimodal", task=task, key=method, recipe="multimodal_supervised", title=title,
                 params=dict(task=task, method=method), requires=["sentence_transformers", "torch"], heavy=True,
                 when="Satu objek memiliki teks, gambar lokal opsional, dan fitur numerik/kategori. Encoder beku cocok untuk VRAM 8 GB.",
                 pitfalls="Cocokkan objek melalui ID sebelum split; preprocessing hanya train. Demo lokal MovieLens memakai teks+tabel, bukan tiga modalitas. Bandingkan tiap modalitas pada split sama; fusion belum tentu menaikkan skor.")
