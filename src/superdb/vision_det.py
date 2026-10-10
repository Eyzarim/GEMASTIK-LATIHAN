"""Deteksi & segmentasi: data PennFudan, AP50, IoU/Dice, U-Net kecil, pelatih Faster R-CNN dan segmentasi torchvision."""
import time

import numpy as np

from .paths import ROOT
from .vision import device, pretrained_path

PF = ROOT / "data_latihan" / "vision" / "detection" / "PennFudanPed"


# ================================================================== data PennFudan
def load_pennfudan(size=None):
    """Kembalikan daftar dict: img (uint8 HxWx3), boxes (n,4 xyxy), mask (HxW biner pejalan kaki). size -> ubah ukuran persegi (mask nearest)."""
    from PIL import Image
    imgs = sorted((PF / "PNGImages").glob("*.png"))
    out = []
    for p in imgs:
        im = Image.open(p).convert("RGB")
        m = np.asarray(Image.open(PF / "PedMasks" / p.name.replace(".png", "_mask.png")))
        ids = [i for i in np.unique(m) if i != 0]
        boxes = []
        for i in ids:
            ys, xs = np.where(m == i)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        sem = (m > 0).astype(np.uint8)
        if size:
            sx, sy = size / im.width, size / im.height
            im = im.resize((size, size), Image.BILINEAR)
            sem = np.asarray(Image.fromarray(sem * 255).resize((size, size), Image.NEAREST)) // 255
            boxes = boxes * np.array([sx, sy, sx, sy], np.float32)
        out.append({"name": p.name, "img": np.asarray(im), "boxes": boxes, "mask": sem.astype(np.uint8)})
    return out


def split_idx(n, n_test=50):
    """Pemisahan tetap: n_test gambar terakhir untuk uji."""
    return np.arange(n - n_test), np.arange(n - n_test, n)


# ================================================================== metrik deteksi/segmentasi
def box_iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def ap50(preds, gts, iou_thr=0.5):
    """Average Precision (VOC, semua titik) satu kelas. preds: list (boxes (n,4), scores (n,)) per gambar; gts: list boxes (m,4) per gambar."""
    dets = []
    for i, (b, s) in enumerate(preds):
        for bb, ss in zip(np.asarray(b).reshape(-1, 4), np.asarray(s).reshape(-1)):
            dets.append((float(ss), i, bb))
    dets.sort(key=lambda t: -t[0])
    n_gt = sum(len(g) for g in gts)
    used = [np.zeros(len(g), bool) for g in gts]
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for k, (s, i, bb) in enumerate(dets):
        g = np.asarray(gts[i]).reshape(-1, 4)
        if len(g):
            iou = box_iou(bb[None], g)[0]; j = int(iou.argmax())
            if iou[j] >= iou_thr and not used[i][j]:
                tp[k] = 1; used[i][j] = True; continue
        fp[k] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / max(n_gt, 1); prec = tp / np.maximum(tp + fp, 1e-9)
    mrec, mpre = np.r_[0, rec, 1], np.r_[0, prec, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"AP50": ap, "recall_maks": float(rec[-1]) if len(rec) else 0.0, "n_deteksi": len(dets), "n_gt": n_gt}


def seg_metrics(pred, gt):
    """IoU dan Dice untuk kelas foreground (rata-rata total piksel gabungan seluruh gambar)."""
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    inter = (pred & gt).sum(); union = (pred | gt).sum()
    return {"IoU": float(inter / max(union, 1)), "Dice": float(2 * inter / max(pred.sum() + gt.sum(), 1)), "pixel_acc": float((pred == gt).mean())}


# ================================================================== Faster R-CNN
def fasterrcnn(pretrained_coco=True, num_classes=None):
    """Faster R-CNN MobileNetV3-320 FPN. pretrained_coco memuat bobot COCO lokal; num_classes mengganti kepala (termasuk latar)."""
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    if pretrained_coco:
        m.load_state_dict(torch.load(pretrained_path("fasterrcnn_mobilenet"), map_location="cpu"))
    if num_classes:
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return m


def _tensor_img(x, dev):
    import torch
    return torch.from_numpy(x).to(dev).permute(2, 0, 1).float() / 255


def detect_fasterrcnn(model, imgs, score_thr=0.05, label=None, batch=8):
    """Jalankan deteksi; kembalikan list (boxes, scores). label: hanya kelas ini (mis. 1=person pada COCO)."""
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            res = model([_tensor_img(x, dev) for x in imgs[i:i + batch]])
            for r in res:
                keep = r["scores"] >= score_thr
                if label is not None:
                    keep &= r["labels"] == label
                out.append((r["boxes"][keep].cpu().numpy(), r["scores"][keep].cpu().numpy()))
    return out


def train_fasterrcnn(model, imgs, boxes, epochs=5, lr=5e-3, batch=4, seed=42, log=print, labels=None):
    """Fine-tuning dengan SGD; satu kelas objek (label 1) + latar."""
    import torch
    torch.manual_seed(seed)
    dev = device()
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.3)
    rng = np.random.default_rng(seed)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(imgs)); tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            ims = [_tensor_img(imgs[j], dev) for j in idx]
            tg = [{"boxes": torch.from_numpy(boxes[j]).to(dev), "labels": torch.ones(len(boxes[j]), dtype=torch.int64, device=dev)} for j in idx]
            loss = sum(model(ims, tg).values())
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        hist.append({"epoch": ep + 1, "loss": tot / len(perm), "detik": round(time.time() - t0)}); log(hist[-1])
    return hist


# ================================================================== segmentasi
def unet(base=16):
    """U-Net kecil (3 level) untuk segmentasi biner. Masukan (B,3,H,W) dengan H,W kelipatan 8."""
    import torch
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = blk(3, base), blk(base, base * 2), blk(base * 2, base * 4)
            self.b = blk(base * 4, base * 8)
            self.u3, self.d3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2), blk(base * 8, base * 4)
            self.u2, self.d2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), blk(base * 4, base * 2)
            self.u1, self.d1 = nn.ConvTranspose2d(base * 2, base, 2, 2), blk(base * 2, base)
            self.out = nn.Conv2d(base, 2, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); b = self.b(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(b), e3], 1)); d2 = self.d2(torch.cat([self.u2(d3), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)
    return UNet()


def seg_model(kind, pretrained=True):
    """Model segmentasi torchvision dengan kepala 2 kelas (latar/orang). kind: 'lraspp' | 'deeplabv3'. Output model(x)['out'] atau tensor."""
    import torch
    import torch.nn as nn
    import torchvision
    if kind == "lraspp":
        m = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
        if pretrained:
            m.load_state_dict(torch.load(pretrained_path("lraspp_mobilenet"), map_location="cpu"))
    else:
        m = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        if pretrained:
            m.load_state_dict(torch.load(pretrained_path("deeplabv3_mobilenet"), map_location="cpu"))
    return m


def to_binary_head(m, kind):
    import torch.nn as nn
    if kind == "lraspp":
        m.classifier.low_classifier = nn.Conv2d(m.classifier.low_classifier.in_channels, 2, 1)
        m.classifier.high_classifier = nn.Conv2d(m.classifier.high_classifier.in_channels, 2, 1)
    else:
        m.classifier[4] = nn.Conv2d(m.classifier[4].in_channels, 2, 1)
        m.aux_classifier = None
    return m


def seg_forward(model, x):
    out = model(x)
    return out["out"] if isinstance(out, dict) else out


def train_segmenter(model, X, M, Xv, Mv, epochs=15, lr=1e-3, batch=8, size=256, seed=42, log=print, imagenet_norm=True):
    """Pelatihan segmentasi biner (cross-entropy). X uint8 (N,H,W,3), M uint8 (N,H,W). Kembalikan riwayat (IoU validation per epoch)."""
    import torch
    from .vision import to_batch
    torch.manual_seed(seed)
    dev = device()
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    Mt = torch.from_numpy(M).long().to(dev)
    rng = np.random.default_rng(seed)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(X)); tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, mb = to_batch(X[idx], size, dev, imagenet_norm), Mt[idx]
            if rng.random() < 0.5:
                xb, mb = xb.flip(-1), mb.flip(-1)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = torch.nn.functional.cross_entropy(seg_forward(model, xb).float(), mb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item() * len(idx)
        pv = predict_masks(model, Xv, size, imagenet_norm=imagenet_norm)
        hist.append({"epoch": ep + 1, "loss": tot / len(X), "val_IoU": seg_metrics(pv, Mv)["IoU"], "detik": round(time.time() - t0)}); log(hist[-1])
    return hist


def predict_masks(model, X, size=256, batch=16, imagenet_norm=True, person_class=None):
    """Mask prediksi biner (N,H,W). person_class: bila model punya 21 kelas VOC (zero-shot), foreground = kelas itu."""
    import torch
    from .vision import to_batch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits = seg_forward(model, to_batch(X[i:i + batch], size, dev, imagenet_norm)).float()
            pred = (logits.argmax(1) == person_class) if person_class is not None else logits.argmax(1).bool()
            out.append(pred.cpu().numpy().astype(np.uint8))
    return np.concatenate(out)
