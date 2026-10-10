# 29 · GPU dan PyTorch: tips praktis

**Kapan:** memakai GPU untuk deep learning. **Kata kunci:** GPU, CUDA, torch, out of memory, mixed precision, batch size, DataLoader, nvidia-smi, device, fp16.

```python
import time, torch, numpy as np

# 1. Cek GPU dan versi. torch+cu128 dibutuhkan untuk GPU Blackwell (RTX 50xx); build '+cpu' tidak memakai GPU sama sekali.
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda, "| tersedia:", torch.cuda.is_available())
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    p = torch.cuda.get_device_properties(0); print(p.name, f"| {p.total_memory / 1e9:.1f} GB VRAM | compute capability {p.major}.{p.minor}")

# 2. Tulis kode device-agnostik (jalan di GPU maupun CPU)
net = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 10)).to(dev)
x = torch.randn(4096, 256, device=dev); y = torch.randint(0, 10, (4096,), device=dev)
opt = torch.optim.AdamW(net.parameters(), 1e-3)

def latih(n_langkah, amp):
    scaler = torch.amp.GradScaler(enabled=amp and dev == "cuda")
    t0 = time.time()
    for _ in range(n_langkah):
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=amp and dev == "cuda"):       # mixed precision: lebih cepat, hemat VRAM
            loss = torch.nn.functional.cross_entropy(net(x), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if dev == "cuda": torch.cuda.synchronize()                                                           # tunggu GPU selesai sebelum mengukur waktu
    return time.time() - t0
print(f"fp32: {latih(100, False):.2f} dtk | mixed precision: {latih(100, True):.2f} dtk")

# 3. Evaluasi: eval() + no_grad (hemat memori, perilaku BatchNorm/Dropout benar)
net.eval()
with torch.no_grad():
    akurasi = (net(x).argmax(1) == y).float().mean().item()
print("akurasi (data acak, hanya demo):", round(akurasi, 3))

# 4. Memori GPU
if dev == "cuda":
    print(f"VRAM dipakai: {torch.cuda.memory_allocated() / 1e6:.0f} MB | puncak: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    torch.cuda.empty_cache()
```

## Jika "CUDA out of memory"
1. Kecilkan `batch_size` (setengahnya) atau ukuran gambar.
2. Aktifkan mixed precision (`torch.autocast`), evaluasi di `torch.no_grad()`.
3. `del` tensor besar yang tidak dipakai lalu `torch.cuda.empty_cache()`; restart kernel bila fragmentasi.
4. Gradient accumulation: beberapa langkah kecil sebelum `opt.step()` (batch efektif besar tanpa VRAM besar).
5. Tutup program lain yang memakai GPU (`nvidia-smi` untuk melihat).

## Jebakan
- Windows: `DataLoader(num_workers=0)` bila macet; data kecil di GPU langsung (seperti notebook di repo) lebih cepat.
- GPU **tidak selalu lebih cepat**: model kecil/data kecil kadang lebih cepat di CPU karena overhead transfer.
- Panas CPU/GPU: batasi proses paralel (`--jobs 2`), jangan jalankan banyak notebook berat sekaligus.
- Hasil GPU tidak persis identik antar-run kecuali mode deterministik (resep 22).
