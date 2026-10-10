"""Kunci cache dari seluruh input dan konfigurasi, bukan cuplikan beberapa baris."""
import hashlib
import json
from pathlib import Path

import numpy as np


def array_key(X, *options):
    a = np.ascontiguousarray(X)
    h = hashlib.sha256(json.dumps([a.shape, str(a.dtype), options], default=str).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()[:24]


def file_key(paths, root, *options):
    root = Path(root)
    records = []
    for path in paths:
        p = (root / str(path)).resolve()
        s = p.stat()
        records.append((str(p), s.st_size, s.st_mtime_ns))
    return hashlib.sha256(json.dumps([records, options], default=str).encode()).hexdigest()[:24]


def weights_signature(folder):
    """Mengganti bobot di folder yang sama membatalkan cache fitur (ukuran + mtime)."""
    return [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(Path(folder).rglob("*")) if p.is_file()]
