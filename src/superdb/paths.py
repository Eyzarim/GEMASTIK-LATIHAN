"""Lokasi folder proyek. Tidak bergantung pada drive letter flashdisk."""
from pathlib import Path


def find_root(start=None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / "src" / "superdb").is_dir():
            return d
    raise RuntimeError("Folder proyek (berisi src/superdb) tidak ditemukan dari " + str(p))


ROOT = Path(__file__).resolve().parents[2]


def out_dir(modality: str, task: str, model: str) -> Path:
    d = ROOT / "outputs" / modality / task / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(*parts) -> Path:
    return ROOT.joinpath("data_latihan", *parts)
