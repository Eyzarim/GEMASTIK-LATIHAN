"""Registry: SATU sumber kebenaran untuk semua kombinasi (modalitas x task x model).

Generator notebook, STATUS.md, dan MATRIX.md semuanya dibaca dari sini.
Kombinasi yang tidak masuk akal TIDAK didaftarkan; alasannya dicatat lewat register_skip().
"""
from dataclasses import dataclass, field
import importlib

ENTRIES: list["Entry"] = []
SKIPS: list[tuple] = []


@dataclass
class Entry:
    modality: str            # tabular, text, timeseries, image, audio, ...
    task: str                # classification, regression, clustering, ...
    key: str                 # nama pendek model, jadi nama file notebook
    title: str               # judul tampilan
    recipe: str              # nama resep di tools/recipes yang menyusun sel notebook
    imports: str = ""        # baris import yang ditampilkan di notebook
    model_code: str = ""     # kode yang mendefinisikan `model` (ditampilkan apa adanya)
    kind: str = "other"      # linear | tree | nn | other (mengatur preprocessing)
    when: str = ""           # kapan metode ini cocok
    pitfalls: str = ""       # jebakan umum
    params: dict = field(default_factory=dict)   # config default notebook (path data, kolom, metrik)
    requires: list = field(default_factory=list)  # modul yang harus bisa di-import
    heavy: bool = False      # butuh torch/GPU/bobot besar; diberi tanda di STATUS

    @property
    def nb_path(self) -> str:
        return f"notebooks/{self.modality}/{self.task}/{self.key}.ipynb"


def register(**kw) -> Entry:
    e = Entry(**kw)
    ENTRIES.append(e)
    return e


def register_skip(modality: str, task: str, model: str, reason: str):
    SKIPS.append((modality, task, model, reason))


MODULES = [
    "superdb.tabular",
    "superdb.tabular_extra",
    "superdb.tabular_extra_b",
    "superdb.tabular_more",
    "superdb.text",
    "superdb.text_extra",
    "superdb.text_heavy",
    "superdb.timeseries",
    "superdb.timeseries_reg",
    "superdb.unsupervised",
    "superdb.unsupervised_reg",
    "superdb.vision",
    "superdb.vision_reg",
    "superdb.vision_reg2",
    "superdb.audio",
    "superdb.audio_reg",
    "superdb.graph",
    "superdb.graph_reg",
    "superdb.recsys",
    "superdb.recsys_reg",
    "superdb.deep_tabular",
    "superdb.multimodal",
]


def load_all():
    """Import semua modul yang mendaftarkan entry. Modul yang belum ada dilewati."""
    for m in MODULES:
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as ex:
            if ex.name != m:   # dependensi di dalam modul hilang -> tampilkan
                raise
    return ENTRIES
