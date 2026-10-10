"""Helper bersama: seed, metrik, plot, simpan hasil. Dipakai semua modalitas."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- metrik
def classification_metrics(y_true, y_pred, y_proba=None, classes=None) -> dict:
    from sklearn import metrics as m
    out = {
        "accuracy": m.accuracy_score(y_true, y_pred),
        "balanced_accuracy": m.balanced_accuracy_score(y_true, y_pred),
        "f1_macro": m.f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": m.f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": m.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": m.recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None:
        try:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                p = y_proba if y_proba.ndim == 1 else y_proba[:, 1]
                class_order = np.asarray(classes if classes is not None else np.unique(y_true))
                if len(class_order) != 2:
                    raise ValueError("Metrik probabilitas biner membutuhkan dua kelas dan urutan kolom kelas yang jelas")
                binary_true = np.asarray(y_true) == class_order[1]
                out["roc_auc"] = m.roc_auc_score(binary_true, p)
                out["pr_auc"] = m.average_precision_score(binary_true, p)
            else:
                out["roc_auc_ovr"] = m.roc_auc_score(y_true, y_proba, multi_class="ovr", labels=classes)
            out["log_loss"] = m.log_loss(y_true, y_proba, labels=classes)
        except ValueError as ex:  # mis. kelas tidak lengkap di valid
            out["proba_metrics_error"] = str(ex)
    return {k: (float(v) if not isinstance(v, str) else v) for k, v in out.items()}


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn import metrics as m
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "rmse": float(np.sqrt(m.mean_squared_error(y_true, y_pred))),
        "mae": float(m.mean_absolute_error(y_true, y_pred)),
        "r2": float(m.r2_score(y_true, y_pred)),
        "median_ae": float(m.median_absolute_error(y_true, y_pred)),
    }
    if np.all(y_true >= 0) and np.all(y_pred >= 0):
        out["rmsle"] = float(np.sqrt(m.mean_squared_log_error(y_true, y_pred)))
    nz = y_true != 0
    if nz.any():
        out["mape"] = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])))
    return out


# ---------------------------------------------------------------- plot
def plot_confusion(y_true, y_pred, labels=None, path=None, normalize="true"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, normalize=normalize,
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix" + (f" (normalisasi: {normalize})" if normalize else ""))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_regression(y_true, y_pred, path=None):
    import matplotlib.pyplot as plt
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_true, y_pred, s=8, alpha=.5)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set(xlabel="Aktual", ylabel="Prediksi", title="Prediksi vs aktual")
    axes[1].hist(y_true - y_pred, bins=40)
    axes[1].set(xlabel="Residual (aktual - prediksi)", title="Distribusi residual")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_importance(names, values, top=20, path=None, title="Feature importance"):
    import matplotlib.pyplot as plt
    s = pd.Series(values, index=names).sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, max(2.5, .3 * len(s))))
    s.plot.barh(ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


# ---------------------------------------------------------------- simpan
def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")


def save_model(model, path):
    import joblib
    joblib.dump(model, path)
    return path


def load_model(path):
    import joblib
    return joblib.load(path)


def make_submission(ids, preds, id_col: str, target_col: str, path):
    """Submission dua kolom (ID, target). Sesuaikan nama kolom dengan sample_submission lomba."""
    sub = pd.DataFrame({id_col: ids, target_col: preds})
    sub.to_csv(path, index=False)
    return sub


def show_table(d: dict, title: str = ""):
    """Tampilkan dict metrik sebagai tabel rapi."""
    df = pd.DataFrame({"nilai": d}).round(4)
    if title:
        print(title)
    return df
