
from __future__ import annotations

import base64
import gc
import html
import io
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
import importlib.metadata as importlib_metadata
import importlib.util
import itertools
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

# =====================================================================
# MODEL PATHS - EDIT PATH DI BAGIAN INI SAJA
# =====================================================================
# Delapan artefak model/tugas = 4 pendekatan x 2 tugas (issue, severity).
# Path boleh menunjuk ke FOLDER model (disarankan) atau ke file bobot di dalam folder itu
# (mis. .../issue/model.safetensors); keduanya dibaca sebagai folder.
# Untuk lokasi lain di Windows, pakai awalan r"...":
#     "issue_model": Path(r"D:\Skripsi\MODEL\indobert\final_issue_model"),
# Folder Transformer harus berisi: config.json, model.safetensors (atau pytorch_model.bin),
# tokenizer_config.json, dan tokenizer.json / vocab.txt / sentencepiece.bpe.model.
# Nama file bobot harus persis "model.safetensors" (dengan huruf 's' di akhir).
MODEL_PATHS: dict[str, dict[str, Path]] = {
    "svm_tfidf": {
        "issue_model": ROOT / "model" / "svm_tfidf" / "svm_tfidf_issue_model.joblib",
        "severity_model": ROOT / "model" / "svm_tfidf" / "svm_tfidf_severity_percat_model.joblib",
        "vectorizer": ROOT / "model" / "svm_tfidf" / "svm_tfidf_vectorizer.joblib",
        "issue_label_binarizer": ROOT / "model" / "svm_tfidf" / "svm_tfidf_issue_label_binarizer.joblib",
    },
    "indobert": {
        "issue_model": ROOT / "model" / "indobert" / "issue",
        "severity_model": ROOT / "model" / "indobert" / "severity",
    },
    "xlm_roberta": {
        "issue_model": ROOT / "model" / "xlmroberta" / "issue",
        "severity_model": ROOT / "model" / "xlmroberta" / "severity",
    },
    "mbert": {
        "issue_model": ROOT / "model" / "mbert" / "issue",
        "severity_model": ROOT / "model" / "mbert" / "severity",
    },
}

# pergantian model repos
# Model Transformer (IndoBERT, XLM-R, mBERT) diambil dari Hugging Face Hub karena file bobotnya
# >100 MB (batas GitHub). Format: model_key -> {task -> "username/nama-repo"}.
# Bila folder lokal model/<nama>/<task> ada (mis. saat jalan di laptop), folder lokal dipakai lebih dulu.
# Pastikan nama repo di bawah SAMA PERSIS dengan repo di akun Hugging Face-mu dan statusnya Public.
HF_REPOS: dict[str, dict[str, str]] = {
    "indobert": {
        "issue_model": "Darrell2005/indobert-issue-alfagift",
        "severity_model": "Darrell2005/indobert-severity-alfagift",
    },
    "xlm_roberta": {
        "issue_model": "Darrell2005/xlmr-issue-alfagift",
        "severity_model": "Darrell2005/xlmr-severity-alfagift",
    },
    "mbert": {
        "issue_model": "Darrell2005/mbert-issue-alfagift",
        "severity_model": "Darrell2005/mbert-severity-alfagift",
    },
}


# pergantian model repos
# Ambil nama repo Hugging Face untuk satu model/tugas. None bila folder lokal tersedia atau repo tidak diatur.
def hf_repo_for(model_key: str, task: str) -> str | None:
    local = Path(MODEL_PATHS[model_key][f"{task}_model"])
    if local.is_dir() and (local / "config.json").is_file():
        return None  # folder lokal ada -> pakai lokal
    return HF_REPOS.get(model_key, {}).get(f"{task}_model")


# pergantian model repos
# Unduh HANYA config.json (kecil) untuk cek status tanpa mengunduh bobot 1 GB.
# Mengembalikan (folder_berisi_config, pesan_error).
@st.cache_data(show_spinner=False, ttl=3600)
def hf_config_folder(repo: str) -> tuple[str, str]:
    try:
        from huggingface_hub import hf_hub_download

        cfg = hf_hub_download(repo_id=repo, filename="config.json")
        return str(Path(cfg).parent), ""
    except Exception as exc:  # noqa: BLE001
        return "", f"Repo Hugging Face `{repo}` tidak bisa diakses: {type(exc).__name__}"


# pergantian model repos
# Ukuran model.safetensors di Hugging Face (tanpa mengunduh).
@st.cache_data(show_spinner=False, ttl=3600)
def hf_weights_size(repo: str) -> int | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().get_paths_info(repo, ["model.safetensors"])
        return int(info[0].size) if info and info[0].size else None
    except Exception:  # noqa: BLE001
        return None


# Folder hasil eksperimen (hasil XLM-R ada di results/xlmr).
RESULT_PATHS: dict[str, Path] = {
    "svm_tfidf": ROOT / "results" / "svm_tfidf",
    "indobert": ROOT / "results" / "indobert",
    "xlm_roberta": ROOT / "results" / "xlmr",
    "mbert": ROOT / "results" / "mbert",
}

# Korpus ulasan berlabel (hanya dibaca di halaman Eksplorasi Dataset / tombol contoh acak).
DATASET_PATH = ROOT / "alfagift_issue_labeled_merged.csv"

# Database SQLite permanen untuk Form Tindak Lanjut Manual & Riwayat Review.
# Dibuat otomatis bila belum ada; TIDAK pernah dihapus/dibuat ulang saat aplikasi dijalankan.
FOLLOWUP_DB_PATH = ROOT / "data" / "alfagift_tindak_lanjut.db"
# =====================================================================

MODEL_ORDER = ["indobert", "xlm_roberta", "mbert", "svm_tfidf"]
MODEL_NAMES = {
    "indobert": "IndoBERT",
    "xlm_roberta": "XLM-RoBERTa",
    "mbert": "mBERT",
    "svm_tfidf": "Linear SVM",
}
MODEL_SUBTITLES = {
    "indobert": "Transformer • IndoBERT",
    "xlm_roberta": "Transformer • XLM-RoBERTa",
    "mbert": "Transformer • Multilingual BERT",
    "svm_tfidf": "Machine Learning Baseline • TF-IDF",
}
MODEL_COLORS = {
    "indobert": "#aa0015",
    "xlm_roberta": "#475569",
    "mbert": "#5b53f1",
    "svm_tfidf": "#2563eb",
}
TRANSFORMER_MODELS = ("indobert", "xlm_roberta", "mbert")

# Urutan label sesuai notebook training Transformer (cadangan bila config.json tidak memuat
# nama label; bila ada, urutan di config.json yang dipakai).
TRANSFORMER_ISSUE_LABELS = [
    "ACCOUNT",
    "INSTALL_UPDATE",
    "LOCATION",
    "ORDER",
    "OTHER",
    "PAYMENT",
    "RECEIPT_NOTIF",
    "SYSTEM",
    "UI_FEATURE",
    "VOUCHER",
]
ISSUE_DISPLAY = {
    "ACCOUNT": "Akun & Login",
    "INSTALL_UPDATE": "Instal & Update",
    "LOCATION": "Lokasi & Toko",
    "ORDER": "Pesanan & Pengiriman",
    "OTHER": "Lainnya",
    "PAYMENT": "Pembayaran",
    "RECEIPT_NOTIF": "Struk & Notifikasi",
    "SYSTEM": "Sistem & Bug",
    "UI_FEATURE": "Tampilan & Fitur",
    "VOUCHER": "Promo & Voucher",
}
SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "BLOCKER"]  # LOW < MEDIUM < HIGH < BLOCKER
SEVERITY_RANK = {label: i for i, label in enumerate(SEVERITY_LABELS)}
SEVERITY_COLORS = {
    "LOW": {"bg": "#ecfdf5", "text": "#047857", "border": "#a7f3d0", "dot": "#10b981"},
    "MEDIUM": {"bg": "#fffbeb", "text": "#b45309", "border": "#fde68a", "dot": "#f59e0b"},
    "HIGH": {"bg": "#fff1f2", "text": "#be123c", "border": "#fecdd3", "dot": "#e11d48"},
    "BLOCKER": {"bg": "#aa0015", "text": "#ffffff", "border": "#880010", "dot": "#ffffff"},
}

DEFAULT_MAX_LENGTH = 128  # sama dengan notebook training (juga dibaca dari summary)
DEFAULT_THRESHOLD = 0.5  # threshold sigmoid yang dipakai notebook saat evaluasi
NO_LABEL_TEXT = "(tidak ada label)"
NA_TEXT = "Belum tersedia"

TOKENIZER_VOCAB_FILES = ("tokenizer.json", "vocab.txt", "sentencepiece.bpe.model")
SVM_TASK_FILES = {
    "issue": ("issue_model", "vectorizer", "issue_label_binarizer"),
    "severity": ("severity_model", "vectorizer"),
}
SVM_ARTIFACT_LABELS = {
    "issue_model": "Model issue (.joblib)",
    "severity_model": "Model severity (.joblib)",
    "vectorizer": "TF-IDF vectorizer (.joblib)",
    "issue_label_binarizer": "Issue label binarizer (.joblib)",
}

TEXT_MODE_RAW = "Ulasan mentah"
TEXT_MODE_NORMALIZED = "Sudah ternormalisasi (text_norm)"

# Metrik yang ditampilkan: (kunci kanonik huruf kecil, nama tampilan)
ISSUE_METRICS = [
    ("macro_f1", "Macro-F1"),
    ("micro_f1", "Micro-F1"),
    ("weighted_f1", "Weighted-F1"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("hamming_loss", "Hamming loss (lebih kecil lebih baik)"),
    ("subset_accuracy", "Subset accuracy"),
]
SEVERITY_METRICS = [
    ("macro_f1", "Macro-F1"),
    ("weighted_f1", "Weighted-F1"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("accuracy", "Accuracy"),
]


# =====================================================================
# Util dasar
# =====================================================================
def fmt_pct(value: float | None, digits: int = 2) -> str:
    return NA_TEXT if value is None or not np.isfinite(value) else f"{value * 100:.{digits}f}%"


def fmt_int(value: Any) -> str:
    """Pemisah ribuan gaya Indonesia: 1431 -> '1.431'."""
    return f"{int(value):,}".replace(",", ".")


# Waktu terakhir file diubah (dipakai agar cache ikut berubah bila file diganti).
def _mtime(path: Path | None) -> float:
    try:
        return Path(path).stat().st_mtime if path else 0.0
    except OSError:
        return 0.0


# Cek file ada dan tidak kosong.
def _is_nonempty_file(path: Path) -> bool:
    try:
        return Path(path).is_file() and Path(path).stat().st_size > 0
    except OSError:
        return False


# Baca file JSON; kalau gagal kembalikan None.
def _json_load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


# Cek nilai kosong (None atau NaN).
def _is_empty_value(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def parse_label_list(value: Any) -> list[str]:
    """Ubah 'A | B' menjadi ['A', 'B']; nilai kosong/'(tidak ada ...)' menjadi []."""
    if _is_empty_value(value):
        return []
    text = str(value).strip()
    if not text or text.startswith("("):
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


# Ubah kolom teks 'True'/'False' menjadi boolean.
def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


# Ambil nama kolom pertama yang ada di tabel dari daftar kandidat.
def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


# Tampilan yang kompatibel lintas versi Streamlit (use_container_width -> width="stretch").
def show_df(data: Any, **kwargs: Any) -> None:
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:  # Streamlit lama tidak mengenal width="stretch"
        st.dataframe(data, use_container_width=True, **kwargs)


# Tombol selebar kolom; ada cadangan untuk Streamlit versi lama.
def show_button(label: str, **kwargs: Any) -> bool:
    try:
        return st.button(label, width="stretch", **kwargs)
    except Exception:
        return st.button(label, use_container_width=True, **kwargs)


# =====================================================================
# Preprocessing  (TODO — belum terverifikasi)
# =====================================================================
def normalize_review_text(raw_text: str) -> str:
    """Normalisasi ulasan mentah -> ``text_norm``.

    TODO(skripsi): Fungsi normalisasi asli yang menghasilkan kolom ``text_norm`` TIDAK ada
    di notebook SVM, IndoBERT, mBERT, maupun XLM-R. Notebook hanya membaca kolom
    ``text_norm`` yang sudah jadi dari ``alfagift_issue_labeled_merged.csv`` (kolom
    ``review`` -> ``text_clean`` -> ``text_norm``). Dari contoh data terlihat normalisasi
    memakai kamus slang dan aturan sufiks (mis. "gak"->"tidak", "klo"->"kalau",
    "aplikasinya"->"aplikasi") yang tidak bisa ditebak dengan benar, jadi sengaja TIDAK
    ditiru di sini.

    Isi fungsi ini dengan kode preprocessing yang persis sama dengan yang dipakai saat
    membuat ``text_norm``. Sampai itu dilakukan, fungsi hanya membuang spasi di tepi,
    dan UI menampilkan peringatan untuk input "Ulasan mentah".
    """
    return raw_text.strip()


NORMALIZER_VERIFIED = False  # ubah ke True HANYA setelah normalize_review_text diverifikasi


# =====================================================================
# Hasil evaluasi: summary, report, confusion matrix, prediksi
# =====================================================================
def extract_test_metrics(summary: dict[str, Any] | None, task: str) -> dict[str, float]:
    """Ambil metrik test untuk satu tugas dari berbagai format summary.

    Mendukung ``test_metrics_<task>`` (Transformer) dan ``<task>_test_metrics`` (SVM),
    serta menyamakan huruf besar/kecil kunci (``macro_F1`` -> ``macro_f1``).
    """
    if not summary:
        return {}
    block = None
    for key in (f"test_metrics_{task}", f"{task}_test_metrics"):
        if isinstance(summary.get(key), dict):
            block = summary[key]
            break
    if block is None:
        return {}
    metrics: dict[str, float] = {}
    for name, value in block.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            metrics[str(name).lower()] = number
    return metrics


# Cari file summary .json di folder hasil model.
def find_summary(model_key: str) -> tuple[Path | None, dict[str, Any] | None]:
    folder = RESULT_PATHS[model_key]
    if not folder.is_dir():
        return None, None
    candidates = sorted(folder.glob("*summary*.json")) + sorted(
        p for p in folder.rglob("*summary*.json") if p.parent != folder
    )
    # Utamakan nama "*experiment_summary*.json"
    candidates.sort(key=lambda p: ("experiment_summary" not in p.name.lower(), str(p)))
    for path in candidates:
        content = _json_load(path)
        if content and (extract_test_metrics(content, "issue") or extract_test_metrics(content, "severity")):
            return path, content
    return None, None


# Ambil satu nilai metrik dari summary.
def metric_value(summary: dict[str, Any] | None, task: str, metric: str) -> float | None:
    return extract_test_metrics(summary, task).get(metric)


# Ambil nilai dari kunci pertama yang ada di summary.
def _nested(summary: dict[str, Any] | None, *keys: str) -> Any:
    if not summary:
        return None
    for key in keys:
        if key in summary:
            return summary[key]
    return None


def split_counts(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Jumlah data per split dari summary (nama kunci berbeda antar notebook)."""
    review = _nested(summary, "split_review_counts", "split_counts_review_level", "split_sizes")
    severity = _nested(summary, "split_severity_annotation_counts", "split_counts_severity_annotations")
    return {
        "review": review if isinstance(review, dict) else {},
        "severity": severity if isinstance(severity, dict) else (review if isinstance(review, dict) else {}),
    }


def result_file(model_key: str, task: str, kind: str) -> Path | None:
    """Cari file hasil. ``kind`` = classification_report | confusion_matrix |
    test_predictions | error_analysis. Mencoba pola per-tugas dulu, lalu pola umum."""
    folder = RESULT_PATHS[model_key]
    if not folder.is_dir():
        return None
    per_task = sorted(folder.glob(f"*{task}*{kind}*.csv"))
    if per_task:
        return per_task[0]
    general = sorted(
        p for p in folder.glob(f"*{kind}*.csv") if "issue" not in p.name.lower() and "severity" not in p.name.lower()
    )
    return general[0] if general else None


# Baca CSV dengan cache; mtime hanya untuk kunci cache.
@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:  # noqa: ARG001 (mtime = cache key)
    return pd.read_csv(path_str, encoding="utf-8-sig")


# Baca CSV (cache otomatis diperbarui bila file berubah).
def read_csv(path: Path) -> pd.DataFrame:
    return _read_csv_cached(str(path), _mtime(path))


# ---- Classification report ------------------------------------------------
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def build_report_from_indicators(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], multiclass: bool
) -> pd.DataFrame:
    """Hitung classification report (definisi sama dengan sklearn, zero_division=0)."""
    rows, tps, fps, fns = [], 0, 0, 0
    for j, label in enumerate(labels):
        t, p = y_true[:, j] == 1, y_pred[:, j] == 1
        tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
        precision, recall, f1 = _prf(tp, fp, fn)
        rows.append(
            {"label": label, "precision": precision, "recall": recall, "f1-score": f1, "support": float(tp + fn)}
        )
        tps, fps, fns = tps + tp, fps + fp, fns + fn
    per_label = pd.DataFrame(rows)
    total = per_label["support"].sum()
    weights = per_label["support"] / total if total else per_label["support"] * 0.0
    averages = []
    if multiclass:
        n = float(len(y_true))
        accuracy = float((y_true == y_pred).all(axis=1).mean()) if n else 0.0
        averages.append(
            {"label": "accuracy", "precision": accuracy, "recall": accuracy, "f1-score": accuracy, "support": n}
        )
    else:
        mp, mr, mf = _prf(tps, fps, fns)
        averages.append({"label": "micro avg", "precision": mp, "recall": mr, "f1-score": mf, "support": total})
    averages.append(
        {
            "label": "macro avg",
            **{c: per_label[c].mean() for c in ("precision", "recall", "f1-score")},
            "support": total,
        }
    )
    averages.append(
        {
            "label": "weighted avg",
            **{c: float((per_label[c] * weights).sum()) for c in ("precision", "recall", "f1-score")},
            "support": total,
        }
    )
    return pd.concat([per_label, pd.DataFrame(averages)], ignore_index=True)


# Ubah daftar label menjadi matriks 0/1.
def _indicators(label_lists: list[list[str]], labels: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(label_lists), len(labels)), dtype=int)
    for r, items in enumerate(label_lists):
        for item in items:
            if item in index:
                matrix[r, index[item]] = 1
    return matrix


# Ambil daftar label issue SVM dari summary dan file prediksi.
def svm_issue_labels(summary: dict[str, Any] | None, predictions: pd.DataFrame | None = None) -> list[str]:
    labels = list(summary.get("issue_labels", [])) if summary else []
    if predictions is not None:
        seen = set()
        for col in ("issue_true", "issue_pred"):
            if col in predictions.columns:
                for value in predictions[col]:
                    seen.update(parse_label_list(value))
        labels += sorted(seen - set(labels))
    return labels


def load_classification_report(model_key: str, task: str) -> tuple[Path | None, pd.DataFrame | None, str]:
    """Kembalikan (sumber, tabel, catatan). Untuk SVM, notebook tidak menyimpan CSV report,
    sehingga dihitung ulang dari file test predictions asli dan dicocokkan dengan summary."""
    path = result_file(model_key, task, "classification_report")
    if path is not None:
        try:
            df = read_csv(path).copy()
            df = df.rename(columns={df.columns[0]: "label"})
            # Pada CSV hasil sklearn.to_csv, baris "accuracy" menyimpan nilai akurasi di kolom
            # support (bukan jumlah data). Ganti dengan total support dari baris "macro avg".
            if {"label", "support"}.issubset(df.columns) and (df["label"] == "macro avg").any():
                total = df.loc[df["label"] == "macro avg", "support"].iloc[0]
                df.loc[df["label"] == "accuracy", "support"] = total
            return path, df, ""
        except Exception as exc:  # noqa: BLE001
            return path, None, f"Gagal membaca {path.name}: {exc}"

    pred_path = result_file(model_key, task, "test_predictions")
    if model_key == "svm_tfidf" and pred_path is not None:
        try:
            raw = read_csv(pred_path)
            _, summary = find_summary(model_key)
            if task == "issue":
                labels = svm_issue_labels(summary, raw)
                y_true = _indicators([parse_label_list(v) for v in raw["issue_true"]], labels)
                y_pred = _indicators([parse_label_list(v) for v in raw["issue_pred"]], labels)
                report = build_report_from_indicators(y_true, y_pred, labels, multiclass=False)
            else:
                labels = list(SEVERITY_LABELS)
                y_true = _indicators([[v] for v in raw["severity_true"].astype(str)], labels)
                y_pred = _indicators([[v] for v in raw["severity_pred"].astype(str)], labels)
                report = build_report_from_indicators(y_true, y_pred, labels, multiclass=True)
            note = (
                f"Notebook SVM tidak menyimpan CSV classification report. Tabel ini dihitung ulang dari "
                f"`{pred_path.name}` ({fmt_int(len(raw))} baris uji)."
            )
            reported = metric_value(summary, task, "macro_f1")
            computed = float(report.loc[report["label"] == "macro avg", "f1-score"].iloc[0])
            if reported is not None:
                ok = abs(reported - computed) < 1e-6
                note += f" Macro-F1 hasil hitung ulang {computed:.4f} " + (
                    "sama dengan nilai di summary."
                    if ok
                    else f"BERBEDA dari summary ({reported:.4f}); periksa kembali berkas hasil."
                )
            return pred_path, report, note
        except Exception as exc:  # noqa: BLE001
            return pred_path, None, f"Gagal menghitung report SVM: {exc}"
    return None, None, ""


def load_confusion_matrix(model_key: str) -> tuple[Path | None, pd.DataFrame | None, str]:
    """Confusion matrix severity (baris = aktual, kolom = prediksi)."""
    path = result_file(model_key, "severity", "confusion_matrix")
    if path is not None:
        try:
            df = read_csv(path)
            df = df.set_index(df.columns[0])
            df.index = [str(i).replace("actual_", "") for i in df.index]
            df.columns = [str(c).replace("predicted_", "") for c in df.columns]
            labels = [l for l in SEVERITY_LABELS if l in df.index and l in df.columns]
            return path, df.loc[labels, labels].astype(int), ""
        except Exception as exc:  # noqa: BLE001
            return path, None, f"Gagal membaca {path.name}: {exc}"
    pred_path = result_file(model_key, "severity", "test_predictions")
    if model_key == "svm_tfidf" and pred_path is not None:
        try:
            raw = read_csv(pred_path)
            cm = pd.crosstab(raw["severity_true"].astype(str), raw["severity_pred"].astype(str))
            cm = cm.reindex(index=SEVERITY_LABELS, columns=SEVERITY_LABELS, fill_value=0).astype(int)
            return (
                pred_path,
                cm,
                (
                    f"Notebook SVM tidak menyimpan CSV confusion matrix; tabel ini dihitung ulang "
                    f"dari `{pred_path.name}`."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return pred_path, None, f"Gagal menghitung confusion matrix SVM: {exc}"
    return None, None, ""


# Cari gambar confusion matrix severity di folder hasil.
def confusion_png(model_key: str) -> Path | None:
    folder = RESULT_PATHS[model_key]
    if not folder.is_dir():
        return None
    found = sorted(folder.glob("*severity*confusion*.png"))
    return found[0] if found else None


# ---- Prediksi uji & error analysis ----------------------------------------
_ISSUE_COLS = {
    "text": ["text_input", "text_used_by_model", "text_norm"],
    "true": ["true_issue_labels", "true_issues", "issue_true"],
    "pred": ["predicted_issue_labels", "predicted_issues", "issue_pred"],
    "ok": ["exact_match", "issue_exact_match"],
}
_SEVERITY_COLS = {
    "text": ["text_norm", "text_used_by_model", "text_input"],
    "category": ["category_Issue"],
    "true": ["true_severity", "severity_Issue", "severity_true"],
    "pred": ["predicted_severity", "severity_pred"],
    "ok": ["correct", "severity_correct"],
}


def unify_predictions(raw: pd.DataFrame, task: str) -> pd.DataFrame:
    """Samakan nama kolom antar model menjadi: review_id, teks, [kategori], label, prediksi, benar."""
    spec = _ISSUE_COLS if task == "issue" else _SEVERITY_COLS
    cols = {field: _first_col(raw, names) for field, names in spec.items()}
    out = pd.DataFrame(index=raw.index)
    out["review_id"] = raw["review_id"] if "review_id" in raw.columns else ""
    if "review" in raw.columns:
        out["ulasan_mentah"] = raw["review"]
    out["teks"] = raw[cols["text"]] if cols["text"] else ""
    if task == "severity" and cols.get("category"):
        out["kategori"] = raw[cols["category"]]
    out["label_sebenarnya"] = raw[cols["true"]] if cols["true"] else ""
    pred = raw[cols["pred"]] if cols["pred"] else pd.Series([None] * len(raw), index=raw.index)
    out["label_prediksi"] = pred.map(
        lambda v: NO_LABEL_TEXT if _is_empty_value(v) or not str(v).strip() or str(v).strip().startswith("(") else v
    )
    if cols["ok"]:
        out["benar"] = _to_bool(raw[cols["ok"]])
    elif task == "issue":
        out["benar"] = [
            set(parse_label_list(a)) == set(parse_label_list(b)) for a, b in zip(out["label_sebenarnya"], pred)
        ]
    else:
        out["benar"] = out["label_sebenarnya"].astype(str) == out["label_prediksi"].astype(str)
    return out


def load_predictions(model_key: str, task: str, kind: str) -> tuple[Path | None, pd.DataFrame | None]:
    """kind = test_predictions | error_analysis. Error analysis memakai berkas aslinya bila ada;
    bila tidak ada, disaring dari test predictions."""
    path = result_file(model_key, task, kind)
    if path is None and kind == "error_analysis":
        base_path = result_file(model_key, task, "test_predictions")
        if base_path is None:
            return None, None
        df = unify_predictions(read_csv(base_path), task)
        return base_path, df[~df["benar"]].reset_index(drop=True)
    if path is None:
        return None, None
    try:
        df = unify_predictions(read_csv(path), task)
    except Exception:  # noqa: BLE001
        return path, None
    if kind == "error_analysis":  # berkas error SVM memuat kesalahan dari kedua tugas
        df = df[~df["benar"]].reset_index(drop=True)
    return path, df


def per_label_f1_table(task: str) -> pd.DataFrame:
    """F1 per label untuk keempat model (dari classification report masing-masing)."""
    columns = {}
    for key, name in MODEL_NAMES.items():
        _, report, _ = load_classification_report(key, task)
        if report is None or "f1-score" not in report.columns:
            continue
        columns[name] = report.set_index("label")["f1-score"]
    if not columns:
        return pd.DataFrame()
    table = pd.DataFrame(columns)
    order = TRANSFORMER_ISSUE_LABELS if task == "issue" else SEVERITY_LABELS
    return table.reindex([l for l in order if l in table.index])


def macro_f1_without_other(model_key: str) -> float | None:
    """Macro-F1 issue tanpa label OTHER, dihitung dari classification report (turunan).
    Dipakai untuk membandingkan SVM dan Transformer pada kategori yang sama."""
    _, report, _ = load_classification_report(model_key, "issue")
    if report is None:
        return None
    labels = [l for l in report["label"] if l in TRANSFORMER_ISSUE_LABELS and l != "OTHER"]
    if not labels:
        return None
    return float(report.set_index("label").loc[labels, "f1-score"].mean())


def test_set_overlap() -> dict[str, Any] | None:
    """Irisan review_id antara test set SVM dan test set Transformer (fakta dari berkas hasil)."""

    # Kumpulkan review_id dari file prediksi uji sebuah model.
    def ids(model_key: str, task: str = "issue") -> set[str] | None:
        path = result_file(model_key, task, "test_predictions")
        if path is None:
            return None
        df = read_csv(path)
        return set(df["review_id"].astype(str)) if "review_id" in df.columns else None

    svm_ids = ids("svm_tfidf")
    transformer_ids = {k: ids(k) for k in TRANSFORMER_MODELS}
    available = {k: v for k, v in transformer_ids.items() if v is not None}
    if svm_ids is None or not available:
        return None
    first = next(iter(available.values()))
    return {
        "svm_n": len(svm_ids),
        "transformer_n": len(first),
        "overlap_n": len(svm_ids & first),
        "transformers_identical": all(v == first for v in available.values()),
        "n_transformers_compared": len(available),
    }


# =====================================================================
# Status artefak (tanpa memuat bobot)
# =====================================================================
def model_dir(model_key: str, task: str) -> tuple[Path, str]:
    """Folder model Transformer dari MODEL_PATHS, plus catatan bila path berupa file.

    MODEL_PATHS seharusnya menunjuk ke FOLDER (berisi config.json, model.safetensors, tokenizer...).
    Bila yang diisi adalah path file (mis. .../issue/model.safetensors, termasuk salah ketik
    ``model.safetensor``), folder induknya dipakai dan diberi catatan; nama file bobot yang
    benar tetap harus ada di folder itu.
    """
    path = Path(MODEL_PATHS[model_key][f"{task}_model"])
    if path.is_dir():
        return path, ""
    looks_like_file = path.is_file() or (path.suffix != "" and path.name.lower().startswith(("model", "pytorch_model")))
    if looks_like_file:
        return path.parent, (
            f"Path di MODEL_PATHS menunjuk ke file `{path.name}`; yang dibaca adalah folder "
            f"induknya `{path.parent}`. Isi MODEL_PATHS dengan FOLDER model, bukan nama file."
        )
    return path, ""


# Cek file bobot Transformer ada di folder model.
def _weights_present(folder: Path) -> bool:
    if any(_is_nonempty_file(folder / n) for n in ("model.safetensors", "pytorch_model.bin")):
        return True
    if _is_nonempty_file(folder / "model.safetensors.index.json") and any(folder.glob("model-*.safetensors")):
        return True
    return _is_nonempty_file(folder / "pytorch_model.bin.index.json") and any(folder.glob("pytorch_model-*.bin"))


def resolve_labels(config: dict[str, Any], expected: list[str], task: str) -> tuple[list[str], list[str]]:
    """Ambil urutan label dari config.json (id2label). Mengembalikan (label, peringatan).

    * jumlah output harus sama dengan ``expected`` (10 issue / 4 severity), jika tidak -> ValueError
    * bila nama label generik (LABEL_0, ...) atau tidak ada, pakai urutan notebook + peringatan
    * bila nama ada tetapi himpunannya berbeda dari yang diharapkan -> ValueError
    """
    id2label = config.get("id2label") or {}
    num = len(id2label) if id2label else config.get("num_labels")
    if num != len(expected):
        raise ValueError(
            f"config.json menyatakan {num} output untuk model {task}, seharusnya {len(expected)} "
            f"({', '.join(expected)}). Folder ini kemungkinan bukan model {task} yang benar."
        )
    try:
        names = [str(id2label[str(i)] if str(i) in id2label else id2label[i]) for i in range(len(expected))]
    except (KeyError, TypeError):
        names = []
    if not names or all(re.fullmatch(r"LABEL_\d+", n) for n in names):
        return list(expected), [
            f"config.json tidak memuat nama label {task}; urutan dari notebook training dipakai "
            f"({', '.join(expected)}). Pastikan ini sesuai model yang dilatih."
        ]
    if set(names) != set(expected):
        raise ValueError(
            f"Nama label di config.json ({', '.join(names)}) tidak cocok dengan label {task} yang diharapkan "
            f"({', '.join(expected)})."
        )
    return names, []


# Periksa config.json model dan cocokkan label-nya.
def inspect_transformer_config(folder: Path, task: str) -> dict[str, Any]:
    expected = TRANSFORMER_ISSUE_LABELS if task == "issue" else SEVERITY_LABELS
    config = _json_load(Path(folder) / "config.json")
    if config is None:
        return {"ok": False, "labels": [], "warnings": [], "error": "config.json tidak ada atau tidak valid."}
    try:
        labels, warns = resolve_labels(config, expected, task)
    except ValueError as exc:
        return {
            "ok": False,
            "labels": [],
            "warnings": [],
            "error": str(exc),
            "problem_type": config.get("problem_type"),
            "model_type": config.get("model_type"),
        }
    return {
        "ok": True,
        "labels": labels,
        "warnings": warns,
        "error": "",
        "problem_type": config.get("problem_type"),
        "model_type": config.get("model_type"),
    }


# Cek apakah file model untuk satu tugas (issue/severity) sudah lengkap.
def task_status(model_key: str, task: str) -> dict[str, Any]:
    paths = MODEL_PATHS[model_key]
    if model_key == "svm_tfidf":
        keys = SVM_TASK_FILES[task]
        missing = [Path(paths[k]).name for k in keys if not _is_nonempty_file(Path(paths[k]))]
        required = [str(paths[k]) for k in keys]
        main_key = "issue_model" if task == "issue" else "severity_model"
        return {
            "ready": not missing,
            "missing": missing,
            "problems": [],
            "warnings": [],
            "location": str(Path(paths[main_key]).parent),
            "required": required,
            "labels": [],
        }

    # pergantian model repos
    # Bila tidak ada folder lokal tetapi ada repo Hugging Face, cek status lewat config.json saja.
    repo = hf_repo_for(model_key, task)
    if repo is not None:
        cfg_folder, err = hf_config_folder(repo)
        if err:
            return {
                "ready": False,
                "missing": ["repo Hugging Face tidak bisa diakses"],
                "problems": [err],
                "warnings": [],
                "location": f"hf://{repo}",
                "required": [repo],
                "labels": [],
            }
        info = inspect_transformer_config(Path(cfg_folder), task)
        return {
            "ready": bool(info["ok"]),
            "missing": [],
            "problems": [] if info["ok"] else [info["error"]],
            "warnings": info["warnings"],
            "location": f"hf://{repo}",
            "required": [repo],
            "labels": info["labels"],
        }

    folder, path_note = model_dir(model_key, task)
    if not folder.is_dir():
        return {
            "ready": False,
            "missing": ["folder model tidak ditemukan"],
            "problems": [],
            "warnings": [path_note] if path_note else [],
            "location": str(folder),
            "required": [str(folder)],
            "labels": [],
        }
    missing = []
    if not _is_nonempty_file(folder / "config.json"):
        missing.append("config.json")
    if not _weights_present(folder):
        missing.append("model.safetensors (atau pytorch_model.bin)")  # nama file persis, dengan huruf 's' di akhir
    if not _is_nonempty_file(folder / "tokenizer_config.json"):
        missing.append("tokenizer_config.json")
    if not any(_is_nonempty_file(folder / n) for n in TOKENIZER_VOCAB_FILES):
        missing.append("tokenizer.json / vocab.txt / sentencepiece.bpe.model")
    problems, warns, labels = [], ([path_note] if path_note else []), []
    if "config.json" not in missing:
        info = inspect_transformer_config(folder, task)
        if info["ok"]:
            labels, warns = info["labels"], warns + info["warnings"]
        else:
            problems.append(info["error"])
    return {
        "ready": not missing and not problems,
        "missing": missing,
        "problems": problems,
        "warnings": warns,
        "location": str(folder),
        "required": [str(folder)],
        "labels": labels,
    }


# Status model: siap bila issue dan severity sama-sama lengkap.
def model_status(model_key: str) -> dict[str, Any]:
    tasks = {task: task_status(model_key, task) for task in ("issue", "severity")}
    return {"tasks": tasks, "ready": all(t["ready"] for t in tasks.values())}


# =====================================================================
# Adapter inferensi SVM-TFIDF (mengikuti SVM_TFIDF_Alfagift.ipynb)
# =====================================================================
# Pipeline training SVM (bukan sama dengan Transformer):
#   * input  : kolom text_norm (sudah ternormalisasi), fitur TF-IDF 1-2 gram
#   * issue  : OneVsRest(LinearSVC) pada 10 kategori (termasuk OTHER); prediksi
#              multi-label = margin decision_function > 0 (tanpa fallback)
#   * severity: CategorySeverityModel (severity_percat.py) -> 4 kelas per pasangan
#              (review, kategori), kategori terprediksi menjadi input (termasuk OTHER)
def _apply_tfidf_compat(vectorizer: Any) -> bool:
    """Perbaiki TfidfVectorizer hasil pickle scikit-learn lama (mis. 1.4.2) di versi baru.

    Di scikit-learn >= 1.5, ``TfidfTransformer.transform`` hanya memakai bobot IDF bila ada
    atribut ``idf_``. Objek hasil pickle lama hanya menyimpan ``_idf_diag`` sehingga IDF
    DIAM-DIAM dilewati dan prediksi berubah tanpa error. Shim ini memulihkan ``idf_``.
    Hasilnya diverifikasi terhadap test predictions asli (lihat verify_svm_reproduction).
    """
    tfidf = getattr(vectorizer, "_tfidf", None)
    if tfidf is not None and not hasattr(tfidf, "idf_") and hasattr(tfidf, "_idf_diag"):
        tfidf.idf_ = np.asarray(tfidf._idf_diag.diagonal()).ravel()
        return True
    return False


# Muat semua file SVM sekali saja (di-cache) lalu periksa isinya.
@st.cache_resource(show_spinner=False)
def load_svm_bundle(
    issue_path: str, severity_path: str, vectorizer_path: str, binarizer_path: str, signature: str
) -> dict[str, Any]:  # noqa: ARG001 (signature = cache key)
    try:
        import joblib
        import sklearn
        import severity_percat  # noqa: F401  (diperlukan joblib untuk memuat CategorySeverityModel)
    except ImportError as exc:
        raise ImportError(
            "scikit-learn dan joblib diperlukan untuk SVM-TFIDF, dan severity_percat.py harus "
            "berada di folder yang sama dengan app.py: "
            "pip install scikit-learn joblib"
        ) from exc

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vectorizer = joblib.load(vectorizer_path)
        issue_model = joblib.load(issue_path)
        severity_model = joblib.load(severity_path)
        binarizer = joblib.load(binarizer_path)
    version_notes = sorted({str(w.message).split("\n")[0] for w in caught if "version" in str(w.message).lower()})
    trained_with = None
    for note in version_notes:
        match = re.search(r"from version (\S+) when using version (\S+)", note)
        if match:
            trained_with = match.group(1)
    shim_applied = _apply_tfidf_compat(vectorizer)

    issue_labels = [str(c) for c in binarizer.classes_]
    severity_classes = [str(c) for c in severity_model.classes_]
    severity_categories = [str(c) for c in (getattr(severity_model, "categories", None) or [])]
    n_features = len(vectorizer.vocabulary_)
    problems = []
    if len(getattr(issue_model, "estimators_", [])) != len(issue_labels):
        problems.append("Jumlah estimator model issue tidak sama dengan jumlah label di binarizer.")
    if any(est.coef_.shape[1] != n_features for est in getattr(issue_model, "estimators_", [])):
        problems.append("Dimensi fitur model issue tidak sama dengan vocabulary vectorizer.")
    if not severity_categories:
        problems.append(
            "Model severity bukan CategorySeverityModel (severity_percat.py) atau daftar kategorinya kosong."
        )
    if set(severity_classes) != set(SEVERITY_LABELS):
        problems.append(f"Kelas severity model ({', '.join(severity_classes)}) tidak sesuai {SEVERITY_LABELS}.")
    if problems:
        raise ValueError(" ".join(problems))
    return {
        "vectorizer": vectorizer,
        "issue_model": issue_model,
        "severity_model": severity_model,
        "issue_labels": issue_labels,
        "severity_classes": severity_classes,
        "severity_categories": severity_categories,
        "n_features": n_features,
        "sklearn_version": sklearn.__version__,
        "trained_with": trained_with,
        "version_notes": version_notes,
        "idf_shim_applied": shim_applied,
    }


# Ambil paket SVM; cache dimuat ulang bila file model berubah.
def get_svm_bundle() -> dict[str, Any]:
    paths = MODEL_PATHS["svm_tfidf"]
    files = [paths[k] for k in ("issue_model", "severity_model", "vectorizer", "issue_label_binarizer")]
    signature = "|".join(f"{p}:{_mtime(p)}" for p in files)
    return load_svm_bundle(str(files[0]), str(files[1]), str(files[2]), str(files[3]), signature)


# Prediksi issue (multi-label) lalu severity untuk tiap kategori terprediksi.
def predict_svm(bundle: dict[str, Any], text_norm: str) -> dict[str, Any]:
    X = bundle["vectorizer"].transform([text_norm])
    issue_model, severity_model = bundle["issue_model"], bundle["severity_model"]
    indicator = issue_model.predict(X)[0]
    margins = np.asarray(issue_model.decision_function(X))[0]
    labels = bundle["issue_labels"]
    predicted = [labels[i] for i, v in enumerate(indicator) if int(v) == 1]
    # Severity per kategori terprediksi (kategori yang tidak dikenal model severity dilewati).
    sev_cats = [c for c in predicted if c in bundle["severity_categories"]]
    severity_rows: list[dict[str, Any]] = []
    if sev_cats:
        import scipy.sparse as sp

        X_rep = sp.vstack([X] * len(sev_cats), format="csr")
        sev_labels, sev_margins = severity_model.predict_with_margin(X_rep, np.array(sev_cats))
        severity_rows = [
            {"category": c, "severity": str(l), "score": float(m)} for c, l, m in zip(sev_cats, sev_labels, sev_margins)
        ]
    return {
        "issue_predicted": predicted,
        "issue_scores": sorted(zip(labels, map(float, margins)), key=lambda x: x[1], reverse=True),
        "severity_rows": severity_rows,
        "severity": highest_severity([r["severity"] for r in severity_rows]),
        "n_active_features": int(X.nnz),
    }


@st.cache_data(show_spinner=False)
def verify_svm_reproduction(signature: str) -> dict[str, Any]:  # noqa: ARG001 (signature = cache key)
    """Prediksi ulang seluruh baris uji SVM dan bandingkan dengan file hasil eksperimen.

    Issue dibandingkan dengan svm_tfidf_test_predictions.csv; severity per kategori dibandingkan
    dengan svm_tfidf_severity_test_predictions.csv (satu baris per pasangan review-kategori).
    """
    path = result_file("svm_tfidf", "issue", "test_predictions")
    sev_path = result_file("svm_tfidf", "severity", "test_predictions")
    if path is None:
        return {"status": "no_reference", "message": "svm_tfidf_test_predictions.csv tidak ditemukan."}
    if sev_path is None:
        return {"status": "no_reference", "message": "svm_tfidf_severity_test_predictions.csv tidak ditemukan."}
    raw, sev_raw = read_csv(path), read_csv(sev_path)
    needed = {"review_id", "text_used_by_model", "issue_pred"}
    sev_needed = {"review_id", "text_used_by_model", "category_Issue", "severity_pred"}
    if not needed.issubset(raw.columns):
        return {"status": "no_reference", "message": f"Kolom {sorted(needed - set(raw.columns))} tidak ada."}
    if not sev_needed.issubset(sev_raw.columns):
        return {
            "status": "no_reference",
            "message": f"Kolom {sorted(sev_needed - set(sev_raw.columns))} tidak ada di file severity.",
        }
    bundle = get_svm_bundle()
    texts = raw["text_used_by_model"].fillna("").astype(str).tolist()
    X = bundle["vectorizer"].transform(texts)
    indicator = bundle["issue_model"].predict(X)
    labels = bundle["issue_labels"]
    issue_ok = [
        set(parse_label_list(saved)) == {labels[i] for i, v in enumerate(row) if int(v) == 1}
        for saved, row in zip(raw["issue_pred"], indicator)
    ]
    # severity: vektor TF-IDF dibentuk per pasangan dari teks yang tersimpan di file severity
    Xs = bundle["vectorizer"].transform(sev_raw["text_used_by_model"].fillna("").astype(str).tolist())
    sev_pred = bundle["severity_model"].predict(Xs, sev_raw["category_Issue"].astype(str).to_numpy())
    severity_ok = [str(a) == str(b) for a, b in zip(sev_raw["severity_pred"], sev_pred)]
    return {
        "status": "ok",
        "n": len(raw),
        "n_severity": len(sev_raw),
        "issue_match": int(sum(issue_ok)),
        "severity_match": int(sum(severity_ok)),
        "all_match": bool(all(issue_ok) and all(severity_ok)),
        "source": f"{path.name} & {sev_path.name}",
    }


# =====================================================================
# Adapter inferensi Transformer (mengikuti notebook IndoBERT/mBERT/XLM-R)
# =====================================================================
# Pipeline training Transformer (identik pada ketiga notebook):
#   * issue   : input = text_norm; 10 output; sigmoid; label terprediksi bila skor >= 0.5
#   * severity: input = "Ulasan: <text_norm> Kategori permasalahan: <kategori>"; 4 output; softmax
#               argmax; satu prediksi per pasangan (review, kategori); max_length = 128
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -30, 30)))


# Ubah skor mentah menjadi nilai 0-1 yang jumlahnya 1.
def softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# Susun teks input severity Transformer: ulasan + kategori.
def build_severity_text(text_norm: str, category: str) -> str:
    return f"Ulasan: {text_norm} Kategori permasalahan: {category}"


# Pilih kategori issue yang skornya di atas threshold.
def decide_issues(logits: np.ndarray, labels: list[str], threshold: float, fallback: bool) -> dict[str, Any]:
    scores = sigmoid(logits)
    ranked = sorted(zip(labels, map(float, scores)), key=lambda x: x[1], reverse=True)
    selected = [(l, s) for l, s in ranked if s >= threshold]
    used_fallback = False
    if not selected and fallback:
        selected, used_fallback = [ranked[0]], True
    return {"scores": ranked, "selected": selected, "used_fallback": used_fallback}


# Pilih severity dengan skor tertinggi untuk tiap kategori.
def decide_severity(logits: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    probs = softmax(np.atleast_2d(logits))
    rows = []
    for p in probs:
        best = int(np.argmax(p))
        rows.append(
            {
                "severity": labels[best],
                "score": float(p[best]),
                "scores": sorted(zip(labels, map(float, p)), key=lambda x: x[1], reverse=True),
            }
        )
    return rows


@st.cache_resource(show_spinner=False)
def _head_registry() -> dict[str, Any]:
    """Penyimpanan model Transformer yang sedang dimuat. Hanya SATU pendekatan yang
    disimpan pada satu waktu (ganti pendekatan => yang lama dibuang) agar RAM/VRAM aman."""
    return {"model_key": None, "device": None, "heads": {}}


# Lepas model dari memori (RAM/GPU).
def release_transformers() -> None:
    registry = _head_registry()
    registry["heads"].clear()
    registry["model_key"], registry["device"] = None, None
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# Cek apakah error disebabkan memori habis.
def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        isinstance(exc, MemoryError)
        or "out of memory" in text
        or "not enough memory" in text
        or "defaultcpuallocator" in text
    )


# Tentukan perangkat: GPU bila tersedia, kalau tidak CPU.
def resolve_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# Import torch dan transformers; beri pesan jelas bila belum terpasang.
def _import_torch_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "PyTorch dan Transformers belum terpasang. Jalankan: " "pip install torch transformers"
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def get_head(model_key: str, task: str, device: str) -> dict[str, Any]:
    """Muat satu kepala model (issue atau severity) bila belum ada di registry."""
    registry = _head_registry()
    if registry["model_key"] != model_key or registry["device"] != device:
        release_transformers()
        registry["model_key"], registry["device"] = model_key, device
    if task not in registry["heads"]:
        torch, model_cls, tokenizer_cls = _import_torch_stack()
        # pergantian model repos
        repo = hf_repo_for(model_key, task)
        if repo is not None:
            # Unduh dari Hugging Face (pertama kali saja, lalu di-cache di disk server)
            tokenizer = tokenizer_cls.from_pretrained(repo)
            model = model_cls.from_pretrained(repo, low_cpu_mem_usage=True)
        else:
            folder, _ = model_dir(model_key, task)
            tokenizer = tokenizer_cls.from_pretrained(str(folder), local_files_only=True)
            model = model_cls.from_pretrained(str(folder), local_files_only=True)
        model.eval()
        model.to(torch.device(device))
        registry["heads"][task] = {"torch": torch, "model": model, "tokenizer": tokenizer, "device": device}
    return registry["heads"][task]


# Buang satu model (issue/severity) dari memori.
def drop_head(task: str) -> None:
    _head_registry()["heads"].pop(task, None)
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# Jalankan model Transformer dan ambil skor mentahnya.
def run_logits(head: dict[str, Any], texts: list[str], max_length: int, expected_outputs: int) -> np.ndarray:
    torch = head["torch"]
    encoded = head["tokenizer"](texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    encoded = {k: v.to(head["device"]) for k, v in encoded.items()}
    with torch.inference_mode():
        logits = head["model"](**encoded).logits
    array = logits.detach().float().cpu().numpy()
    if array.shape[-1] != expected_outputs:
        raise ValueError(f"Model menghasilkan {array.shape[-1]} output, seharusnya {expected_outputs}.")
    return array


# Satu kali prediksi Transformer (issue lalu severity per kategori).
def _predict_transformer_once(
    model_key: str, text_norm: str, threshold: float, fallback: bool, device: str, low_memory: bool
) -> dict[str, Any]:
    started = time.time()
    status = model_status(model_key)
    _, summary = find_summary(model_key)
    max_length = int((summary or {}).get("max_length") or DEFAULT_MAX_LENGTH)
    issue_labels = status["tasks"]["issue"]["labels"]
    severity_labels = status["tasks"]["severity"]["labels"]

    head = get_head(model_key, "issue", device)
    issue_logits = run_logits(head, [text_norm], max_length, len(issue_labels))[0]
    if low_memory:
        del head
        drop_head("issue")
    decision = decide_issues(issue_logits, issue_labels, threshold, fallback)

    severity_rows: list[dict[str, Any]] = []
    if decision["selected"]:
        categories = [label for label, _ in decision["selected"]]
        head = get_head(model_key, "severity", device)
        sev_logits = run_logits(
            head, [build_severity_text(text_norm, c) for c in categories], max_length, len(severity_labels)
        )
        if low_memory:
            del head
            drop_head("severity")
        for (category, issue_score), sev in zip(decision["selected"], decide_severity(sev_logits, severity_labels)):
            severity_rows.append({"category": category, "issue_score": issue_score, **sev})

    return {
        "kind": "transformer",
        "model_key": model_key,
        "issue": decision,
        "severity_rows": severity_rows,
        "warnings": status["tasks"]["issue"]["warnings"] + status["tasks"]["severity"]["warnings"],
        "device": device,
        "seconds": time.time() - started,
        "max_length": max_length,
    }


# Prediksi Transformer; bila memori habis di GPU, ulangi di CPU.
def predict_transformer(
    model_key: str, text_norm: str, threshold: float, fallback: bool, device_pref: str, low_memory: bool
) -> dict[str, Any]:
    status = model_status(model_key)
    if not status["ready"]:
        raise FileNotFoundError(f"Model {MODEL_NAMES[model_key]} belum terhubung.")
    device = resolve_device(device_pref)
    try:
        result = _predict_transformer_once(model_key, text_norm, threshold, fallback, device, low_memory)
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        release_transformers()
        if device == "cuda":  # coba ulang di CPU
            try:
                result = _predict_transformer_once(model_key, text_norm, threshold, fallback, "cpu", True)
                result["warnings"].append("GPU kehabisan memori (VRAM tidak cukup) sehingga prediksi diulang di CPU.")
            except Exception as exc2:  # noqa: BLE001
                release_transformers()
                if _is_oom(exc2):
                    raise MemoryError(
                        "Memori tidak cukup untuk memuat model, baik di GPU maupun CPU. Tutup "
                        "aplikasi lain, aktifkan 'Hemat memori', atau pilih model yang lebih kecil "
                        "(IndoBERT < mBERT < XLM-RoBERTa)."
                    ) from exc2
                raise
        else:
            raise MemoryError(
                "RAM tidak cukup untuk memuat model. Tutup aplikasi lain, aktifkan 'Hemat memori', "
                "atau pilih model yang lebih kecil (IndoBERT < mBERT < XLM-RoBERTa)."
            ) from exc
    if low_memory:
        release_transformers()
    return result


# Jalankan SVM dan bungkus hasilnya untuk ditampilkan.
def predict_svm_result(text_norm: str) -> dict[str, Any]:
    started = time.time()
    bundle = get_svm_bundle()
    result = predict_svm(bundle, text_norm)
    notes = []
    if bundle["idf_shim_applied"]:
        notes.append(
            f"Artefak dilatih dengan scikit-learn {bundle['trained_with'] or 'versi lama'}, dijalankan di "
            f"{bundle['sklearn_version']}; bobot IDF dipulihkan lewat shim kompatibilitas "
            f"(cek 'Kesiapan Artefak' untuk uji reproduksi)."
        )
    return {
        "kind": "svm",
        "model_key": "svm_tfidf",
        **result,
        "warnings": notes,
        "device": "cpu",
        "seconds": time.time() - started,
    }


# Pilih fungsi prediksi sesuai model (SVM atau Transformer).
def run_prediction(
    model_key: str, text_norm: str, threshold: float, fallback: bool, device_pref: str, low_memory: bool
) -> dict[str, Any]:
    if model_key == "svm_tfidf":
        return predict_svm_result(text_norm)
    return predict_transformer(model_key, text_norm, threshold, fallback, device_pref, low_memory)


# Severity tertinggi dari daftar (LOW < MEDIUM < HIGH < BLOCKER).
def highest_severity(labels: list[str]) -> str | None:
    labels = [l for l in labels if l in SEVERITY_RANK]
    return max(labels, key=lambda l: SEVERITY_RANK[l]) if labels else None


# =====================================================================
# Lingkungan
# =====================================================================
def package_versions() -> dict[str, str]:
    names = ["streamlit", "pandas", "numpy", "scikit-learn", "joblib", "torch", "transformers"]
    out = {}
    for name in names:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = "belum terpasang"
    return out


# Info GPU untuk halaman status.
def gpu_info() -> str:
    try:
        import torch
    except ImportError:
        return "PyTorch belum terpasang."
    if not torch.cuda.is_available():
        return "CUDA tidak tersedia (inferensi memakai CPU)."
    props = torch.cuda.get_device_properties(0)
    return f"{props.name} — VRAM {props.total_memory / 1024 ** 3:.1f} GB"


# =====================================================================
# Dataset korpus (lazy; hanya kolom yang dibutuhkan, tanpa user_name/developer_reply)
# =====================================================================
DATASET_COLUMNS = [
    "review_id",
    "review",
    "text_clean",
    "text_norm",
    "token_count",
    "rating",
    "review_date",
    "review_year",
    "category_Issue",
    "severity_Issue",
]


# Baca kolom penting dari dataset utama (di-cache).
@st.cache_data(show_spinner=False)
def _load_dataset_cached(path_str: str, mtime: float) -> pd.DataFrame:  # noqa: ARG001 (mtime = cache key)
    header = pd.read_csv(path_str, nrows=0, encoding="utf-8-sig").columns
    use = [c for c in DATASET_COLUMNS if c in header]
    return pd.read_csv(path_str, usecols=use, encoding="utf-8-sig")


# Ambil dataset utama; kosong bila file tidak ada.
def get_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        return pd.DataFrame()
    return _load_dataset_cached(str(DATASET_PATH), _mtime(DATASET_PATH))


@st.cache_data(show_spinner=False)
def dataset_overview(path_str: str, mtime: float) -> dict[str, Any]:
    """Statistik korpus. Kategori/severity dihitung per baris anotasi (review x kategori);
    rating, token, tahun, dan multi-label dihitung per review unik."""
    df = _load_dataset_cached(path_str, mtime)
    out: dict[str, Any] = {"n_rows": len(df)}
    if "review_id" not in df.columns:
        return out
    reviews = df.drop_duplicates("review_id")
    out["n_reviews"] = int(reviews["review_id"].nunique())
    if "review_year" in df.columns:
        years = reviews["review_year"].dropna().astype(int)
        out["year_min"], out["year_max"] = int(years.min()), int(years.max())
    if "review_date" in df.columns:
        dates = pd.to_datetime(reviews["review_date"], errors="coerce").dropna()
        if len(dates):
            out["date_max"] = dates.max().strftime("%d %b %Y")
    if "token_count" in reviews.columns:
        out["tok_mean"], out["tok_min"], out["tok_max"] = (
            float(reviews["token_count"].mean()),
            int(reviews["token_count"].min()),
            int(reviews["token_count"].max()),
        )
    if "rating" in reviews.columns:
        counts = reviews["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        out["rating_counts"] = {int(k): int(v) for k, v in counts.items()}
        out["rating_avg"] = float(reviews["rating"].mean())
    if "category_Issue" in df.columns:
        out["cat_counts"] = {k: int(v) for k, v in df["category_Issue"].value_counts().items()}
        per_review = df.groupby("review_id")["category_Issue"].agg(lambda s: sorted(set(s.dropna())))
        multi = per_review[per_review.map(len) > 1]
        out["multi_n"] = int(len(multi))
        out["multi_pct"] = float(len(multi) / max(len(per_review), 1))
        pairs: dict[tuple[str, str], int] = {}
        for cats in multi:
            for pair in itertools.combinations(cats, 2):
                pairs[pair] = pairs.get(pair, 0) + 1
        out["top_pairs"] = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if "severity_Issue" in df.columns:
        out["sev_counts"] = {k: int(v) for k, v in df["severity_Issue"].value_counts().items()}
    return out


# Jumlah ulasan per tahun untuk satu kategori.
@st.cache_data(show_spinner=False)
def category_year_trend(path_str: str, mtime: float, category: str) -> dict[int, int]:
    df = _load_dataset_cached(path_str, mtime)
    if not {"category_Issue", "review_year"}.issubset(df.columns):
        return {}
    sub = df[df["category_Issue"] == category]
    counts = sub["review_year"].dropna().astype(int).value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}


@st.cache_data(show_spinner=False)
def _quick_samples_cached(path_str: str, mtime: float) -> list[dict[str, str]]:  # noqa: ARG001
    """Empat contoh nyata dari data uji SVM (review mentah + text_norm + label asli), deterministik."""
    df = read_csv(Path(path_str))
    need = {"review", "text_used_by_model", "issue_true", "severity_true"}
    if not need.issubset(df.columns):
        return []
    df = df[df["review"].astype(str).str.len().between(28, 85)]
    picks, seen = [], set()
    for _, row in df.iterrows():
        first = parse_label_list(row["issue_true"])
        key = first[0] if first else ""
        if not key or key in seen:
            continue
        seen.add(key)
        picks.append(
            {
                "review": str(row["review"]).strip(),
                "text_norm": str(row["text_used_by_model"]),
                "issue_true": str(row["issue_true"]),
                "severity_true": str(row["severity_true"]),
            }
        )
        if len(picks) == 4:
            break
    return picks


# Contoh ulasan dari data uji untuk tombol cepat.
def quick_samples() -> list[dict[str, str]]:
    path = result_file("svm_tfidf", "issue", "test_predictions")
    return _quick_samples_cached(str(path), _mtime(path)) if path else []


# =====================================================================
# Footprint bobot & status lingkungan (tanpa mengimpor torch)
# =====================================================================
def fmt_size(num_bytes: int | None) -> str:
    if not num_bytes:
        return NA_TEXT
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return NA_TEXT


def weights_footprint(model_key: str) -> int | None:
    """Total ukuran bobot di disk. None bila ada bagian yang belum tersedia."""
    total = 0
    if model_key == "svm_tfidf":
        for path in MODEL_PATHS[model_key].values():
            if not _is_nonempty_file(Path(path)):
                return None
            total += Path(path).stat().st_size
        return total
    for task in ("issue", "severity"):
        # pergantian model repos
        repo = hf_repo_for(model_key, task)
        if repo is not None:
            hf_size = hf_weights_size(repo)
            if hf_size is None:
                return None
            total += hf_size
            continue
        folder, _ = model_dir(model_key, task)
        sizes = [
            (folder / n).stat().st_size
            for n in ("model.safetensors", "pytorch_model.bin")
            if _is_nonempty_file(folder / n)
        ]
        if not sizes:
            return None
        total += sizes[0]
    return total


def torch_status() -> str:
    """Cek status PyTorch tanpa membuat aplikasi crash."""
    if importlib.util.find_spec("torch") is None:
        return "PyTorch belum terpasang"

    torch_module = sys.modules.get("torch")

    if torch_module is None:
        return "PyTorch terpasang"

    cuda_module = getattr(torch_module, "cuda", None)

    if cuda_module is None:
        return "PyTorch belum siap dimuat"

    try:
        return (
            "PyTorch • CUDA GPU"
            if cuda_module.is_available()
            else "PyTorch • CPU"
        )
    except Exception:
        return "PyTorch gagal dimuat"


# Format angka desimal.
def fmt_float(value: float | None, digits: int = 4) -> str:
    return NA_TEXT if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


# Format durasi: ms bila kurang dari 1 detik, selain itu detik.
def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return NA_TEXT
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} dtk"


# =====================================================================
# Tindak lanjut manual: penyimpanan permanen (SQLite) + ekspor CSV/Excel
# =====================================================================
# Prinsip:
#   * Hanya menyimpan hasil prediksi yang BENAR-BENAR dijalankan (st.session_state["pred_store"]),
#     ditambah isian manual pengguna. Tidak ada prediksi, solusi, atau model tambahan.
#   * Skema dibuat dengan CREATE TABLE IF NOT EXISTS: database lama tidak pernah dihapus/ditimpa.
#   * Setiap submit = INSERT baris baru dengan review_id unik. Perubahan status = UPDATE kolom
#     status saja (+ jejak di status_history); data lain tidak disentuh.
FOLLOWUP_STATUSES = ["Belum Ditindaklanjuti", "Dalam Peninjauan", "Selesai"]
FOLLOWUP_STATUS_PILL = {
    "Belum Ditindaklanjuti": ("red", "#d12027"),
    "Dalam Peninjauan": ("amber", "#f59e0b"),
    "Selesai": ("green", "#10b981"),
}
RUN_STATUS_TEXT = {"ok": "Berhasil", "unavailable": "Belum terhubung", "error": "Gagal", "not_run": "Tidak dijalankan"}
EXCEL_SHEET_REVIEWS = "Review Analysis"
EXCEL_SHEET_DETAILS = "Issue Details"
DB_SCHEMA_VERSION = 1

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id         TEXT    NOT NULL UNIQUE,
    prediction_run_id TEXT,
    submitted_at      TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL,
    review_text       TEXT    NOT NULL,
    input_text        TEXT,
    text_mode         TEXT,
    threshold         REAL,
    fallback          INTEGER,
    predicted_at      TEXT,
    issues_reviewed   TEXT    NOT NULL,
    solution          TEXT    NOT NULL,
    notes             TEXT,
    status            TEXT    NOT NULL,
    predictions_json  TEXT
);
CREATE TABLE IF NOT EXISTS model_predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id             TEXT NOT NULL REFERENCES reviews(review_id),
    model_key             TEXT NOT NULL,
    model_name            TEXT NOT NULL,
    run_status            TEXT NOT NULL,
    issue_pred            TEXT,
    severity_pred         TEXT,
    severity_basis        TEXT,
    severity_by_category  TEXT,
    message               TEXT,
    seconds               REAL,
    device                TEXT,
    UNIQUE (review_id, model_key)
);
CREATE TABLE IF NOT EXISTS issue_details (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id       TEXT NOT NULL REFERENCES reviews(review_id),
    model_key       TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    run_status      TEXT NOT NULL,
    category        TEXT,
    predicted       INTEGER,
    issue_score     REAL,
    score_type      TEXT,
    severity        TEXT,
    severity_score  REAL,
    severity_basis  TEXT,
    reviewed        INTEGER
);
CREATE TABLE IF NOT EXISTS status_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id   TEXT NOT NULL REFERENCES reviews(review_id),
    old_status  TEXT,
    new_status  TEXT NOT NULL,
    changed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_review   ON model_predictions(review_id);
CREATE INDEX IF NOT EXISTS idx_detail_review ON issue_details(review_id);
CREATE INDEX IF NOT EXISTS idx_hist_review   ON status_history(review_id);
"""


# Waktu sekarang sebagai teks.
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Buka koneksi database SQLite (folder dibuat bila belum ada).
def _db_connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or FOLLOWUP_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=15)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_followup_db(path: Path | None = None) -> Path:
    """Buat tabel bila belum ada. Aman dipanggil berulang: data lama tetap utuh."""
    path = Path(path or FOLLOWUP_DB_PATH)
    with closing(_db_connect(path)) as con:
        con.executescript(_DB_SCHEMA)
        if con.execute("PRAGMA user_version").fetchone()[0] < DB_SCHEMA_VERSION:
            con.execute(f"PRAGMA user_version = {DB_SCHEMA_VERSION}")
        con.commit()
    return path


# Ubah tipe numpy agar bisa disimpan sebagai JSON.
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def summarize_model_result(key: str, res: dict[str, Any] | None) -> dict[str, Any]:
    """Ringkas satu hasil run_prediction() menjadi bentuk yang bisa disimpan.

    Hanya MEMBACA hasil prediksi yang sudah ada; tidak menghitung ulang apa pun.
    """
    base = {
        "model_key": key,
        "model_name": MODEL_NAMES[key],
        "issues": [],
        "severity": None,
        "severity_basis": "",
        "severity_by_category": [],
        "detail_rows": [],
        "message": "",
        "seconds": None,
        "device": "",
    }
    if res is None:
        return {**base, "run_status": "not_run", "message": "Model tidak dipilih saat prediksi."}
    kind = res.get("kind")
    if kind == "unavailable":
        miss = []
        for task, info in res.get("status", {}).get("tasks", {}).items():
            if not info.get("ready"):
                miss.append(f"{task}: " + "; ".join(map(str, info.get("missing", []) + info.get("problems", []))))
        return {**base, "run_status": "unavailable", "message": " | ".join(miss) or "Model belum terhubung."}
    if kind == "error":
        return {**base, "run_status": "error", "message": str(res.get("message", ""))}

    out = {**base, "run_status": "ok", "seconds": res.get("seconds"), "device": str(res.get("device", ""))}
    if kind == "svm":
        picked = list(res.get("issue_predicted", []))
        sev_map = {r["category"]: (r["severity"], r["score"]) for r in res.get("severity_rows", [])}
        out.update(
            issues=picked,
            severity=highest_severity([sv for sv, _ in sev_map.values()]),
            severity_basis="tertinggi dari kategori terprediksi",
            severity_by_category=[(c, sev_map[c][0], sev_map[c][1]) for c in picked if c in sev_map],
        )
        for label, score in res.get("issue_scores", []):
            chosen = label in picked
            sev, sev_score = sev_map.get(label, (None, None))
            out["detail_rows"].append(
                {
                    "category": label,
                    "predicted": chosen,
                    "issue_score": float(score),
                    "score_type": "margin SVM",
                    "severity": sev,
                    "severity_score": sev_score,
                    "severity_basis": "per kategori (margin SVM)" if sev else "",
                }
            )
        if not picked:
            out["message"] = "Tidak ada kategori issue terprediksi; severity tidak dihitung."
        elif not sev_map:
            out["message"] = "Kategori terprediksi tidak dikenali model severity."
        return out

    # Transformer
    selected = [label for label, _ in res.get("issue", {}).get("selected", [])]
    sev_map = {r["category"]: (r["severity"], r["score"]) for r in res.get("severity_rows", [])}
    out.update(
        issues=selected,
        severity=highest_severity([s for s, _ in sev_map.values()]),
        severity_basis="tertinggi dari kategori terprediksi",
        severity_by_category=[(c, sev_map[c][0], sev_map[c][1]) for c in selected if c in sev_map],
    )
    for label, score in res.get("issue", {}).get("scores", []):
        chosen = label in selected
        sev, sev_score = sev_map.get(label, (None, None))
        out["detail_rows"].append(
            {
                "category": label,
                "predicted": chosen,
                "issue_score": float(score),
                "score_type": "sigmoid",
                "severity": sev,
                "severity_score": sev_score,
                "severity_basis": "per kategori (softmax)" if sev else "",
            }
        )
    if res.get("issue", {}).get("used_fallback"):
        out["message"] = "Fallback ke kategori skor tertinggi aktif."
    elif not selected:
        out["message"] = "Tidak ada kategori ≥ threshold; severity tidak dihitung."
    return out


def save_followup(
    store: dict[str, Any], issues_reviewed: list[str], solution: str, notes: str, status: str, path: Path | None = None
) -> str:
    """Simpan SATU submission baru (INSERT, tidak pernah menimpa). Mengembalikan review_id."""
    if status not in FOLLOWUP_STATUSES:
        raise ValueError(f"Status tidak dikenal: {status}")
    if not solution.strip():
        raise ValueError("Usulan solusi wajib diisi.")
    if not issues_reviewed:
        raise ValueError("Pilih minimal satu issue yang ingin ditinjau.")
    summaries = [summarize_model_result(k, store["results"].get(k)) for k in MODEL_ORDER]
    if not any(s["run_status"] == "ok" for s in summaries):
        raise ValueError("Belum ada model yang berhasil menghasilkan prediksi.")
    path = init_followup_db(path)
    now = _now_str()
    reviewed_set = set(issues_reviewed)
    raw_json = json.dumps(
        {"order": store.get("order"), "results": store.get("results")}, default=_json_default, ensure_ascii=False
    )
    with closing(_db_connect(path)) as con:
        for _ in range(5):  # review_id unik; ulangi bila (sangat jarang) bentrok
            review_id = f"RV-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
            if not con.execute("SELECT 1 FROM reviews WHERE review_id = ?", (review_id,)).fetchone():
                break
        try:
            con.execute(
                "INSERT INTO reviews (review_id, prediction_run_id, submitted_at, updated_at, review_text, input_text, "
                "text_mode, threshold, fallback, predicted_at, issues_reviewed, solution, notes, status, predictions_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    review_id,
                    store.get("run_id"),
                    now,
                    now,
                    store["text"],
                    store.get("input_text"),
                    store.get("text_mode"),
                    store.get("threshold"),
                    int(bool(store.get("fallback"))),
                    store.get("predicted_at"),
                    " | ".join(issues_reviewed),
                    solution.strip(),
                    notes.strip() or None,
                    status,
                    raw_json,
                ),
            )
            for s in summaries:
                con.execute(
                    "INSERT INTO model_predictions (review_id, model_key, model_name, run_status, issue_pred, "
                    "severity_pred, severity_basis, severity_by_category, message, seconds, device) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        review_id,
                        s["model_key"],
                        s["model_name"],
                        s["run_status"],
                        " | ".join(s["issues"]),
                        s["severity"],
                        s["severity_basis"],
                        "; ".join(f"{c}={sev}" for c, sev, _ in s["severity_by_category"]),
                        s["message"],
                        s["seconds"],
                        s["device"],
                    ),
                )
                rows = s["detail_rows"] or [
                    {
                        "category": None,
                        "predicted": None,
                        "issue_score": None,
                        "score_type": None,
                        "severity": None,
                        "severity_score": None,
                        "severity_basis": "",
                    }
                ]
                for r in rows:
                    con.execute(
                        "INSERT INTO issue_details (review_id, model_key, model_name, run_status, category, predicted, "
                        "issue_score, score_type, severity, severity_score, severity_basis, reviewed) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            review_id,
                            s["model_key"],
                            s["model_name"],
                            s["run_status"],
                            r["category"],
                            None if r["predicted"] is None else int(r["predicted"]),
                            r["issue_score"],
                            r["score_type"],
                            r["severity"],
                            r["severity_score"],
                            r["severity_basis"] or None,
                            None if r["category"] is None else int(r["category"] in reviewed_set),
                        ),
                    )
            con.execute(
                "INSERT INTO status_history (review_id, old_status, new_status, changed_at) VALUES (?,?,?,?)",
                (review_id, None, status, now),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
    return review_id


def update_followup_status(review_id: str, new_status: str, path: Path | None = None) -> bool:
    """Ubah status saja (data lain tetap). Mengembalikan False bila status tidak berubah."""
    if new_status not in FOLLOWUP_STATUSES:
        raise ValueError(f"Status tidak dikenal: {new_status}")
    path = init_followup_db(path)
    with closing(_db_connect(path)) as con:
        row = con.execute("SELECT status FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
        if row is None:
            raise KeyError(f"Review {review_id} tidak ditemukan.")
        if row[0] == new_status:
            return False
        now = _now_str()
        con.execute("UPDATE reviews SET status = ?, updated_at = ? WHERE review_id = ?", (new_status, now, review_id))
        con.execute(
            "INSERT INTO status_history (review_id, old_status, new_status, changed_at) VALUES (?,?,?,?)",
            (review_id, row[0], new_status, now),
        )
        con.commit()
    return True


def load_followups(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Baca seluruh isi database (reviews, model_predictions, issue_details) tanpa cache."""
    path = init_followup_db(path)
    with closing(_db_connect(path)) as con:
        reviews = pd.read_sql_query("SELECT * FROM reviews ORDER BY id", con)
        preds = pd.read_sql_query("SELECT * FROM model_predictions ORDER BY id", con)
        details = pd.read_sql_query("SELECT * FROM issue_details ORDER BY id", con)
    return reviews, preds, details


# Ubah kode kategori menjadi nama tampilan.
def _issue_names(codes: Any) -> str:
    items = parse_label_list(codes)
    return " | ".join(ISSUE_DISPLAY.get(c, c) for c in items)


def build_review_analysis(reviews: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Sheet 1 / CSV: satu baris per review yang disubmit."""
    rows = []
    pred_idx = {(r.review_id, r.model_key): r for r in preds.itertuples(index=False)}
    for r in reviews.itertuples(index=False):
        row: dict[str, Any] = {
            "No": int(r.id),
            "Review ID": r.review_id,
            "Teks Ulasan (dipakai model)": r.review_text,
            "Teks Input Asli": r.input_text,
            "Format Teks": r.text_mode,
            "Threshold Issue": r.threshold,
            "Fallback": "Ya" if r.fallback else "Tidak",
        }
        for key in MODEL_ORDER:
            name = MODEL_NAMES[key]
            p = pred_idx.get((r.review_id, key))
            status = RUN_STATUS_TEXT.get(p.run_status, p.run_status) if p is not None else RUN_STATUS_TEXT["not_run"]
            ok = p is not None and p.run_status == "ok"
            row[f"{name} | Status Prediksi"] = status
            row[f"{name} | Issue"] = (p.issue_pred or NO_LABEL_TEXT) if ok else ""
            row[f"{name} | Severity"] = (p.severity_pred or "—") if ok else ""
            row[f"{name} | Dasar Severity"] = p.severity_basis if ok else ""
            row[f"{name} | Severity per Kategori"] = (p.severity_by_category or "") if ok else ""
            row[f"{name} | Keterangan"] = (p.message or "") if p is not None else ""
        row.update(
            {
                "Issue Ditinjau (kode)": r.issues_reviewed,
                "Issue Ditinjau": _issue_names(r.issues_reviewed),
                "Usulan Solusi / Tindak Lanjut": r.solution,
                "Catatan Tambahan": r.notes or "",
                "Status Tindak Lanjut": r.status,
                "Waktu Prediksi": r.predicted_at or "",
                "Waktu Submit": r.submitted_at,
                "Terakhir Diperbarui": r.updated_at,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_issue_details(reviews: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    """Sheet 2: satu baris per kombinasi review x model x kategori issue."""
    if details.empty:
        return pd.DataFrame(
            columns=[
                "Review ID",
                "No",
                "Waktu Submit",
                "Model",
                "Status Prediksi",
                "Kategori (kode)",
                "Kategori",
                "Diprediksi",
                "Skor Issue",
                "Jenis Skor",
                "Severity",
                "Skor Severity",
                "Dasar Severity",
                "Ditinjau Manual",
            ]
        )
    meta = reviews.set_index("review_id")[["id", "submitted_at"]]
    order = {k: i for i, k in enumerate(MODEL_ORDER)}
    d = details.copy()
    d["_m"] = d["model_key"].map(order)
    d["_no"] = d["review_id"].map(meta["id"])
    d = d.sort_values(["_no", "_m", "id"])
    yn = lambda v: "" if pd.isna(v) else ("Ya" if int(v) else "Tidak")  # noqa: E731
    return pd.DataFrame(
        {
            "Review ID": d["review_id"].values,
            "No": d["_no"].astype("Int64").values,
            "Waktu Submit": d["review_id"].map(meta["submitted_at"]).values,
            "Model": d["model_name"].values,
            "Status Prediksi": d["run_status"].map(lambda s: RUN_STATUS_TEXT.get(s, s)).values,
            "Kategori (kode)": d["category"].fillna("").values,
            "Kategori": d["category"].map(lambda c: ISSUE_DISPLAY.get(c, c) if isinstance(c, str) else "").values,
            "Diprediksi": d["predicted"].map(yn).values,
            "Skor Issue": d["issue_score"].values,
            "Jenis Skor": d["score_type"].fillna("").values,
            "Severity": d["severity"].fillna("").values,
            "Skor Severity": d["severity_score"].values,
            "Dasar Severity": d["severity_basis"].fillna("").values,
            "Ditinjau Manual": d["reviewed"].map(yn).values,
        }
    )


# Ubah tabel menjadi bytes CSV untuk tombol unduh.
def export_csv_bytes(df: pd.DataFrame) -> bytes:
    # utf-8-sig agar huruf/emoji terbaca benar saat CSV dibuka langsung di Excel.
    return df.to_csv(index=False).encode("utf-8-sig")


def export_excel_bytes(review_df: pd.DataFrame, detail_df: pd.DataFrame) -> bytes:
    """Excel 2 sheet. Kolom Review ID di 'Issue Details' berupa tautan ke baris review di sheet 1."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.hyperlink import Hyperlink

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        review_df.to_excel(writer, sheet_name=EXCEL_SHEET_REVIEWS, index=False)
        detail_df.to_excel(writer, sheet_name=EXCEL_SHEET_DETAILS, index=False)
        head_fill = PatternFill("solid", start_color="AA0015")
        head_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        body_font = Font(name="Arial", size=10)
        link_font = Font(name="Arial", size=10, color="1D4ED8", underline="single")
        thin = Side(style="thin", color="E2E8F0")
        wide = {
            "Teks Ulasan (dipakai model)": 60,
            "Teks Input Asli": 50,
            "Usulan Solusi / Tindak Lanjut": 50,
            "Catatan Tambahan": 40,
        }
        for ws, df in (
            (writer.sheets[EXCEL_SHEET_REVIEWS], review_df),
            (writer.sheets[EXCEL_SHEET_DETAILS], detail_df),
        ):
            for cell in ws[1]:
                cell.fill, cell.font = head_fill, head_font
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.font = body_font
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                    cell.border = Border(bottom=thin)
            for i, col in enumerate(df.columns, start=1):
                longest = max([len(str(col))] + [len(str(v)) for v in df[col].head(200).tolist()])
                ws.column_dimensions[get_column_letter(i)].width = wide.get(col, min(max(12, longest + 2), 42))
                if col in ("Skor Issue", "Skor Severity", "Threshold Issue"):
                    for (cell,) in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                        cell.number_format = "0.0000"
            ws.freeze_panes = "C2"
            ws.row_dimensions[1].height = 32
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
        # Hubungkan Issue Details -> Review Analysis lewat Review ID
        ws_r, ws_d = writer.sheets[EXCEL_SHEET_REVIEWS], writer.sheets[EXCEL_SHEET_DETAILS]
        row_of = {ws_r.cell(row=i, column=2).value: i for i in range(2, ws_r.max_row + 1)}
        for i in range(2, ws_d.max_row + 1):
            cell = ws_d.cell(row=i, column=1)
            target = row_of.get(cell.value)
            if target:
                cell.hyperlink = Hyperlink(
                    ref=cell.coordinate,
                    location=f"'{EXCEL_SHEET_REVIEWS}'!A{target}",
                    tooltip="Buka baris review di sheet Review Analysis",
                )
                cell.font = link_font
    return buffer.getvalue()


# =====================================================================
# UI: ikon, CSS, dan pembangun HTML (semua satu baris agar aman dari aturan Markdown)
# =====================================================================
def _h(markup: str) -> str:
    """Padatkan HTML menjadi satu baris: Markdown mengubah baris berindentasi / HTML dengan baris
    kosong menjadi blok kode, sehingga kartu tampil sebagai teks mentah."""
    text = re.sub(r">\s*\n\s*<", "><", markup.strip())  # antar tag: tanpa spasi
    return re.sub(r"\s*\n\s*", " ", text)  # dalam teks: satu spasi


# Amankan teks sebelum dimasukkan ke HTML.
def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


# Ikon SVG inline (gaya garis 24x24) - tidak bergantung pada font ikon eksternal.
_ICON_PATHS = {
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "compare": '<path d="m8 3-4 4 4 4"/><path d="M4 7h16"/><path d="m16 21 4-4-4-4"/><path d="M20 17H4"/>',
    "chart": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "drive": '<path d="M22 12H2"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><path d="M6 16h.01"/><path d="M10 16h.01"/>',
    "cap": '<path d="M21.42 10.92a1 1 0 0 0-.02-1.84L12.83 5.18a2 2 0 0 0-1.66 0L2.6 9.08a1 1 0 0 0 0 1.83l8.57 3.91a2 2 0 0 0 1.66 0z"/><path d="M22 10v6"/><path d="M6 12.5V16a6 3 0 0 0 12 0v-3.5"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M13 8H7"/><path d="M17 12H7"/>',
    "trend": '<path d="M22 7 13.5 15.5 8.5 10.5 2 17"/><path d="M16 7h6v6"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "list": '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    "star": '<path d="M11.5 2.3a.6.6 0 0 1 1 0l2.3 4.7a2.1 2.1 0 0 0 1.6 1.2l5.1.7a.5.5 0 0 1 .3.9l-3.7 3.6a2.1 2.1 0 0 0-.6 1.9l.9 5.1a.5.5 0 0 1-.8.6l-4.6-2.4a2.1 2.1 0 0 0-2 0l-4.6 2.4a.5.5 0 0 1-.8-.6l.9-5.1a2.1 2.1 0 0 0-.6-1.9L2.2 9.8a.5.5 0 0 1 .3-.9l5.1-.7a2.1 2.1 0 0 0 1.6-1.2z"/>',
}


# Buat ikon SVG dari nama ikon.
def icon(name: str, size: int = 18, color: str = "currentColor", stroke: float = 2) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="flex:none">{_ICON_PATHS[name]}</svg>'
    )


# Ubah ikon menjadi data-URI untuk dipakai di CSS.
def _icon_uri(name: str) -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>'
    )
    return (
        'url("data:image/svg+xml;utf8,'
        + svg.replace('"', "'").replace("#", "%23").replace("<", "%3C").replace(">", "%3E")
        + '")'
    )


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{
  --crimson:#aa0015; --crimson-2:#d12027; --navy:#0f172a;
  --canvas:#f8fafc; --surface:#ffffff; --muted:#f1f5f9; --border:#e2e8f0; --border-2:#cbd5e1;
  --text:#0f172a; --text-2:#475569; --text-3:#64748b;
  --font-body:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif;
  --font-mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
/* ---------- Dasar & tipografi ---------- */
html,body,.stApp{background:var(--canvas)!important;}
.stApp,.stApp p,.stApp li,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp summary,
.stApp [data-baseweb="select"] div,.stApp [data-baseweb="tab"],.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stWidgetLabel"]{font-family:var(--font-body);font-size:15px;color:var(--text);}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:var(--font-body);color:var(--navy);letter-spacing:-.01em;}
.stApp code,.stApp pre{font-family:var(--font-mono);font-size:13px;}
.stApp [data-testid="stWidgetLabel"] p{font-size:13px;font-weight:600;color:var(--text-2);}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] p{color:var(--text-2);font-size:13px;line-height:1.5;}
#MainMenu, footer, .stDeployButton, [data-testid="stDecoration"], [data-testid="stStatusWidget"]{display:none!important;}
header[data-testid="stHeader"]{background:transparent!important;height:3rem!important;min-height:3rem!important;overflow:visible!important;}
[data-testid="stSidebarCollapsedControl"]{display:flex!important;visibility:visible!important;opacity:1!important;pointer-events:auto!important;position:fixed!important;top:.5rem!important;left:.5rem!important;z-index:999999!important;}
[data-testid="stSidebarCollapsedControl"] button{display:flex!important;visibility:visible!important;opacity:1!important;pointer-events:auto!important;}
.block-container{max-width:1600px!important;padding:1rem 2rem 3rem!important;}
div[data-testid="stVerticalBlock"]{gap:1rem;}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{background:#fff!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"][aria-expanded="true"]{min-width:320px!important;max-width:320px!important;}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:0!important;}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:.25rem 1rem 1.25rem!important;}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{gap:.6rem;}
.brand{display:flex;align-items:center;gap:12px;padding:2px 2px 14px;border-bottom:1px solid var(--muted);margin-top:-40px;}
.brand-logo{width:40px;height:40px;border-radius:10px;background:var(--crimson-2);color:#fff;display:flex;align-items:center;justify-content:center;flex:none;}
.brand-logo-img{
    width:44px;
    height:44px;
    object-fit:contain;
    flex:none;
    display:block;
}
.brand-title{font-size:16px;font-weight:700;color:var(--crimson);line-height:1.2;}
.brand-sub{font-size:13px;color:var(--text-2);margin-top:2px;}
.side-label{display:block;font-size:13px;font-weight:600;color:var(--text-3);margin:12px 2px 6px;}
.module-label{margin-top:24px!important;}
.side-status{background:var(--canvas);border:1px solid var(--border);border-radius:12px;padding:12px;font-size:13px;}
.side-row{display:flex;align-items:center;justify-content:space-between;gap:8px;}
.side-row + .side-row{margin-top:8px;padding-top:8px;border-top:1px solid var(--border);}
.side-row span:first-child{color:var(--text-2);font-weight:600;white-space:nowrap;}
.side-user{display:flex;align-items:center;gap:10px;padding:8px 2px;margin-top:12px;}
.side-avatar{width:32px;height:32px;border-radius:50%;background:var(--muted);color:#334155;display:flex;align-items:center;justify-content:center;flex:none;}

/* Navigasi: radio-style tombol */
.st-key-nav div[data-testid="stButton"] button{justify-content:flex-start;text-align:left;}
.st-key-nav div[data-testid="stButton"] button p{font-size:15px;font-weight:600;}

/* ---------- Komponen kartu ---------- */
.card,[class*="st-key-card-"]{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px;}
div[class*="st-key-card-"]{gap:.85rem;}
.card-head{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px 12px;margin-bottom:12px;}
.card-title{display:flex;align-items:center;gap:8px;font-size:17px;font-weight:700;color:var(--navy);}
.card-title svg{color:var(--crimson);}
.card-sub{font-size:13px;color:var(--text-2);margin-top:4px;line-height:1.5;}
.muted{color:var(--text-2);}
.mono{font-family:var(--font-mono);font-variant-numeric:tabular-nums;}
.label-sm{font-size:13px;font-weight:600;color:var(--text-2);}

/* top bar & banner */
.topbar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px 12px;background:#fff;border:1px solid var(--border);border-radius:12px;padding:10px 16px;}
.crumbs{display:flex;align-items:center;gap:8px;font-size:14px;font-weight:500;color:var(--text-3);}
.crumbs .sep{color:var(--border-2);}
.crumbs .active{color:var(--crimson);font-weight:700;}
.banner h1{font-size:30px;line-height:1.25;font-weight:700;margin:8px 0 6px;padding:0;color:var(--navy);}
.banner p{font-size:15px;line-height:1.6;color:var(--text-2);margin:0;max-width:900px;}
.kpi-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;width:100%;}
.kpi-box{background:var(--canvas);border:1px solid var(--border);border-radius:12px;padding:12px 16px;min-width:0;display:flex;flex-direction:column;justify-content:center;}
.kpi-box .l{font-size:13px;font-weight:500;color:var(--text-2);}
.kpi-box .v{margin-top:2px;font-size:18px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--navy);}
@media (max-width:900px){.kpi-row{grid-template-columns:minmax(0,1fr);}}

/* pill & dot */
.pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:9999px;font-size:13px;font-weight:600;line-height:20px;border:1px solid var(--border);background:var(--muted);color:var(--text-2);white-space:nowrap;}
.pill.mono{font-family:var(--font-mono);font-size:13px;}
.pill.red{background:#fef2f2;color:var(--crimson);border-color:#fecaca;}
.pill.green{background:#ecfdf5;color:#047857;border-color:#a7f3d0;}
.pill.amber{background:#fffbeb;color:#b45309;border-color:#fde68a;}
.pill.indigo{background:#eef2ff;color:#4338ca;border-color:#c7d2fe;}
.pill.slate{background:var(--muted);color:var(--text-2);}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none;}
.sev{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:9999px;font-size:13px;font-weight:700;white-space:nowrap;}

/* grid */
.g2,.g3,.g4{display:grid;gap:14px;align-items:stretch;}
.g2{grid-template-columns:repeat(2,minmax(0,1fr));}
.g3{grid-template-columns:repeat(3,minmax(0,1fr));}
.g4{grid-template-columns:repeat(4,minmax(0,1fr));}
@media (max-width:1200px){.g4{grid-template-columns:repeat(2,minmax(0,1fr));}.g3{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media (max-width:720px){.g4,.g3,.g2{grid-template-columns:minmax(0,1fr);}}

/* section header */
.sec{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;border-bottom:1px solid var(--border);padding-bottom:12px;margin-top:12px;}
.sec-l{display:flex;align-items:center;gap:12px;}
.sec-badge{width:34px;height:34px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px;flex:none;}
.sec-title{font-size:21px;font-weight:700;color:var(--navy);line-height:1.25;}
.sec-sub{font-size:13px;color:var(--text-2);margin-top:2px;}

/* bars */
.bar-row{margin-bottom:12px;}
.bar-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-size:13px;font-weight:600;color:var(--text-2);margin-bottom:4px;}
.bar-top b{font-family:var(--font-mono);font-weight:600;font-variant-numeric:tabular-nums;color:var(--text);}
.track{width:100%;height:8px;background:var(--muted);border-radius:9999px;overflow:hidden;}
.fill{height:100%;border-radius:9999px;}
.mini2{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;}
.mini2 div{background:var(--canvas);border:1px solid var(--border);border-radius:8px;padding:8px;text-align:center;}
.mini2 span{display:block;font-size:13px;color:var(--text-2);}
.mini2 b{font-family:var(--font-mono);font-size:15px;font-variant-numeric:tabular-nums;}

/* model metric card */
.mc{display:flex;flex-direction:column;gap:4px;}
.mc-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;gap:8px;}
.mc-name{font-size:16px;font-weight:700;color:var(--navy);display:flex;align-items:center;gap:8px;white-space:nowrap;}
.mc-line{display:flex;justify-content:space-between;gap:8px;font-size:14px;padding:4px 0;border-bottom:1px solid var(--muted);}
.mc-line span{color:var(--text-2);}
.mc-line b{font-family:var(--font-mono);font-weight:600;font-variant-numeric:tabular-nums;}

/* kartu hasil prediksi */
.rc{display:flex;flex-direction:column;}
.rc.off{background:var(--canvas);}
.rc-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;padding-bottom:12px;border-bottom:1px solid var(--muted);}
.rc-name{font-size:17px;font-weight:700;color:var(--navy);}
.rc-sub{font-size:13px;color:var(--text-2);margin-top:2px;}
.rc-block{padding:14px 0 0;}
.rc-label{font-size:13px;font-weight:600;color:var(--text-2);margin-bottom:8px;}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:8px;background:#eef2ff;border:1px solid #c7d2fe;color:#4338ca;font-size:14px;font-weight:600;}
.chip small{font-family:var(--font-mono);font-size:13px;font-weight:500;opacity:.85;}
.chip.none{background:var(--muted);border-color:var(--border);color:var(--text-2);}
.chips{display:flex;flex-wrap:wrap;gap:6px;}
.sev-list{display:flex;flex-direction:column;gap:8px;}
.sev-row{display:flex;justify-content:space-between;align-items:center;gap:12px;background:var(--canvas);border:1px solid var(--border);border-radius:10px;padding:8px 12px;}
.sev-top{margin-top:10px;font-size:13px;color:var(--text-2);}
.sev-unavail{font-size:13px;color:var(--text-2);}
.rc details{margin-top:14px;border-top:1px solid var(--muted);padding-top:10px;}
.rc summary{cursor:pointer;font-size:13px;font-weight:600;color:var(--text-2);}
.rc table{margin-top:8px;}
.rc .err{color:#b91c1c;font-size:14px;line-height:1.5;word-break:break-word;}

/* tabel HTML */
.tbl{width:100%;border-collapse:collapse;font-size:14px;}
.tbl th{background:var(--canvas);color:var(--text-2);font-size:13px;font-weight:600;text-align:left;padding:8px 10px;border-bottom:1px solid var(--border);}
.tbl td{padding:8px 10px;border-bottom:1px solid var(--muted);color:var(--text);}
.tbl td.n,.tbl th.n{text-align:right;font-family:var(--font-mono);font-variant-numeric:tabular-nums;}
.tbl tr:last-child td{border-bottom:none;}
.scroll-x{overflow-x:auto;}

/* confusion matrix */
.cm{width:100%;border-collapse:separate;border-spacing:6px;table-layout:fixed;}
.cm th{font-size:13px;font-weight:600;color:var(--text-2);padding:6px 4px;text-align:center;}
.cm td{height:60px;text-align:center;border-radius:8px;font-family:var(--font-mono);font-size:15px;font-weight:600;font-variant-numeric:tabular-nums;}
.cm td small{display:block;font-size:13px;font-weight:500;opacity:.9;}
.cm th.rowh{text-align:right;padding-right:8px;}

/* stacked bar (distribusi severity) */
.stack{display:flex;height:14px;border-radius:9999px;overflow:hidden;background:var(--muted);margin:6px 0 12px;}
.stack div{height:100%;}
.tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;}
.tile{border-radius:10px;padding:10px 12px;border:1px solid;}
.tile .t{display:flex;align-items:center;gap:6px;font-size:13px;font-weight:600;}
.tile b{display:block;font-family:var(--font-mono);font-size:17px;margin-top:2px;font-variant-numeric:tabular-nums;}
.tile small{font-family:var(--font-mono);font-size:13px;}

/* callout */
.note{display:flex;gap:10px;align-items:flex-start;border-radius:10px;padding:12px 14px;font-size:14px;line-height:1.55;border:1px solid var(--border);background:var(--canvas);color:#334155;}
.note svg{margin-top:2px;}
.note.warn{background:#fffbeb;border-color:#fde68a;color:#78350f;}
.note.red{background:#fef2f2;border-color:#fecaca;color:#7f1d1d;}
.note.blue{background:#eff6ff;border-color:#bfdbfe;color:#1e3a8a;}

/* ---------- Widget Streamlit ---------- */
div[data-testid="stTextArea"] textarea,div[data-testid="stTextInput"] input{background:#fff!important;border:1px solid var(--border-2)!important;border-radius:10px!important;font-size:15px!important;line-height:1.55!important;color:var(--text)!important;}
div[data-testid="stTextArea"] textarea:focus,div[data-testid="stTextInput"] input:focus{border-color:var(--crimson)!important;box-shadow:0 0 0 2px rgba(170,0,21,.14)!important;}
div[data-baseweb="select"] > div{background:#fff!important;border:1px solid var(--border-2)!important;border-radius:10px!important;min-height:42px;font-size:15px;}
div[data-baseweb="select"] > div:focus-within{border-color:var(--crimson)!important;}
div[data-testid="stButton"] button,div[data-testid="stDownloadButton"] button{min-height:42px;border-radius:10px;font-size:15px;font-weight:600;border:1px solid var(--border);background:#fff;color:#334155;box-shadow:none;}
div[data-testid="stButton"] button:hover,div[data-testid="stDownloadButton"] button:hover{background:var(--muted);border-color:var(--border-2);color:var(--navy);}
div[data-testid="stButton"] button[kind="primary"],div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]{background:var(--crimson)!important;border-color:#8b0011!important;color:#fff!important;}
div[data-testid="stButton"] button[kind="primary"]:hover,div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover{background:#8b0011!important;}
div[data-testid="stButton"] button[kind="primary"] p,div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] p{color:#fff!important;}
div[data-testid="stButton"] button:disabled{opacity:.55;}
.st-key-chips div[data-testid="stButton"] button{min-height:34px;padding:4px 14px;border-radius:9999px;background:var(--muted);}
.st-key-chips div[data-testid="stButton"] button p{font-size:14px;font-weight:500;}
.st-key-chips div[data-testid="stButton"] button:hover{background:#fef2f2;border-color:#fecaca;color:var(--crimson);}
[class*="st-key-seg-"] div[role="radiogroup"]{gap:6px;flex-wrap:wrap;}
[class*="st-key-seg-"] div[role="radiogroup"] > label{margin:0;padding:6px 14px;border:1px solid var(--border);border-radius:9999px;background:#fff;cursor:pointer;}
[class*="st-key-seg-"] div[role="radiogroup"] > label > div:first-child:not([data-testid="stMarkdownContainer"]){display:none;}
[class*="st-key-seg-"] div[role="radiogroup"] > label p{margin:0;font-size:14px;font-weight:600;color:#475569;}
[class*="st-key-seg-"] div[role="radiogroup"] > label:has(input:checked){background:var(--navy);border-color:var(--navy);}
[class*="st-key-seg-"] div[role="radiogroup"] > label:has(input:checked) p{color:#fff;}
div[data-testid="stExpander"]{border:1px solid var(--border)!important;border-radius:12px!important;background:#fff;}
div[data-testid="stExpander"] summary p{font-size:15px;font-weight:600;color:var(--navy);}
button[data-baseweb="tab"] p{font-size:15px;font-weight:600;}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--crimson);}
div[data-baseweb="tab-highlight"]{background:var(--crimson)!important;}
div[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:10px;overflow:hidden;}
div[data-testid="stAlert"]{border-radius:10px;}
div[data-testid="stAlert"] p{font-size:15px;}
div[data-testid="stSlider"] [role="slider"]{background:var(--crimson)!important;}
div[data-testid="stCheckbox"] label p{font-size:14px;font-weight:500;color:#334155;}
.stApp .tbl th,.stApp .tbl td{border-top:0!important;border-left:0!important;border-right:0!important;}
.stApp .cm th,.stApp .cm td{border:0!important;}
.stApp .cm td{padding:0!important;}
/* Form tindak lanjut */
div[class*="st-key-card-"] div[data-testid="stForm"]{border:none!important;padding:0!important;background:transparent!important;}
div[data-testid="stForm"] div[data-testid="stVerticalBlock"]{gap:.85rem;}
div[data-testid="stTextArea"] textarea:disabled{background:var(--canvas)!important;color:#334155!important;-webkit-text-fill-color:#334155!important;opacity:1!important;}
div[data-testid="stFormSubmitButton"] button{min-height:42px;border-radius:10px;font-size:15px;font-weight:600;background:var(--crimson)!important;border:1px solid #8b0011!important;color:#fff!important;}
div[data-testid="stFormSubmitButton"] button:hover{background:#8b0011!important;}
div[data-testid="stFormSubmitButton"] button p{color:#fff!important;font-weight:600;}
.stApp span[data-baseweb="tag"]{background:#eef2ff!important;border:1px solid #c7d2fe!important;border-radius:8px!important;}
.stApp span[data-baseweb="tag"] span{color:#4338ca!important;font-size:14px;font-weight:600;}
.stApp span[data-baseweb="tag"] svg{fill:#4338ca!important;color:#4338ca!important;}
.fu-text{background:var(--canvas);border:1px solid var(--border);border-radius:10px;padding:12px 14px;font-size:15px;line-height:1.6;color:var(--text);white-space:pre-wrap;word-break:break-word;}
.fu-models{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;}
@media (max-width:1200px){.fu-models{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media (max-width:640px){.fu-models{grid-template-columns:minmax(0,1fr);}}
.fu-model{border:1px solid var(--border);border-radius:12px;padding:12px;background:#fff;display:flex;flex-direction:column;gap:8px;min-width:0;}
.fu-model.off{background:var(--canvas);}
.fu-model .h{display:flex;align-items:center;justify-content:space-between;gap:6px;flex-wrap:wrap;font-size:15px;font-weight:700;color:var(--navy);}
.fu-model .m{font-size:13px;color:var(--text-2);line-height:1.5;}
.fu-label{font-size:13px;font-weight:600;color:var(--text-2);margin:4px 0 6px;}
.fu-model .r{display:flex;flex-wrap:wrap;gap:6px;align-items:center;}
"""


# Susun CSS akhir (ikon dimasukkan ke placeholder).
def build_css() -> str:
    css = _CSS
    css = css.replace("%%IC_COMPARE%%", _icon_uri("compare")).replace("%%IC_CHART%%", _icon_uri("chart"))
    css = css.replace("%%IC_DB%%", _icon_uri("database"))
    return css


# Pasang CSS ke halaman.
def inject_styles() -> None:
    st.markdown(f"<style>{build_css()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pembangun HTML kecil
# ---------------------------------------------------------------------------
def pill(text: str, kind: str = "slate", dot: str | None = None, mono: bool = False, upper: bool = False) -> str:
    dot_html = f'<span class="dot" style="background:{dot}"></span>' if dot else ""
    cls = f"pill {kind}{' mono' if mono else ''}"
    return f'<span class="{cls}">{dot_html}{esc(text)}</span>'


# Lencana warna untuk satu severity.
def sev_badge(severity: str | None) -> str:
    if severity not in SEVERITY_COLORS:
        return pill("—")
    c = SEVERITY_COLORS[severity]
    return (
        f'<span class="sev" style="background:{c["bg"]};color:{c["text"]};border:1px solid {c["border"]}">'
        f'<span class="dot" style="background:{c["dot"]}"></span>{esc(severity)}</span>'
    )


# Batang progres dengan label dan nilai.
def bar(label: str, value_text: str, frac: float | None, color: str, label_color: str = "#475569") -> str:
    width = 0 if frac is None or not np.isfinite(frac) else max(0.0, min(frac, 1.0)) * 100
    return _h(f"""<div class="bar-row"><div class="bar-top"><span style="color:{label_color}">{esc(label)}</span>
        <b>{esc(value_text)}</b></div>
        <div class="track"><div class="fill" style="width:{width:.1f}%;background:{color}"></div></div></div>""")


# Buat tabel HTML sederhana; kolom angka rata kanan.
def table_html(headers: list[str], rows: list[list[Any]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(f'<th class="{"n" if i in numeric else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"n" if i in numeric else ""}">{cell if isinstance(cell, _Raw) else esc(cell)}</td>'
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f'<div class="scroll-x"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


class _Raw(str):
    """Penanda: string ini sudah berupa HTML aman (jangan di-escape lagi)."""


# Tandai teks sebagai HTML yang sudah aman.
def raw(markup: str) -> _Raw:
    return _Raw(markup)


# Kotak catatan berwarna.
def note(text_html: str, kind: str = "", icon_name: str = "alert") -> str:
    return f'<div class="note {kind}">{icon(icon_name, 16)}<div>{text_html}</div></div>'


# Judul bagian halaman.
def section_head(letter: str, title: str, sub: str, color: str, soft: str, right_html: str = "") -> str:
    return _h(
        f"""<div class="sec"><div class="sec-l"><div class="sec-badge" style="background:{soft};color:{color}">{esc(letter)}</div>
        <div><div class="sec-title">{esc(title)}</div><div class="sec-sub">{esc(sub)}</div></div></div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">{right_html}</div></div>"""
    )


# Pembungkus kartu.
def card(inner: str, extra_style: str = "") -> str:
    return f'<div class="card" style="{extra_style}">{inner}</div>'


# Kepala kartu: judul, ikon, dan sub-judul.
def card_head(title: str, icon_name: str | None = None, right_html: str = "", sub: str = "") -> str:
    ic = icon(icon_name, 18) if icon_name else ""
    sub_html = f'<div class="card-sub">{esc(sub)}</div>' if sub else ""
    return _h(
        f'<div class="card-head"><div><div class="card-title">{ic}{esc(title)}</div>{sub_html}</div>'
        f"<div>{right_html}</div></div>"
    )


# Kartu angka ringkas (KPI).
def kpi_card(
    label: str,
    value: str,
    sub: str = "",
    color: str = "var(--navy)",
    sub_color: str = "#475569",
    icon_name: str | None = None,
) -> str:
    ic = f'<span style="color:#94a3b8">{icon(icon_name, 18)}</span>' if icon_name else ""
    return _h(
        f"""<div class="card" style="padding:16px 18px"><div style="display:flex;justify-content:space-between;align-items:center">
        <span class="label-sm">{esc(label)}</span>{ic}</div>
        <div class="mono" style="font-size:26px;font-weight:700;color:{color};margin:6px 0 4px;line-height:1.2">{value}</div>
        <div style="font-size:13px;color:{sub_color}">{sub}</div></div>"""
    )


def stacked_bar_html(segments: list[tuple[str, int, str]]) -> str:
    """Bar horizontal tersusun untuk distribusi (warna solid, tanpa gradient)."""
    total = sum(v for _, v, _ in segments) or 1
    parts = "".join(
        f'<div title="{esc(name)}: {fmt_int(v)}" style="width:{v / total * 100:.2f}%;background:{color}"></div>'
        for name, v, color in segments
        if v
    )
    return f'<div class="stack">{parts}</div>'


# Grafik garis sederhana (SVG) untuk tren per tahun.
def trend_svg(points: dict[int, int], color: str = "#aa0015") -> str:
    if len(points) < 2:
        return '<div class="muted" style="font-size:14px">Data tren belum tersedia.</div>'
    years = sorted(points)
    values = [points[y] for y in years]
    w, h, pl, pr, pt, pb = 760, 260, 64, 40, 40, 36
    vmax = max(values) * 1.2 or 1
    xs = [pl + i * (w - pl - pr) / (len(years) - 1) for i in range(len(years))]
    ys = [pt + (1 - v / vmax) * (h - pt - pb) for v in values]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"{xs[0]:.1f},{h - pb} {line} {xs[-1]:.1f},{h - pb}"
    grid = ""
    for frac in (0, 0.25, 0.5, 0.75, 1):
        gy = pt + (1 - frac) * (h - pt - pb)
        grid += (
            f'<line x1="{pl}" x2="{w - pr}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3 4"/>'
            f'<text x="{pl - 8}" y="{gy + 4:.1f}" text-anchor="end" font-size="13" fill="#64748b" '
            f'font-family="JetBrains Mono,monospace">{fmt_int(vmax * frac)}</text>'
        )
    pts = ""
    peak = values.index(max(values))
    for i, (x, y, v) in enumerate(zip(xs, ys, values)):
        weight = "700" if i == peak else "500"
        pts += (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#fff" stroke="{color}" stroke-width="2.5"/>'
            f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" font-size="13" font-weight="{weight}" '
            f'fill="#0f172a" font-family="JetBrains Mono,monospace">{fmt_int(v)}</text>'
            f'<text x="{x:.1f}" y="{h - 8}" text-anchor="middle" font-size="14" fill="#475569" '
            f'font-family="Inter,sans-serif">{years[i]}</text>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Tren per tahun" style="display:block">'
        f'{grid}<polygon points="{area}" fill="{color}" fill-opacity="0.08"/><polyline points="{line}" fill="none" stroke="{color}" '
        f'stroke-width="2.5" stroke-linejoin="round"/>{pts}</svg>'
    )


def heat_table(cm: pd.DataFrame) -> str:
    """Confusion matrix: diagonal diwarnai warna severity, sel lain biru dengan intensitas sesuai nilai."""
    labels = list(cm.index)
    vmax_off = max([int(cm.loc[r, c]) for r in labels for c in labels if r != c] + [1])
    soft = {
        "LOW": ("#ecfdf5", "#047857"),
        "MEDIUM": ("#fffbeb", "#b45309"),
        "HIGH": ("#fff1f2", "#be123c"),
        "BLOCKER": ("#fef2f2", "#aa0015"),
    }
    head = (
        '<tr><th class="rowh">Aktual ↓ / Prediksi →</th>'
        + "".join(
            f'<th style="background:{soft[l][0]};color:{soft[l][1]};border-radius:8px;">{esc(l)}</th>' for l in labels
        )
        + "</tr>"
    )
    rows = ""
    for r in labels:
        row_total = int(cm.loc[r].sum()) or 1
        cells = ""
        for c in labels:
            v = int(cm.loc[r, c])
            share = f"{v / row_total * 100:.1f}%"
            if r == c:
                color = SEVERITY_COLORS[r]["dot"] if r != "BLOCKER" else "#aa0015"
                cells += f'<td style="background:{color};color:#fff">{fmt_int(v)}<small>{share}</small></td>'
            else:
                alpha = 0.06 + 0.5 * (v / vmax_off)
                cells += f'<td style="background:rgba(79,70,229,{alpha:.2f});color:#1e1b4b">{fmt_int(v)}<small>{share}</small></td>'
        rows += (
            f'<tr><th class="rowh">{esc(r)}<br><small style="font-weight:500;color:#64748b">n={fmt_int(row_total)}</small></th>'
            f"{cells}</tr>"
        )
    return f'<table class="cm"><thead>{head}</thead><tbody>{rows}</tbody></table>'


# =====================================================================
# Helper layout Streamlit
# =====================================================================
def columns(spec: Any, **kwargs: Any) -> list[Any]:
    try:
        return st.columns(spec, **kwargs)
    except TypeError:  # Streamlit lama tanpa vertical_alignment
        kwargs.pop("vertical_alignment", None)
        return st.columns(spec, **kwargs)


def keyed_container(key: str) -> Any:
    """Container bergaya kartu (kelas CSS st-key-<key>). Cadangan: container bergaris untuk Streamlit lama."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container(border=True)


# Tampilkan HTML di halaman Streamlit.
def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# Ringkasan: berapa model yang sudah terhubung.
def model_status_summary() -> dict[str, Any]:
    statuses = {k: model_status(k) for k in MODEL_ORDER}
    return {"statuses": statuses, "connected": sum(int(s["ready"]) for s in statuses.values()), "total": len(statuses)}


# Lencana status (hijau, kuning, atau merah) sesuai jumlah model terhubung.
def status_pill(connected: int, total: int) -> str:
    if connected == total:
        return pill(f"{connected}/{total} terhubung", "green", dot="#10b981")
    if connected == 0:
        return pill(f"{connected}/{total} terhubung", "red", dot="#d12027")
    return pill(f"{connected}/{total} terhubung", "amber", dot="#f59e0b")


# Bilah atas halaman (breadcrumb dan status).
def top_bar(section: str, active: str, right_html: str) -> None:
    md(_h(f"""<div class="topbar"><div class="crumbs"><span>{esc(section)}</span><span class="sep">/</span>
        <span class="active">{esc(active)}</span></div><div>{right_html}</div></div>"""))


PERSIST_KEYS = ("review_text", "text_mode", "model_choice", "thr", "fb")


def init_state() -> None:
    """Streamlit menghapus state widget yang tidak dirender (mis. saat pindah halaman). Nilai input
    disalin ke kunci '_p_*' di akhir halaman (save_state) lalu dipulihkan di sini."""
    if not st.session_state.get("_init_done"):
        samples = quick_samples()
        st.session_state["_p_text_mode"] = TEXT_MODE_NORMALIZED if samples else TEXT_MODE_RAW
        st.session_state["_p_review_text"] = samples[0]["text_norm"] if samples else ""
        st.session_state["sample_meta"] = {"source": "data uji SVM", **samples[0]} if samples else None
        st.session_state["latency_log"] = {}
        st.session_state["_init_done"] = True
    for key in PERSIST_KEYS:
        if key not in st.session_state and f"_p_{key}" in st.session_state:
            st.session_state[key] = st.session_state[f"_p_{key}"]


# Simpan isian penting agar tidak hilang saat pindah halaman.
def save_state() -> None:
    for key in PERSIST_KEYS:
        if key in st.session_state:
            st.session_state[f"_p_{key}"] = st.session_state[key]


# =====================================================================
# Sidebar
# =====================================================================
NAV_PAGES = ["Analisis Komparasi Model", "Evaluasi & Benchmark", "Eksplorasi Dataset Ulasan"]


# Sidebar: menu, pengaturan, dan pilihan model.
def render_sidebar() -> tuple[str, dict[str, Any]]:
    with st.sidebar:
        logo_path = ROOT / "assets" / "alfagift_logo.png"

        if logo_path.is_file():
            logo_data = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
            logo_html = (
                f'<img class="brand-logo-img" '
                f'src="data:image/png;base64,{logo_data}" '
                f'alt="Logo Alfagift">'
            )
        else:
            logo_html = f'<div class="brand-logo">{icon("book", 22)}</div>'

        md(
            _h(
                f"""<div class="brand">{logo_html}<div>
                <div class="brand-title">Alfagift Review Analytics</div>
                <div class="brand-sub">Dashboard Prototype</div></div></div>"""
            )
        )
        md('<div class="side-label module-label">Menu</div>')
        # Tombol navigasi dipakai alih-alih st.radio; routing tetap dikelola oleh session_state Streamlit.
        page = st.session_state.get("nav_page", NAV_PAGES[0])
        nav_icons = [":material/swap_horiz:", ":material/monitoring:", ":material/database:"]
        with keyed_container("nav"):
            for idx, nav_page in enumerate(NAV_PAGES):
                clicked = st.button(
                    nav_page,
                    key=f"nav_btn_{idx}",
                    type="primary" if page == nav_page else "secondary",
                    icon=nav_icons[idx],
                    width="stretch",
                )
                if clicked and page != nav_page:
                    st.session_state["nav_page"] = nav_page
                    st.rerun()
        md('<div class="side-label">Pengaturan inferensi</div>')
        device_label = st.selectbox(
            "Perangkat eksekusi",
            ["Otomatis (GPU bila tersedia)", "CPU saja"],
            help="Transformer di GPU 2 GB bisa kehabisan VRAM (terutama XLM-R); bila terjadi, "
            "prediksi otomatis diulang di CPU.",
        )
        low_memory = st.checkbox(
            "Hemat memori (muat satu model per waktu)",
            value=True,
            help="Model issue dan severity tidak ditahan bersamaan; lebih lambat, tetapi aman "
            "untuk RAM 8 GB / VRAM 2 GB.",
        )
        if st.button("Bersihkan cache model"):
            release_transformers()
            load_svm_bundle.clear()
            st.success("Model dibuang dari memori.")
        summary = model_status_summary()
        md(_h(f"""<div class="side-status"><div class="side-row"><span>Status model</span>
            {status_pill(summary["connected"], summary["total"])}</div>
            <div class="side-row"><span>Lingkungan</span><span class="mono" style="font-weight:600">
            {esc(torch_status())}</span></div></div>
            <div class="side-user"><div class="side-avatar">{icon("cap", 16)}</div><div>
            <div style="font-size:14px;font-weight:600;color:#0f172a">Penelitian Darrell Jeremy</div>
            <div style="font-size:13px;color:#64748b">Dashboard Prototype</div></div></div>"""))
    return page, {"device_pref": "cpu" if device_label.startswith("CPU") else "auto", "low_memory": low_memory}


# =====================================================================
# Halaman 1: Analisis Komparasi Model
# =====================================================================
def _apply_sample(sample: dict[str, str]) -> None:
    st.session_state["review_text"] = sample["text_norm"]
    st.session_state["text_mode"] = TEXT_MODE_NORMALIZED
    st.session_state["sample_meta"] = {"source": "data uji SVM", **sample}


# Isi kotak ulasan dengan satu ulasan acak dari dataset.
def _apply_random() -> None:
    df = get_dataset()
    if df.empty or "text_norm" not in df.columns:
        st.session_state["sample_meta"] = None
        return
    row = df.dropna(subset=["text_norm"]).sample(1, random_state=int(time.time() * 1000) % 1_000_000).iloc[0]
    st.session_state["review_text"] = str(row["text_norm"])
    st.session_state["text_mode"] = TEXT_MODE_NORMALIZED
    st.session_state["sample_meta"] = {
        "source": "korpus (bisa termasuk data latih)",
        "review": str(row.get("review", "")),
        "issue_true": str(row.get("category_Issue", "")),
        "severity_true": str(row.get("severity_Issue", "")),
        "text_norm": str(row["text_norm"]),
    }


# Kosongkan kotak ulasan dan hasil prediksi.
def _reset_input() -> None:
    st.session_state["review_text"] = ""
    st.session_state["sample_meta"] = None
    st.session_state.pop("pred_store", None)


# Chip kecil untuk satu kategori, boleh dengan skor.
def _cat_chip(label: str, score: float | None = None) -> str:
    extra = f" <small>{score:.2f}</small>" if score is not None else ""
    return f'<span class="chip" title="{esc(label)}">{esc(ISSUE_DISPLAY.get(label, label))}{extra}</span>'


def severity_block_html(
    model_key: str, issues: list[str], severity: str | None, sev_by_cat: list[tuple], basis: str
) -> str:
    """Tampilan severity yang langsung terlihat (tanpa expander).

    Transformer dan Linear SVM sama-sama menampilkan SETIAP kategori terpilih bersama severity-nya sendiri.
    Kategori tanpa severity ditandai 'Belum tersedia'. Severity tertinggi hanya ringkasan.
    """
    if not issues:
        return '<div class="sev-unavail">Severity tidak dihitung karena tidak ada kategori di atas threshold.</div>'
    sev_map = {t[0]: t[1] for t in sev_by_cat}
    rows = "".join(
        f'<div class="sev-row">{_cat_chip(c)}'
        f'{sev_badge(sev_map[c]) if sev_map.get(c) else pill(NA_TEXT, "slate")}</div>'
        for c in issues
    )
    top = highest_severity(list(sev_map.values()))
    top_html = f'<div class="sev-top">Severity tertinggi: <b>{esc(top)}</b></div>' if top else ""
    return f'<div class="sev-list">{rows}</div>{top_html}'


# Kartu hasil satu model: kategori, severity, dan rincian skor.
def result_card_html(key: str, res: dict[str, Any]) -> str:
    name, sub, color = MODEL_NAMES[key], MODEL_SUBTITLES[key], MODEL_COLORS[key]
    head = lambda right: (
        f'<div class="rc-head"><div><div class="rc-name">'  # noqa: E731
        f'<span class="dot" style="background:{color};margin-right:8px;vertical-align:middle"></span>'
        f'{esc(name)}</div><div class="rc-sub">{esc(sub)}</div></div><div>{right}</div></div>'
    )
    if res["kind"] == "unavailable":
        miss = []
        for task, info in res["status"]["tasks"].items():
            if not info["ready"]:
                miss.append(f"<b>{esc(task)}</b>: " + esc("; ".join(info["missing"] + info["problems"])))
        body = (
            f'<div class="rc-block"><div class="rc-label">Status</div>{pill("Prediksi tidak dijalankan", "slate")}</div>'
            f'<div class="card-sub" style="margin-top:12px"><b style="color:#334155">Yang belum ada:</b><br>{"<br>".join(miss)}</div>'
        )
        return _h(f'<div class="card rc off">{head(pill("Belum terhubung", "amber", dot="#f59e0b"))}{body}</div>')
    if res["kind"] == "error":
        return _h(
            f'<div class="card rc">{head(pill("Gagal", "red"))}<div class="err" style="margin-top:12px">'
            f'{esc(res["message"])}</div></div>'
        )

    s = summarize_model_result(key, res)
    time_pill = pill(fmt_duration(res.get("seconds")), "slate", mono=True)
    if res["kind"] == "svm":
        picked = res["issue_predicted"]
        scores = dict(res["issue_scores"])
        chips = (
            "".join(_cat_chip(c, scores.get(c)) for c in picked) or '<span class="chip none">(tidak ada label)</span>'
        )
        rows_scores = [[l, f"{sc:.3f}", "Ya" if l in picked else ""] for l, sc in res["issue_scores"]]
        detail = (
            "<details><summary>Rincian skor margin (decision_function)</summary>"
            + table_html(["Kategori", "Margin", "Terpilih"], rows_scores, {1})
            + "</details>"
        )
        score_note = "Skor = margin SVM"
    else:
        sel = res["issue"]["selected"]
        chips = "".join(_cat_chip(c, sc) for c, sc in sel) or '<span class="chip none">(tidak ada label)</span>'
        selected_set = {c for c, _ in sel}
        rows_scores = [[l, f"{sc:.4f}", "Ya" if l in selected_set else ""] for l, sc in res["issue"]["scores"]]
        detail = (
            "<details><summary>Rincian skor sigmoid</summary>"
            + table_html(["Kategori", "Skor sigmoid", "Terpilih"], rows_scores, {1})
            + "</details>"
        )
        score_note = "Skor = sigmoid (bukan probabilitas terkalibrasi)"
        if res["issue"]["used_fallback"]:
            score_note += " • fallback aktif"

    sev_block = severity_block_html(key, s["issues"], s["severity"], s["severity_by_category"], s["severity_basis"])
    warns = "".join(
        f'<div class="card-sub" style="margin-top:10px;color:#b45309">{esc(w)}</div>' for w in res.get("warnings", [])
    )
    msg = (
        f'<div class="card-sub" style="margin-top:8px;color:#b45309">{esc(s["message"])}</div>' if s["message"] else ""
    )
    return _h(f"""<div class="card rc">{head(time_pill)}
        <div class="rc-block"><div class="rc-label">Kategori terprediksi</div><div class="chips">{chips}</div></div>
        <div class="rc-block"><div class="rc-label">Tingkat keparahan (severity)</div>{sev_block}</div>
        <div class="card-sub" style="margin-top:12px">{esc(score_note)} • perangkat: {esc(str(res.get("device", "cpu")).upper())}</div>
        {msg}{warns}{detail}</div>""")


# Jalankan semua model terpilih pada satu ulasan.
def run_comparison(
    keys: list[str], text: str, threshold: float, fallback: bool, settings: dict[str, Any]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for key in keys:
        status = model_status(key)
        if not status["ready"]:
            results[key] = {"kind": "unavailable", "status": status}
            continue
        try:
            with st.spinner(f"Menjalankan {MODEL_NAMES[key]}…"):
                res = run_prediction(key, text, threshold, fallback, settings["device_pref"], settings["low_memory"])
            results[key] = res
            st.session_state.setdefault("latency_log", {})[key] = res.get("seconds")
        except (MemoryError, ImportError) as exc:
            results[key] = {"kind": "error", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            results[key] = {"kind": "error", "message": f"{type(exc).__name__}: {exc}"}
    return {"results": results, "order": keys, "text": text}


# ---------------------------------------------------------------------------
# Form Tindak Lanjut Manual & Riwayat Review (data permanen di SQLite)
# ---------------------------------------------------------------------------
def _form_submit(label: str, **kwargs: Any) -> bool:
    try:
        return st.form_submit_button(label, width="stretch", **kwargs)
    except TypeError:
        return st.form_submit_button(label, use_container_width=True, **kwargs)


# Tombol unduh selebar kolom; ada cadangan untuk Streamlit lama.
def _download(label: str, **kwargs: Any) -> None:
    try:
        st.download_button(label, width="stretch", **kwargs)
    except TypeError:
        st.download_button(label, use_container_width=True, **kwargs)


# Nama kategori beserta kodenya.
def _issue_label(code: str) -> str:
    return f"{ISSUE_DISPLAY.get(code, code)} ({code})"


# Lencana status tindak lanjut.
def _status_pill(status: str) -> str:
    kind, dot = FOLLOWUP_STATUS_PILL.get(status, ("slate", "#64748b"))
    return pill(status, kind, dot=dot)


def _fu_model_html(
    model_key: str,
    run_status: str,
    issues: list[str],
    severity: str | None,
    basis: str,
    sev_by_cat: list[tuple],
    message: str,
) -> str:
    """Kartu ringkas hasil satu model (dipakai di form dan di detail riwayat). Severity tampil langsung."""
    color = MODEL_COLORS[model_key]
    head = (
        f'<div class="h"><span><span class="dot" style="background:{color};margin-right:6px;vertical-align:middle">'
        f"</span>{esc(MODEL_NAMES[model_key])}</span></div>"
    )
    if run_status != "ok":
        kind = {"error": "red", "unavailable": "amber"}.get(run_status, "slate")
        msg = f'<div class="m">{esc(message[:180])}</div>' if message else ""
        return f'<div class="fu-model off">{head}<div>{pill(RUN_STATUS_TEXT.get(run_status, run_status), kind)}</div>{msg}</div>'
    chips = "".join(_cat_chip(c) for c in issues) or '<span class="chip none">(tidak ada label)</span>'
    sev_html = severity_block_html(model_key, issues, severity, sev_by_cat, basis)
    msg = f'<div class="m" style="color:#b45309">{esc(message)}</div>' if message else ""
    return f'<div class="fu-model">{head}<div class="r">{chips}</div>{sev_html}{msg}</div>'


# Susun argumen kartu ringkas dari hasil satu model.
def _summary_card_args(s: dict[str, Any]) -> tuple:
    return (
        s["model_key"],
        s["run_status"],
        s["issues"],
        s["severity"],
        s["severity_basis"],
        [(c, sev) for c, sev, _ in s["severity_by_category"]],
        s["message"],
    )


# Hitung berapa kali hasil run ini sudah disimpan.
def _count_run_submissions(run_id: str | None) -> int:
    if not run_id:
        return 0
    try:
        with closing(_db_connect(init_followup_db())) as con:
            return int(con.execute("SELECT COUNT(*) FROM reviews WHERE prediction_run_id = ?", (run_id,)).fetchone()[0])
    except sqlite3.Error:
        return 0


# Form tindak lanjut: simpan hasil prediksi ke database.
def render_followup_form(store: dict[str, Any]) -> None:
    summaries = [summarize_model_result(k, store["results"].get(k)) for k in MODEL_ORDER]
    n_ok = sum(s["run_status"] == "ok" for s in summaries)
    n_run = sum(s["run_status"] != "not_run" for s in summaries)
    predicted_by = {}
    for s in summaries:
        for c in s["issues"]:
            predicted_by[c] = predicted_by.get(c, 0) + 1

    md(
        section_head(
            "",
            "Tindak Lanjut Manual",
            "Hasil prediksi di atas disimpan bersama isian Anda ke database SQLite.",
            "#aa0015",
            "#fef2f2",
            pill("Disimpan permanen", "slate"),
        )
    )
    with keyed_container("card-followup"):
        md(
            card_head(
                "Form tindak lanjut",
                "message",
                right_html=pill(
                    f"{n_ok} dari {len(MODEL_ORDER)} model berhasil", "green" if n_ok == len(MODEL_ORDER) else "amber"
                ),
            )
        )
        md(_h(f"""<div><div class="fu-label">Teks ulasan yang diprediksi</div>
            <div class="fu-text">{esc(store["text"])}</div></div>"""))
        current = st.session_state.get("review_text") or ""
        if current.strip() and current.strip() != (store.get("input_text") or store["text"]).strip():
            st.caption(
                "Teks di kotak input sudah berubah setelah prediksi. Form ini tetap memakai teks yang benar-benar "
                "diprediksi (di atas); tekan Prediksi lagi untuk menindaklanjuti teks baru."
            )
        md(
            '<div class="fu-label">Hasil prediksi yang ikut disimpan</div>'
            + '<div class="fu-models">'
            + "".join(_fu_model_html(*_summary_card_args(s)) for s in summaries)
            + "</div>"
        )
        if n_run < len(MODEL_ORDER):
            md(
                note(
                    f"Prediksi ini hanya menjalankan <b>{n_run}</b> model. Model yang tidak dipilih disimpan dengan status "
                    "“Tidak dijalankan”. Pilih “Semua Model” lalu tekan Prediksi untuk menyimpan keempat model.",
                    "warn",
                )
            )
        if n_ok == 0:
            md(
                note(
                    "Belum ada model yang berhasil menghasilkan prediksi, jadi tindak lanjut belum bisa disimpan.",
                    "red",
                )
            )
            return
        already = _count_run_submissions(store.get("run_id"))
        if already:
            st.caption(
                f"Hasil prediksi ini sudah disubmit {already} kali. Submit lagi akan menambah review baru "
                "(data lama tidak ditimpa)."
            )

        with st.form("fu_form", clear_on_submit=True, border=False):
            issues = st.multiselect(
                "Issue yang ingin ditinjau",
                TRANSFORMER_ISSUE_LABELS,
                format_func=lambda c: _issue_label(c)
                + (f" • diprediksi {predicted_by[c]} model" if c in predicted_by else ""),
                placeholder="Pilih satu atau lebih kategori masalah…",
            )
            solution = st.text_area(
                "Usulan solusi atau tindak lanjut",
                height=110,
                placeholder="Tulis usulan solusi / tindakan yang perlu dilakukan…",
            )
            notes = st.text_area("Catatan tambahan (opsional)", height=80, placeholder="Catatan tambahan bila ada…")
            c_status, c_btn = columns([2.2, 1], vertical_alignment="bottom")
            status = c_status.selectbox("Status tindak lanjut", FOLLOWUP_STATUSES)
            with c_btn:
                submitted = _form_submit("Submit tindak lanjut", type="primary")
        if submitted:
            problems = []
            if not issues:
                problems.append("pilih minimal satu issue yang ingin ditinjau")
            if not solution.strip():
                problems.append("isi usulan solusi atau tindak lanjut")
            if problems:
                st.warning("Belum disimpan: " + " dan ".join(problems) + ". Isian form perlu diisi ulang.")
            else:
                try:
                    rid = save_followup(store, issues, solution, notes, status)
                    st.success(f"Tindak lanjut tersimpan dengan ID {rid}. Lihat di Riwayat Review di bawah.")
                except (sqlite3.Error, OSError, ValueError) as exc:
                    st.error(f"Gagal menyimpan ke database: {exc}")


# Riwayat perubahan status satu review.
def _load_status_history(review_id: str) -> pd.DataFrame:
    with closing(_db_connect(init_followup_db())) as con:
        return pd.read_sql_query(
            "SELECT changed_at, old_status, new_status FROM status_history WHERE review_id = ? " "ORDER BY id",
            con,
            params=(review_id,),
        )


# Daftar riwayat tindak lanjut dari database.
def render_followup_history() -> None:
    try:
        reviews, preds, details = load_followups()
    except (sqlite3.Error, OSError) as exc:
        md(
            note(
                f"<b>Database tindak lanjut tidak bisa dibaca.</b> {esc(exc)} • Lokasi: <code>{esc(FOLLOWUP_DB_PATH)}</code>",
                "red",
            )
        )
        return
    n = len(reviews)
    md(
        section_head(
            "",
            "Riwayat Review & Export",
            "Seluruh tindak lanjut yang sudah disubmit, tersimpan permanen.",
            "#aa0015",
            "#fef2f2",
            pill(f"{fmt_int(n)} review tersimpan", "slate", mono=True),
        )
    )
    counts = reviews["status"].value_counts().to_dict() if n else {}
    md(
        '<div class="g4">'
        + kpi_card("Total review", fmt_int(n), "submission tersimpan", icon_name="database")
        + "".join(
            kpi_card(
                s,
                fmt_int(counts.get(s, 0)),
                f"{counts.get(s, 0) / n * 100:.0f}% dari total" if n else "—",
                color=FOLLOWUP_STATUS_PILL[s][1] if s != "Dalam Peninjauan" else "#b45309",
                icon_name="clock",
            )
            for s in FOLLOWUP_STATUSES
        )
        + "</div>"
    )
    if n == 0:
        md(
            note(
                f"Belum ada review yang disubmit. Jalankan prediksi lalu isi form tindak lanjut di atas. "
                f"Database: <code>{esc(FOLLOWUP_DB_PATH)}</code>",
                "blue",
                "database",
            )
        )
        return

    review_df = build_review_analysis(reviews, preds)
    detail_df = build_issue_details(reviews, details)

    # --- daftar + filter ---
    with keyed_container("card-history"):
        md(
            card_head(
                "Daftar review tersimpan",
                "list",
                sub="Teks ulasan, prediksi keempat model, issue ditinjau, solusi, status, dan waktu submit.",
            )
        )
        f_status, f_issue, f_search = columns([1.2, 1.4, 1.4])
        sel_status = f_status.multiselect(
            "Filter status tindak lanjut", FOLLOWUP_STATUSES, key="fu_f_status", placeholder="Semua status"
        )
        sel_issue = f_issue.multiselect(
            "Filter kategori issue (yang ditinjau)",
            TRANSFORMER_ISSUE_LABELS,
            format_func=_issue_label,
            key="fu_f_issue",
            placeholder="Semua kategori",
        )
        query = f_search.text_input("Cari teks / solusi / ID", key="fu_f_search", placeholder="mis. login, RV-2026…")
        mask = pd.Series(True, index=reviews.index)
        if sel_status:
            mask &= reviews["status"].isin(sel_status)
        if sel_issue:
            mask &= reviews["issues_reviewed"].map(lambda v: bool(set(parse_label_list(v)) & set(sel_issue)))
        if query.strip():
            q = query.strip().lower()
            hay = (
                reviews["review_text"].fillna("")
                + " "
                + reviews["solution"].fillna("")
                + " "
                + reviews["notes"].fillna("")
                + " "
                + reviews["review_id"]
            ).str.lower()
            mask &= hay.str.contains(q, regex=False)
        filtered = reviews[mask]
        pred_idx = {(r.review_id, r.model_key): r for r in preds.itertuples(index=False)}

        # Teks sel tabel: kategori dan severity hasil satu model.
        def model_cell(rid: str, key: str) -> str:
            p = pred_idx.get((rid, key))
            if p is None or p.run_status != "ok":
                return RUN_STATUS_TEXT.get(p.run_status if p is not None else "not_run")
            return f"{p.issue_pred or NO_LABEL_TEXT} → {p.severity_pred or '—'}"

        table = pd.DataFrame(
            {
                "No": filtered["id"].values,
                "Review ID": filtered["review_id"].values,
                "Status": filtered["status"].values,
                "Teks ulasan": filtered["review_text"].values,
                **{MODEL_NAMES[k]: [model_cell(rid, k) for rid in filtered["review_id"]] for k in MODEL_ORDER},
                "Issue ditinjau": filtered["issues_reviewed"].map(_issue_names).values,
                "Usulan solusi": filtered["solution"].values,
                "Catatan": filtered["notes"].fillna("").values,
                "Waktu submit": filtered["submitted_at"].values,
                "Diperbarui": filtered["updated_at"].values,
            }
        ).iloc[::-1]
        show_df(
            table,
            hide_index=True,
            column_config={
                "Teks ulasan": st.column_config.TextColumn(width="large"),
                "Usulan solusi": st.column_config.TextColumn(width="medium"),
            },
        )
        st.caption(
            f"Menampilkan {fmt_int(len(filtered))} dari {fmt_int(n)} review (terbaru di atas). "
            "Format kolom model: issue terprediksi → severity."
        )

    # --- detail & ubah status ---
    with keyed_container("card-history-status"):
        md(
            card_head(
                "Detail & ubah status",
                "clock",
                sub="Hanya kolom status yang diubah; teks, prediksi, solusi, dan catatan tetap seperti saat disubmit.",
            )
        )
        options = filtered["review_id"].tolist()[::-1] or reviews["review_id"].tolist()[::-1]
        if st.session_state.get("fu_sel_review") not in options:
            st.session_state.pop("fu_sel_review", None)
        by_id = reviews.set_index("review_id")

        # Teks pilihan review (nomor, id, status, potongan ulasan).
        def fmt_review(rid: str) -> str:
            r = by_id.loc[rid]
            snippet = str(r["review_text"])
            snippet = snippet if len(snippet) <= 60 else snippet[:59].rstrip() + "…"
            return f"#{int(r['id'])} • {rid} • {r['status']} • “{snippet}”"

        c_sel, c_new, c_btn = columns([3, 1.4, 1], vertical_alignment="bottom")
        rid = c_sel.selectbox("Pilih review", options, format_func=fmt_review, key="fu_sel_review")
        cur = by_id.loc[rid, "status"]
        c_new.selectbox(
            "Status baru",
            FOLLOWUP_STATUSES,
            index=FOLLOWUP_STATUSES.index(cur) if cur in FOLLOWUP_STATUSES else 0,
            key=f"fu_new_status_{rid}",
        )
        with c_btn:
            save_status_clicked = show_button("Simpan status", type="primary", key="fu_save_status")

        if save_status_clicked:
            new_status = st.session_state.get(f"fu_new_status_{rid}")
            if not rid:
                st.error("Review belum dipilih.")
            elif new_status not in FOLLOWUP_STATUSES:
                st.error("Status baru tidak terbaca. Pilih kembali status lalu tekan Simpan status.")
            else:
                try:
                    changed = update_followup_status(rid, new_status)
                    if changed:
                        st.success(f"Status {rid} berhasil diubah menjadi '{new_status}'.")
                    else:
                        st.info(f"Status {rid} sudah '{new_status}'. Tidak ada perubahan.")
                except (sqlite3.Error, OSError, ValueError, KeyError) as exc:
                    st.error(f"Gagal mengubah status: {exc}")

        r = by_id.loc[rid]
        cards = []
        for key in MODEL_ORDER:
            p = pred_idx.get((rid, key))
            if p is None:
                cards.append(_fu_model_html(key, "not_run", [], None, "", [], ""))
                continue
            sev_cat = [
                tuple(part.split("=", 1)) for part in str(p.severity_by_category or "").split("; ") if "=" in part
            ]
            cards.append(
                _fu_model_html(
                    key,
                    p.run_status,
                    parse_label_list(p.issue_pred),
                    p.severity_pred,
                    p.severity_basis or "",
                    sev_cat,
                    p.message or "",
                )
            )
        issue_chips = "".join(_cat_chip(c) for c in parse_label_list(r["issues_reviewed"]))
        notes_html = esc(r["notes"]) if isinstance(r["notes"], str) and r["notes"] else '<span class="muted">—</span>'
        md(_h(f"""<div style="display:flex;flex-direction:column;gap:12px">
            <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">{_status_pill(r["status"])}
            {pill(rid, "slate", mono=True)}{pill("Submit " + str(r["submitted_at"]), "slate", mono=True)}
            {pill("Diperbarui " + str(r["updated_at"]), "slate", mono=True)}</div>
            <div><div class="fu-label">Teks ulasan</div><div class="fu-text">{esc(r["review_text"])}</div></div>
            <div><div class="fu-label">Hasil prediksi saat submit</div><div class="fu-models">{"".join(cards)}</div></div>
            <div class="g2"><div><div class="fu-label">Issue ditinjau</div><div class="chips">{issue_chips}</div></div>
            <div><div class="fu-label">Catatan tambahan</div><div class="fu-text">{notes_html}</div></div></div>
            <div><div class="fu-label">Usulan solusi / tindak lanjut</div><div class="fu-text">{esc(r["solution"])}</div></div>
            </div>"""))
        try:
            hist = _load_status_history(rid)
            if not hist.empty:
                rows = [
                    [h.changed_at, h.old_status or "— (submit awal)", raw(_status_pill(h.new_status))]
                    for h in hist.itertuples()
                ]
                with st.expander(f"Riwayat perubahan status ({len(hist)})"):
                    md(table_html(["Waktu", "Status lama", "Status baru"], rows))
        except sqlite3.Error:
            pass

    # --- export ---
    with keyed_container("card-export"):
        md(
            card_head(
                "Export data",
                "download",
                right_html=pill(f"{fmt_int(n)} review • {fmt_int(len(detail_df))} baris detail", "slate", mono=True),
                sub="Selalu berisi SELURUH data di database (tidak terpengaruh filter). Review ID menghubungkan kedua sheet Excel.",
            )
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        e1, e2, e3 = columns(3)
        with e1:
            _download(
                "Export CSV (Review Analysis)",
                data=export_csv_bytes(review_df),
                file_name=f"alfagift_review_analysis_{stamp}.csv",
                mime="text/csv",
                key="fu_dl_csv",
            )
        with e2:
            _download(
                "Export CSV (Issue Details)",
                data=export_csv_bytes(detail_df),
                file_name=f"alfagift_issue_details_{stamp}.csv",
                mime="text/csv",
                key="fu_dl_csv_detail",
            )
        with e3:
            try:
                xlsx = export_excel_bytes(review_df, detail_df)
                _download(
                    "Export Excel (.xlsx)",
                    data=xlsx,
                    file_name=f"alfagift_review_followup_{stamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="fu_dl_xlsx",
                )
            except ImportError:
                st.warning("Ekspor Excel memerlukan paket openpyxl: pip install openpyxl")
        st.caption(f"Database: {FOLLOWUP_DB_PATH}")


# Halaman Komparasi Model: prediksi satu ulasan di semua model.
def page_comparison(settings: dict[str, Any]) -> None:
    init_state()
    summary = model_status_summary()
    top_bar("Analisis", "Komparasi Model", status_pill(summary["connected"], summary["total"]))

    # --- banner ---
    tests = {k: split_counts(find_summary(k)[1])["review"].get("test") for k in MODEL_ORDER}
    t_tr = next((tests[k] for k in TRANSFORMER_MODELS if tests[k]), None)
    test_txt = f"{fmt_int(t_tr)} ulasan" if t_tr else NA_TEXT
    md(_h(f"""<div class="card"><div class="banner">
        <h1>Komparasi Klasifikasi Isu &amp; Severity</h1>
        <p>Masukkan ulasan Alfagift atau pilih contoh dari data uji, lalu bandingkan prediksi empat model.</p>
        <div class="kpi-row" style="margin-top:16px"><div class="kpi-box"><span class="l">Dataset uji</span><span class="v">{esc(test_txt)}</span></div>
        <div class="kpi-box"><span class="l">Target 1</span><span class="v">10 kategori isu</span></div>
        <div class="kpi-box"><span class="l">Target 2</span><span class="v">4 level severity</span></div></div>
        </div></div>"""))

    # --- input ---
    with keyed_container("card-input"):
        md(card_head("Input ulasan", "message"))
        samples = quick_samples()
        with keyed_container("chips"):
            chip_cols = columns(len(samples) + 1 if samples else 1)
            for i, (col, sample) in enumerate(zip(chip_cols, samples)):
                label = sample["review"] if len(sample["review"]) <= 34 else sample["review"][:33].rstrip() + "…"
                col.button(
                    f"“{label}”",
                    key=f"chip_{i}",
                    on_click=_apply_sample,
                    args=(sample,),
                    help=f"Contoh data uji • label asli: {sample['issue_true']} / {sample['severity_true']}",
                )
            chip_cols[-1].button(
                "Acak dari korpus",
                key="chip_random",
                on_click=_apply_random,
                help="Mengambil satu ulasan acak dari alfagift_issue_labeled_merged.csv",
            )
        st.text_area(
            "Teks ulasan",
            key="review_text",
            height=112,
            label_visibility="collapsed",
            placeholder="Tulis ulasan aplikasi Alfagift di sini…",
        )
        meta = st.session_state.get("sample_meta")
        if meta and meta.get("text_norm") == st.session_state.get("review_text"):
            st.caption(
                f"Contoh dari {meta['source']} • ulasan asli: “{meta['review'][:140]}” • label asli: "
                f"issue = {meta['issue_true']}, severity = {meta['severity_true']}"
            )

        c_model, c_count, c_reset, c_go = columns([3, 1, 1.6, 2], vertical_alignment="bottom")
        choice = c_model.selectbox(
            "Model evaluasi",
            ["all"] + MODEL_ORDER,
            key="model_choice",
            format_func=lambda k: (
                "Semua model (bandingkan 4 sekaligus)"
                if k == "all"
                else f"{MODEL_NAMES[k]} — {MODEL_SUBTITLES[k].split('•')[-1].strip()}"
            ),
        )
        text_now = st.session_state.get("review_text") or ""
        c_count.markdown(
            f'<div class="mono" style="font-size:13px;color:#475569;padding-bottom:10px">'
            f'<b style="color:#0f172a">{len(text_now)}</b> karakter</div>',
            unsafe_allow_html=True,
        )
        with c_reset:
            show_button("Reset", on_click=_reset_input)
        with c_go:
            go = show_button("Prediksi", type="primary", disabled=not text_now.strip())
        with keyed_container("seg-mode"):
            mode = st.radio(
                "Format teks yang dimasukkan", [TEXT_MODE_RAW, TEXT_MODE_NORMALIZED], key="text_mode", horizontal=True
            )
        if mode == TEXT_MODE_RAW and not NORMALIZER_VERIFIED:
            md(
                note(
                    "<b>Preprocessing belum terverifikasi.</b> Model dilatih pada kolom <code>text_norm</code>, "
                    "tetapi fungsi normalisasinya belum ada di aplikasi ini. Untuk ulasan mentah, aplikasi hanya membuang spasi di tepi, "
                    "sehingga hasilnya belum layak menjadi bukti penelitian. Pilih contoh di atas atau tempel teks dari kolom <code>text_norm</code>.",
                    "warn",
                )
            )
        with st.expander("Threshold & opsi lanjutan"):
            p1, p2 = columns(2)
            threshold = p1.slider(
                "Threshold sigmoid issue (Transformer)",
                0.10,
                0.90,
                DEFAULT_THRESHOLD,
                0.05,
                key="thr",
                help="0,50 adalah threshold yang dipakai pada evaluasi di notebook.",
            )
            fallback = p2.checkbox(
                "Fallback ke kategori skor tertinggi",
                value=False,
                key="fb",
                help="Tidak ada dalam protokol evaluasi; hanya untuk eksplorasi.",
            )

    keys = MODEL_ORDER if choice == "all" else [choice]
    if go and text_now.strip():
        model_text = text_now.strip() if mode == TEXT_MODE_NORMALIZED else normalize_review_text(text_now)
        new_store = run_comparison(keys, model_text, threshold, fallback, settings)
        # Metadata untuk Form Tindak Lanjut (tidak memengaruhi prediksi).
        new_store.update(
            run_id=uuid.uuid4().hex,
            input_text=text_now,
            text_mode=mode,
            threshold=float(threshold),
            fallback=bool(fallback),
            predicted_at=_now_str(),
        )
        st.session_state["pred_store"] = new_store

    store = st.session_state.get("pred_store")
    if store:
        ok = sum(1 for r in store["results"].values() if r["kind"] in ("svm", "transformer"))
        md(_h(f"""<div class="sec" style="border:none;margin-top:8px;padding-bottom:0"><div class="sec-l">
            <div class="sec-title" style="font-size:20px">Hasil prediksi</div></div>
            <div>{pill(f"{ok} dari {len(store['order'])} model menghasilkan prediksi", "slate")}</div></div>"""))
        md('<div class="g2">' + "".join(result_card_html(k, store["results"][k]) for k in store["order"]) + "</div>")
        md(
            note(
                "Skor sigmoid dan margin menunjukkan keyakinan relatif model, <b>bukan</b> probabilitas terkalibrasi. "
                "Severity semua model dihitung per kategori terprediksi. "
                "Waktu mencakup pemuatan model bila mode hemat memori aktif.",
                "blue",
                "shield",
            )
        )
        render_followup_form(store)
    render_followup_history()
    save_state()


# =====================================================================
# Halaman 2: Evaluasi & Benchmark
# =====================================================================
def _summaries() -> dict[str, dict[str, Any] | None]:
    return {k: find_summary(k)[1] for k in MODEL_ORDER}


# Kartu metrik utama satu model.
def _metric_card(key: str, task: str, summ: dict[str, Any] | None) -> str:
    color, name = MODEL_COLORS[key], MODEL_NAMES[key]
    g = lambda m: metric_value(summ, task, m)  # noqa: E731
    if task == "issue":
        bars = bar("Macro-F1", fmt_pct(g("macro_f1"), 1), g("macro_f1"), color) + bar(
            "Micro-F1", fmt_pct(g("micro_f1"), 1), g("micro_f1"), "#64748b"
        )
        mini = (
            f'<div class="mini2"><div><span>Macro P</span><b>{esc(fmt_pct(g("macro_precision"), 1))}</b></div>'
            f'<div><span>Macro R</span><b>{esc(fmt_pct(g("macro_recall"), 1))}</b></div></div>'
        )
        return _h(
            f'<div class="card mc"><div class="mc-head"><div class="mc-name"><span class="dot" style="background:{color}"></span>{esc(name)}</div></div>{bars}{mini}</div>'
        )
    rows = "".join(
        f'<div class="mc-line"><span>{esc(lbl)}</span><b>{esc(fmt_pct(g(m), 1))}</b></div>'
        for lbl, m in (
            ("Akurasi", "accuracy"),
            ("Macro-F1", "macro_f1"),
            ("Macro precision", "macro_precision"),
            ("Macro recall", "macro_recall"),
        )
    )
    return _h(
        f'<div class="card mc"><div class="mc-head"><div class="mc-name"><span class="dot" style="background:{color}"></span>{esc(name)}</div></div>{rows}</div>'
    )


# Tabel F1 per label untuk semua model.
def _f1_matrix_html(task: str, labels_display: dict[str, str] | None = None) -> str:
    table = per_label_f1_table(task)
    if table.empty:
        return f'<div class="muted" style="font-size:14px">F1 per label {NA_TEXT.lower()}.</div>'
    present = [k for k in MODEL_ORDER if MODEL_NAMES[k] in table.columns]
    headers = ["Label"] + [MODEL_NAMES[k] for k in present]
    rows = []
    for label in table.index:
        shown = (labels_display or {}).get(label, label)
        code_line = (
            f'<br><span class="mono" style="font-size:13px;color:#64748b">{esc(label)}</span>' if shown != label else ""
        )
        cells = [raw(f"<b>{esc(shown)}</b>{code_line}")]
        for k in present:
            v = table.loc[label, MODEL_NAMES[k]]
            if pd.isna(v):
                cells.append(raw('<span style="color:#94a3b8">—</span>'))
            else:
                cells.append(
                    raw(
                        f'<div style="text-align:right">{esc(fmt_pct(float(v), 1))}</div><div class="track" style="height:5px;margin-top:4px">'
                        f'<div class="fill" style="width:{float(v) * 100:.0f}%;background:{MODEL_COLORS[k]}"></div></div>'
                    )
                )
        rows.append(cells)
    return table_html(headers, rows, set(range(1, len(headers))))


# Cari model dengan nilai metrik terbaik.
def _best(summaries: dict[str, Any], task: str, metric: str, lower_better: bool = False) -> tuple[str, float] | None:
    vals = [(k, metric_value(summaries[k], task, metric)) for k in MODEL_ORDER]
    vals = [(k, v) for k, v in vals if v is not None]
    if not vals:
        return None
    return (min if lower_better else max)(vals, key=lambda kv: kv[1])


# =====================================================================
# Halaman 2: Evaluasi & Benchmark
# Ringkasan pemenang per tugas di atas; detail teknis tersembunyi secara default.
# Ranking dan pemenang dihitung dari Macro-F1 aktual di file hasil (tidak di-hardcode).
# =====================================================================
_BM_CSS = _h("""<style>
.bm-h1{font-size:30px;font-weight:700;color:var(--navy);margin:4px 0 4px;line-height:1.25;}
.bm-h2{font-size:22px;font-weight:700;color:var(--navy);margin:28px 0 12px;}
.bm-win{background:#fff;border:1px solid var(--border);border-left:4px solid var(--crimson);border-radius:12px;padding:20px 24px;}
.bm-win-label{font-size:13px;font-weight:600;color:var(--crimson);}
.bm-win-name{font-size:22px;font-weight:700;color:var(--navy);margin-top:4px;}
.bm-win-score{font-family:var(--font-mono);font-size:40px;font-weight:700;color:var(--navy);margin-top:6px;line-height:1.15;font-variant-numeric:tabular-nums;}
.bm-win-note{font-size:14px;color:var(--text-2);margin-top:8px;}
.bm-small{background:#fff;border:1px solid var(--border);border-radius:12px;padding:14px 16px;}
.bm-small-name{font-size:15px;font-weight:600;color:var(--navy);}
.bm-small-score{font-family:var(--font-mono);font-size:22px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums;}
.bm-muted{font-size:13px;color:var(--text-2);}
</style>""")

_BM_DETAIL_OPTIONS = [
    "Perbandingan metrik lengkap",
    "F1 per kategori isu",
    "Error multi-label & co-occurrence",
    "F1 per kelas severity",
    "Confusion matrix severity",
    "Waktu proses & ukuran model",
    "Classification report & prediksi uji",
    "Kesiapan artefak & uji reproduksi SVM",
]


# Saklar buka/tutup bagian; ada cadangan untuk Streamlit lama.
def _bm_toggle(label: str, key: str) -> bool:
    try:
        return st.toggle(label, value=False, key=key)
    except AttributeError:  # Streamlit lama tanpa st.toggle
        return st.checkbox(label, value=False, key=key)


def _bm_ranked(summaries: dict[str, Any], task: str) -> tuple[list[tuple[str, float]], list[str]]:
    """Urutkan model menurut Macro-F1 aktual. Model tanpa nilai dipisahkan (tidak diberi peringkat)."""
    scored = [(k, metric_value(summaries[k], task, "macro_f1")) for k in MODEL_ORDER]
    ranked = sorted([(k, v) for k, v in scored if v is not None], key=lambda kv: kv[1], reverse=True)
    unavailable = [k for k, v in scored if v is None]
    return ranked, unavailable


def _bm_top_keys(ranked: list[tuple[str, float]]) -> set[str]:
    """Model dengan skor tertinggi. Lebih dari satu berarti seri."""
    if not ranked:
        return set()
    top = ranked[0][1]
    return {k for k, v in ranked if abs(v - top) < 1e-12}


# Kartu pemenang (nilai tertinggi) benchmark.
def _bm_winner_card(ranked: list[tuple[str, float]]) -> str:
    if not ranked:
        return _h(
            '<div class="bm-win"><div class="bm-win-label">Belum tersedia</div>'
            '<div class="bm-win-note">Skor Macro-F1 untuk tugas ini tidak ada di file hasil.</div></div>'
        )
    top_keys = _bm_top_keys(ranked)
    top = ranked[0][1]
    if len(top_keys) > 1:
        label = "Seri"
        names = " & ".join(MODEL_NAMES[k] for k in MODEL_ORDER if k in top_keys)
        note = "Skor Macro-F1 tertinggi sama. Tidak ada pemenang tunggal."
    else:
        label = "Terbaik"
        names = MODEL_NAMES[ranked[0][0]]
        if len(ranked) > 1:
            k2, v2 = ranked[1]
            note = f"Unggul {(top - v2) * 100:.2f} poin persentase dari {MODEL_NAMES[k2]} ({fmt_pct(v2, 2)})."
        else:
            note = "Hanya satu model yang memiliki skor untuk tugas ini."
    return _h(f"""<div class="bm-win"><div class="bm-win-label">{esc(label)}</div>
        <div class="bm-win-name">{esc(names)}</div>
        <div class="bm-win-score">{esc(fmt_pct(top, 2))}</div>
        <div class="bm-win-note">Macro-F1 • {esc(note)}</div></div>""")


# Daftar model lain di bawah pemenang.
def _bm_others(ranked: list[tuple[str, float]], unavailable: list[str], top_keys: set[str]) -> str:
    items = []
    for k, v in ranked:
        if k in top_keys:
            continue
        items.append(
            _h(
                f'<div class="bm-small"><div class="bm-small-name">{esc(MODEL_NAMES[k])}</div>'
                f'<div class="bm-small-score">{esc(fmt_pct(v, 2))}</div></div>'
            )
        )
    for k in unavailable:
        items.append(
            _h(
                f'<div class="bm-small"><div class="bm-small-name">{esc(MODEL_NAMES[k])}</div>'
                f'<div class="bm-small-score bm-muted">{esc(NA_TEXT)}</div></div>'
            )
        )
    if not items:
        return ""
    return '<div class="g3">' + "".join(items) + "</div>"


# Satu bagian benchmark untuk satu tugas.
def _bm_task_section(title: str, task: str, summaries: dict[str, Any]) -> None:
    ranked, unavailable = _bm_ranked(summaries, task)
    md(_h(f'<div class="bm-h2">{esc(title)}</div>'))
    md(_bm_winner_card(ranked))
    others = _bm_others(ranked, unavailable, _bm_top_keys(ranked))
    if others:
        md(_h('<div class="bm-muted" style="margin:14px 0 8px">Model lainnya • Macro-F1</div>') + others)


# Tabel perbandingan metrik antar model.
def _bm_metric_table(summaries: dict[str, Any], task: str, metrics: list[tuple[str, str]]) -> str:
    headers = ["Metrik"] + [MODEL_NAMES[k] for k in MODEL_ORDER]
    rows = []
    for key, name in metrics:
        cells = [name]
        for k in MODEL_ORDER:
            v = metric_value(summaries[k], task, key)
            cells.append(fmt_float(v, 4) if key == "hamming_loss" else fmt_pct(v, 2))
        rows.append(cells)
    return table_html(headers, rows, set(range(1, len(headers))))


# Bagian analisis error multi-label.
def _bm_error_group(summaries: dict[str, Any]) -> None:
    hl_rows = []
    for k in MODEL_ORDER:
        hl = metric_value(summaries[k], "issue", "hamming_loss")
        ex = metric_value(summaries[k], "issue", "subset_accuracy")
        wo = macro_f1_without_other(k)
        hl_rows.append(
            [
                raw(
                    f'<span class="dot" style="background:{MODEL_COLORS[k]};margin-right:6px"></span>{esc(MODEL_NAMES[k])}'
                ),
                fmt_float(hl, 4) if hl is not None else NA_TEXT,
                fmt_pct(ex, 1),
                fmt_pct(wo, 1),
            ]
        )
    ov = dataset_overview(str(DATASET_PATH), _mtime(DATASET_PATH)) if DATASET_PATH.is_file() else {}
    pairs_html = "".join(
        f'<div class="mc-line"><span>{esc(ISSUE_DISPLAY.get(a, a))} + {esc(ISSUE_DISPLAY.get(b, b))}</span><b>{fmt_int(n)} review</b></div>'
        for (a, b), n in ov.get("top_pairs", [])
    )
    body = table_html(["Model", "Hamming loss", "Exact match", "Macro-F1 tanpa OTHER†"], hl_rows, {1, 2, 3})
    if pairs_html:
        body += (
            '<div class="rc-label" style="margin:16px 0 8px">Pasangan kategori yang sering muncul bersama</div>'
            + pairs_html
        )
    body += (
        '<div class="card-sub" style="margin-top:12px">† Rata-rata F1 sembilan label non-OTHER dari classification report. '
        "Pasangan kategori dihitung dari korpus, bukan dari prediksi model.</div>"
    )
    md(card(card_head("Hamming loss & exact match") + body))


# Rincian tambahan benchmark.
def _bm_details(summaries: dict[str, Any], t_tr: int | None) -> None:
    test_txt = f"{fmt_int(t_tr)} review" if t_tr else NA_TEXT
    md(
        _h(
            f'<div class="bm-muted" style="margin-bottom:12px">Test set: {esc(test_txt)} • threshold 0,50 untuk Transformer • '
            f"One-vs-Rest untuk Linear SVM • angka dibaca dari folder results/</div>"
        )
    )
    view = st.selectbox("Bagian detail", _BM_DETAIL_OPTIONS, key="bm_detail_view")

    if view == _BM_DETAIL_OPTIONS[0]:
        md(card(card_head("Metrik klasifikasi isu") + _bm_metric_table(summaries, "issue", ISSUE_METRICS)))
        md(card(card_head("Metrik severity") + _bm_metric_table(summaries, "severity", SEVERITY_METRICS)))

    elif view == _BM_DETAIL_OPTIONS[1]:
        md(card(card_head("F1-score per kategori isu") + _f1_matrix_html("issue", ISSUE_DISPLAY)))

    elif view == _BM_DETAIL_OPTIONS[2]:
        _bm_error_group(summaries)

    elif view == _BM_DETAIL_OPTIONS[3]:
        md(card(card_head("F1-score per kelas severity") + _f1_matrix_html("severity")))

    elif view == _BM_DETAIL_OPTIONS[4]:
        with keyed_container("card-cm"):
            md(card_head("Confusion matrix severity", sub="Baris = aktual, kolom = prediksi"))
            with keyed_container("seg-cm"):
                pick = st.radio(
                    "Model",
                    MODEL_ORDER,
                    format_func=lambda k: MODEL_NAMES[k],
                    horizontal=True,
                    key="cm_model",
                    label_visibility="collapsed",
                )
            path, cm, cm_note = load_confusion_matrix(pick)
            if cm is None:
                st.info(f"Confusion matrix {MODEL_NAMES[pick]} {NA_TEXT.lower()}.")
            else:
                md(heat_table(cm))
                total = int(cm.values.sum())
                correct = int(np.trace(cm.values))
                st.caption(
                    f"Sumber: {path.name if path else '-'} • benar {fmt_int(correct)} dari {fmt_int(total)} "
                    f"({correct / max(total, 1) * 100:.1f}%)" + (f" • {cm_note}" if cm_note else "")
                )

    elif view == _BM_DETAIL_OPTIONS[5]:
        lat = st.session_state.get("latency_log", {})
        vals = {k: lat[k] for k in MODEL_ORDER if lat.get(k) is not None}
        if vals:
            vmax = max(vals.values())
            inner = "".join(bar(MODEL_NAMES[k], fmt_duration(v), v / vmax, MODEL_COLORS[k]) for k, v in vals.items())
            inner += '<div class="card-sub">Waktu proses terakhir per model, termasuk pemuatan model bila mode hemat memori aktif.</div>'
        else:
            inner = (
                '<div class="note">' + icon("clock", 16) + "<div>Belum ada pengukuran. Jalankan prediksi di halaman "
                "<b>Analisis Komparasi Model</b>; waktunya akan muncul di sini.</div></div>"
            )
        lat_card = card(card_head("Waktu proses prediksi (sesi ini)", "clock") + inner)
        sizes = {k: weights_footprint(k) for k in MODEL_ORDER}
        known = {k: v for k, v in sizes.items() if v}
        vmax_size = max(known.values()) if known else 1
        inner = "".join(
            bar(
                MODEL_NAMES[k],
                fmt_size(sizes[k]) if sizes[k] else "Belum terhubung",
                (sizes[k] / vmax_size) if sizes[k] else 0,
                MODEL_COLORS[k] if sizes[k] else "#94a3b8",
            )
            for k in MODEL_ORDER
        )
        fp_card = card(
            card_head("Ukuran bobot di disk") + inner + '<div class="card-sub">Total issue + severity per model.</div>'
        )
        md(f'<div class="g2">{lat_card}{fp_card}</div>')

    elif view == _BM_DETAIL_OPTIONS[6]:
        with keyed_container("card-detail"):
            s1, s2 = columns([2, 1])
            model_key = s1.selectbox("Model", MODEL_ORDER, format_func=lambda k: MODEL_NAMES[k], key="eval_model")
            task = s2.selectbox(
                "Tugas",
                ["issue", "severity"],
                format_func=lambda t: "Klasifikasi isu" if t == "issue" else "Prediksi severity",
                key="eval_task",
            )
            t_rep, t_pred, t_err, t_met = st.tabs(
                ["Classification report", "Prediksi uji", "Error analysis", "Metrik (summary)"]
            )
            with t_rep:
                path, report, rnote = load_classification_report(model_key, task)
                if report is None:
                    st.info(
                        f"Classification report {task} {MODEL_NAMES[model_key]} {NA_TEXT.lower()}."
                        + (f" {rnote}" if rnote else "")
                    )
                else:
                    st.caption(f"Sumber: {path.name}")
                    if rnote:
                        st.caption(rnote)
                    show_df(
                        report.style.format(
                            {c: "{:.4f}" for c in ("precision", "recall", "f1-score")} | {"support": "{:.0f}"}
                        ),
                        hide_index=True,
                    )
            with t_pred:
                path, preds = load_predictions(model_key, task, "test_predictions")
                if preds is None:
                    st.info(f"Prediksi uji {task} {NA_TEXT.lower()}.")
                else:
                    st.caption(
                        f"Sumber: {path.name} • {fmt_int(len(preds))} baris • benar {fmt_int(int(preds['benar'].sum()))} ({preds['benar'].mean() * 100:.2f}%)"
                    )
                    f1, f2 = columns([1, 3])
                    only_wrong = f1.checkbox("Hanya yang salah", key=f"wrong_{model_key}_{task}")
                    query = f2.text_input(
                        "Cari kata dalam teks", key=f"q_{model_key}_{task}", placeholder="mis. voucher, login"
                    )
                    view_df = preds[~preds["benar"]] if only_wrong else preds
                    if query.strip():
                        view_df = view_df[
                            view_df["teks"].astype(str).str.contains(query.strip(), case=False, regex=False)
                        ]
                    show_df(view_df.head(1000), hide_index=True)
                    if len(view_df) > 1000:
                        st.caption(f"Menampilkan 1.000 dari {fmt_int(len(view_df))} baris.")
            with t_err:
                path, errors = load_predictions(model_key, task, "error_analysis")
                if errors is None:
                    st.info(f"Error analysis {task} {NA_TEXT.lower()}.")
                else:
                    st.caption(f"Sumber: {path.name} • {fmt_int(len(errors))} kesalahan {task}")
                    show_df(errors.head(1000), hide_index=True)
            with t_met:
                metrics = extract_test_metrics(summaries[model_key], task)
                if not metrics:
                    st.info(f"Metrik test {task} {NA_TEXT.lower()}.")
                else:
                    show_df(
                        pd.DataFrame({"Metrik": list(metrics), "Nilai": list(metrics.values())}).style.format(
                            {"Nilai": "{:.4f}"}
                        ),
                        hide_index=True,
                    )

    elif view == _BM_DETAIL_OPTIONS[7]:
        render_artifacts_section()


# Halaman Benchmark: perbandingan hasil uji semua model.
def page_benchmark() -> None:
    summaries = _summaries()
    tests = {k: split_counts(summaries[k])["review"].get("test") for k in MODEL_ORDER}
    t_tr = next((tests[k] for k in TRANSFORMER_MODELS if tests[k]), None)
    top_bar(
        "Evaluasi", "Benchmark", pill(f"Test set: {fmt_int(t_tr)} review" if t_tr else "Test set: " + NA_TEXT, "slate")
    )

    md(_BM_CSS)
    md(_h('<div class="bm-h1">Evaluasi &amp; Benchmark Model</div>'))
    _bm_task_section("Klasifikasi Isu", "issue", summaries)
    _bm_task_section("Prediksi Severity", "severity", summaries)

    if _bm_toggle("Tampilkan detail evaluasi", "bm_show_details"):
        _bm_details(summaries, t_tr)


# Tabel kesiapan file model dan hasil.
def render_artifacts_section() -> None:
    rows = []
    for key in MODEL_ORDER:
        for task, info in model_status(key)["tasks"].items():
            rows.append(
                {
                    "Pendekatan": MODEL_NAMES[key],
                    "Tugas": "Klasifikasi isu" if task == "issue" else "Prediksi severity",
                    "Status": "Terhubung (file lengkap)" if info["ready"] else "Belum terhubung",
                    "Lokasi dibaca": info["location"],
                    "Yang belum ada / masalah": "; ".join(info["missing"] + info["problems"]) or "—",
                }
            )
    show_df(pd.DataFrame(rows), hide_index=True)
    st.caption(
        "“Terhubung” hanya berarti file lengkap. Bisa tidaknya model dimuat dan memprediksi baru terbukti setelah Anda menjalankan prediksi dengan bobot asli."
    )
    path_note_shown = False
    for key in TRANSFORMER_MODELS:
        for task, info in model_status(key)["tasks"].items():
            if info["labels"]:
                st.caption(f"{MODEL_NAMES[key]} • {task}: urutan label dari config.json = {', '.join(info['labels'])}")
            for w in info["warnings"]:
                if w.startswith("Path di MODEL_PATHS menunjuk ke file"):
                    if not path_note_shown:  # satu catatan ringkas, bukan satu per tugas
                        st.info(
                            "MODEL_PATHS menunjuk ke berkas bobot (mis. .../model.safetensors); dashboard membaca folder induknya. "
                            "Ini valid, asalkan config.json, tokenizer, dan bobot ada di folder yang sama."
                        )
                        path_note_shown = True
                else:
                    st.warning(w)

    st.markdown("**Uji reproduksi adapter Linear SVM**")
    st.write(
        "Memprediksi ulang seluruh baris uji dengan artefak .joblib lalu membandingkannya dengan svm_tfidf_test_predictions.csv."
    )
    if model_status("svm_tfidf")["ready"]:
        if st.button("Jalankan uji reproduksi SVM", key="btn_reprod"):
            try:
                paths = MODEL_PATHS["svm_tfidf"]
                ref = result_file("svm_tfidf", "issue", "test_predictions")
                sig = "|".join([f"{p}:{_mtime(p)}" for p in paths.values()] + [f"{ref}:{_mtime(ref)}"])
                with st.spinner("Memuat artefak SVM dan memprediksi ulang data uji…"):
                    bundle = get_svm_bundle()
                    out = verify_svm_reproduction(sig)
                if bundle["idf_shim_applied"]:
                    st.info(
                        f"scikit-learn terpasang {bundle['sklearn_version']}; artefak dilatih dengan {bundle['trained_with'] or 'versi lain'}. Shim IDF diterapkan."
                    )
                if out["status"] != "ok":
                    st.warning(out["message"])
                elif out["all_match"]:
                    st.success(
                        f"Cocok: {fmt_int(out['issue_match'])}/{fmt_int(out['n'])} prediksi issue dan {fmt_int(out['severity_match'])}/{fmt_int(out['n'])} prediksi severity sama dengan {out['source']}."
                    )
                else:
                    st.error(
                        f"Tidak cocok: issue {fmt_int(out['issue_match'])}/{fmt_int(out['n'])}, severity {fmt_int(out['severity_match'])}/{fmt_int(out['n'])}. "
                        f"Pasang scikit-learn {bundle['trained_with'] or '1.4.2'} lalu ulangi."
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Uji reproduksi gagal: {type(exc).__name__}: {exc}")
    else:
        st.info("Belum bisa dijalankan: artefak SVM belum lengkap.")
    st.markdown("**Lingkungan**")
    show_df(pd.DataFrame(package_versions().items(), columns=["Paket", "Versi"]), hide_index=True)
    if st.button("Periksa GPU (memuat PyTorch)", key="btn_gpu"):
        st.write(gpu_info())


# =====================================================================
# Halaman 3: Eksplorasi Dataset Ulasan
# =====================================================================
def page_dataset() -> None:
    if not DATASET_PATH.is_file():
        top_bar("Dataset", "Eksplorasi korpus", pill(NA_TEXT, "amber"))
        st.warning(
            f"Berkas {DATASET_PATH.name} tidak ditemukan di {DATASET_PATH.parent}. Letakkan CSV korpus di folder dashboard atau ubah DATASET_PATH."
        )
        return
    ov = dataset_overview(str(DATASET_PATH), _mtime(DATASET_PATH))
    n_rows, n_rev = ov.get("n_rows", 0), ov.get("n_reviews", 0)
    top_bar("Dataset", "Eksplorasi korpus", pill(f"{fmt_int(n_rows)} baris anotasi", "green", dot="#10b981"))

    md(_h(f"""<div class="card"><div class="banner">
        <h1>Eksplorasi Dataset Ulasan Alfagift</h1>
        <p>{fmt_int(n_rows)} baris anotasi (satu baris per pasangan ulasan–kategori) dari {fmt_int(n_rev)} ulasan unik.
        Kolom nama pengguna tidak dimuat.</p></div></div>"""))

    tok = f'{ov["tok_mean"]:.1f} kata' if "tok_mean" in ov else NA_TEXT
    period = f'{ov["year_min"]} – {ov["year_max"]}' if "year_min" in ov else NA_TEXT
    md(
        '<div class="g4">'
        + "".join(
            [
                kpi_card("Total ulasan unik", fmt_int(n_rev), f"{fmt_int(n_rows)} baris anotasi", icon_name="list"),
                kpi_card("Rentang tahun", period, f'Data terakhir: {esc(ov.get("date_max", "-"))}', icon_name="clock"),
                kpi_card(
                    "Rata-rata panjang",
                    tok,
                    f'Min {ov.get("tok_min", "-")} • maks {ov.get("tok_max", "-")} kata',
                    icon_name="layers",
                ),
                kpi_card(
                    "Multi-label",
                    fmt_pct(ov.get("multi_pct"), 1),
                    f'{fmt_int(ov.get("multi_n", 0))} ulasan memiliki lebih dari satu kategori',
                    icon_name="database",
                ),
            ]
        )
        + "</div>"
    )

    cat = ov.get("cat_counts", {})
    sev = ov.get("sev_counts", {})
    rat = ov.get("rating_counts", {})
    cat_total = sum(cat.values()) or 1
    cat_html = "".join(
        bar(
            f"{ISSUE_DISPLAY.get(k, k)}",
            f"{fmt_int(v)}  {v / cat_total * 100:.1f}%",
            v / (max(cat.values()) or 1),
            "#aa0015" if i == 0 else "#475569" if i == 1 else "#94a3b8",
        )
        for i, (k, v) in enumerate(cat.items())
    )
    sev_total = sum(sev.values()) or 1
    seg = [
        (s, sev.get(s, 0), SEVERITY_COLORS[s]["dot"] if s != "BLOCKER" else "#aa0015")
        for s in ("BLOCKER", "HIGH", "MEDIUM", "LOW")
    ]
    tiles = "".join(
        f'<div class="tile" style="background:{SEVERITY_COLORS[s]["bg"] if s != "BLOCKER" else "#fef2f2"};border-color:{SEVERITY_COLORS[s]["border"] if s != "BLOCKER" else "#fecaca"};'
        f'color:{SEVERITY_COLORS[s]["text"] if s != "BLOCKER" else "#aa0015"}"><div class="t"><span class="dot" style="background:{c}"></span>{s.title()}</div>'
        f"<b>{fmt_int(v)}</b><small>{v / sev_total * 100:.1f}%</small></div>"
        for s, v, c in seg
    )
    rat_total = sum(rat.values()) or 1
    star_colors = {1: "#aa0015", 2: "#f59e0b", 3: "#94a3b8", 4: "#94a3b8", 5: "#10b981"}
    rat_html = "".join(
        bar(
            f"{s} bintang",
            f"{fmt_int(rat.get(s, 0))}  {rat.get(s, 0) / rat_total * 100:.1f}%",
            rat.get(s, 0) / (max(rat.values()) or 1),
            star_colors[s],
        )
        for s in (1, 2, 3, 4, 5)
    )
    neg = (rat.get(1, 0) + rat.get(2, 0)) / rat_total
    md(
        '<div class="g3">'
        + card(card_head("Kategori isu", "chart", sub="Dihitung per baris anotasi") + cat_html)
        + card(
            card_head("Tingkat keparahan", "alert", sub="Dihitung per baris anotasi")
            + stacked_bar_html(seg)
            + f'<div class="tiles">{tiles}</div>'
            + (
                f'<div class="card-sub" style="margin-top:10px">{fmt_int(n_rows - sev_total)} baris tanpa label severity tidak dihitung.</div>'
                if n_rows > sev_total
                else ""
            )
        )
        + card(
            card_head(
                "Rating bintang", "star", sub=f'Rata-rata {ov.get("rating_avg", 0):.2f} • dihitung per ulasan unik'
            )
            + rat_html
            + f'<div class="note" style="margin-top:6px">Ulasan berbintang 1–2: <b>{neg * 100:.1f}%</b> dari seluruh ulasan.</div>'
        )
        + "</div>"
    )

    with keyed_container("card-trend"):
        t1, t2 = columns([3, 1], vertical_alignment="bottom")
        t1.markdown(
            card_head(
                "Tren ulasan per tahun", "trend", sub="Jumlah baris anotasi pada kategori terpilih per tahun ulasan"
            ),
            unsafe_allow_html=True,
        )
        cats = list(cat.keys()) or ["SYSTEM"]
        pick = t2.selectbox("Kategori", cats, format_func=lambda k: ISSUE_DISPLAY.get(k, k), key="trend_cat")
        trend = category_year_trend(str(DATASET_PATH), _mtime(DATASET_PATH), pick)
        md(trend_svg(trend))
        if trend:
            st.caption(
                f"Total {fmt_int(sum(trend.values()))} baris untuk {ISSUE_DISPLAY.get(pick, pick)}. Data tahun {max(trend)} berjalan sampai {ov.get('date_max', '-')}."
            )

    with keyed_container("card-filter"):
        df = get_dataset()
        md(
            card_head(
                "Korpus ulasan", "database", sub="Cari ulasan berdasarkan kata kunci, kategori, severity, atau tahun"
            )
        )
        f1, f2, f3, f4 = columns([2.2, 1, 1, 1], vertical_alignment="bottom")
        q = f1.text_input("Cari ulasan", placeholder="Ketik kata, mis. voucher, lemot, login…", key="ds_q")
        f_cat = f2.selectbox("Kategori", ["Semua"] + sorted(df["category_Issue"].dropna().unique()), key="ds_cat")
        f_sev = f3.selectbox("Severity", ["Semua"] + SEVERITY_LABELS, key="ds_sev")
        years = sorted(df["review_year"].dropna().astype(int).unique()) if "review_year" in df.columns else []
        f_year = f4.selectbox("Tahun", ["Semua"] + years, key="ds_year")
        view = df
        if q.strip():
            view = view[view["review"].fillna("").astype(str).str.contains(q.strip(), case=False, regex=False)]
        if f_cat != "Semua":
            view = view[view["category_Issue"] == f_cat]
        if f_sev != "Semua":
            view = view[view["severity_Issue"] == f_sev]
        if f_year != "Semua":
            view = view[view["review_year"] == f_year]
        shown = [
            c
            for c in ["review_id", "review", "text_norm", "category_Issue", "severity_Issue", "rating", "review_year"]
            if c in view.columns
        ]
        st.caption(f"Menampilkan {fmt_int(min(len(view), 500))} dari {fmt_int(len(view))} baris yang cocok.")
        show_df(view[shown].head(500), hide_index=True)
        st.download_button(
            "Unduh hasil filter (.csv)",
            data=view[shown].to_csv(index=False).encode("utf-8"),
            file_name="alfagift_korpus_terfilter.csv",
            mime="text/csv",
        )


# =====================================================================
def main() -> None:
    st.set_page_config(
        page_title="Alfagift Review Analytics", page_icon="🛒", layout="wide", initial_sidebar_state="expanded"
    )
    inject_styles()
    page, settings = render_sidebar()
    if page == NAV_PAGES[0]:
        page_comparison(settings)
    elif page == NAV_PAGES[1]:
        page_benchmark()
    else:
        page_dataset()
    md(
        '<div style="text-align:center;font-size:13px;color:#64748b;padding:22px 0 6px">Hasil prediksi adalah bantuan analitis, '
        "bukan penetapan severity otomatis tanpa validasi manusia.</div>"
    )


if __name__ == "__main__":
    main()
