"""Uji statistik perbandingan 4 model (SVM-TF-IDF, IndoBERT, mBERT, XLM-RoBERTa).

Cara pakai (dari folder dashboard, yang berisi folder results/):
    pip install pandas numpy scipy
    python uji_statistik.py
    python uji_statistik.py --root "D:/folder/dashboard" --boot 5000

Yang dihitung:
  1. Issue    : bootstrap per review -> interval kepercayaan 95% Macro-F1 tiap model
                dan selisih Macro-F1 antar model (semua pasangan).
  2. Severity : bootstrap per review (pasangan review-kategori milik review yang sama
                diambil bersama) + uji McNemar eksak untuk tiap pasangan model.
Semua model dibandingkan HANYA pada review/pasangan yang ada di semua file prediksi.

Hasil: tabel di layar dan file CSV di folder uji_statistik_output/.
"""
import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

# nama tampilan -> nama folder di results/
MODELS = {"SVM-TF-IDF": "svm_tfidf", "IndoBERT": "indobert", "mBERT": "mbert", "XLM-RoBERTa": "xlmr"}
SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "BLOCKER"]

# kandidat nama kolom (sama dengan yang dibaca dashboard)
COL_REVIEW_ID = ["review_id"]
COL_ISSUE_TRUE = ["true_issue_labels", "true_issues", "issue_true"]
COL_ISSUE_PRED = ["predicted_issue_labels", "predicted_issues", "issue_pred"]
COL_CATEGORY = ["category_Issue"]
COL_SEV_TRUE = ["true_severity", "severity_Issue", "severity_true"]
COL_SEV_PRED = ["predicted_severity", "severity_pred"]


# ---------------------------------------------------------------- baca file
def find_file(folder: Path, task: str):
    """Cari file prediksi uji (aturan sama dengan dashboard)."""
    per_task = sorted(folder.glob(f"*{task}*test_predictions*.csv"))
    if per_task:
        return per_task[0]
    general = [
        p
        for p in sorted(folder.glob("*test_predictions*.csv"))
        if "issue" not in p.name.lower() and "severity" not in p.name.lower()
    ]
    return general[0] if general else None


def pick(df: pd.DataFrame, candidates, what: str, path: Path) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(f"[ERROR] Kolom {what} tidak ada di {path.name}. Kolom tersedia: {list(df.columns)}")


def parse_labels(value) -> set:
    """'A | B' -> {'A','B'}; kosong atau '(tidak ada ...)' -> set()."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).strip()
    if not text or text.startswith("("):
        return set()
    # pemisah label berbeda antar file: " | " (SVM, IndoBERT) atau ", " (mBERT, XLM-R)
    return {p.strip().upper() for p in re.split(r"[|,;]", text) if p.strip()}


def load_issue(path: Path) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    rid, t, p = (pick(df, c, w, path) for c, w in
                 [(COL_REVIEW_ID, "review_id"), (COL_ISSUE_TRUE, "label issue sebenarnya"),
                  (COL_ISSUE_PRED, "label issue prediksi")])
    return {str(r): (parse_labels(a), parse_labels(b)) for r, a, b in zip(df[rid], df[t], df[p])}


def load_severity(path: Path) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    rid, cat, t, p = (pick(df, c, w, path) for c, w in
                      [(COL_REVIEW_ID, "review_id"), (COL_CATEGORY, "category_Issue"),
                       (COL_SEV_TRUE, "severity sebenarnya"), (COL_SEV_PRED, "severity prediksi")])
    return {(str(r), str(c).upper()): (str(a).upper(), str(b).upper())
            for r, c, a, b in zip(df[rid], df[cat], df[t], df[p])}


def check_overlap(data: dict, common: list, what: str, force: bool):
    """Hentikan jika test set antar model tidak (hampir) sama, supaya tidak menghasilkan angka menyesatkan."""
    smallest = min(len(d) for d in data.values())
    share = len(common) / smallest
    if share < 0.98 and not force:
        sys.exit(
            f"\n[BERHENTI] Hanya {len(common)} dari {smallest} {what} ({share:.0%}) yang sama di semua model.\n"
            "Test set model-model ini TIDAK sama, jadi perbandingannya tidak sah.\n"
            "Jalankan cek_data.py untuk melihat penyebabnya. (Paksa lanjut dengan --force, hanya untuk uji coba.)"
        )


# ---------------------------------------------------------------- bootstrap
def bootstrap_weights(n: int, boot: int, seed: int) -> np.ndarray:
    """Matriks (boot x n): berapa kali tiap review terambil pada tiap ulangan."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(boot, n))
    return np.stack([np.bincount(row, minlength=n) for row in idx]).astype(np.float64)


def f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)


def ci(x: np.ndarray):
    return np.percentile(x, [2.5, 97.5])


def summarize_pairwise(boot_scores: dict, point: dict, name: str):
    rows = []
    for a, b in itertools.combinations(boot_scores, 2):
        d = boot_scores[a] - boot_scores[b]
        lo, hi = ci(d)
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        rows.append({"model_A": a, "model_B": b, f"{name}_A": point[a], f"{name}_B": point[b],
                     "selisih_A_minus_B": point[a] - point[b], "CI95_bawah": lo, "CI95_atas": hi,
                     "p_bootstrap": p, "CI_mencakup_0": bool(lo <= 0 <= hi)})
    out = pd.DataFrame(rows)
    # koreksi Holm untuk banyak perbandingan
    order = out["p_bootstrap"].argsort().to_numpy()
    m, holm, running = len(out), np.zeros(len(out)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * out["p_bootstrap"].iloc[i]))
        holm[i] = running
    out["p_Holm"] = holm
    out["signifikan_0.05_Holm"] = out["p_Holm"] < 0.05
    return out


# ---------------------------------------------------------------- issue
def run_issue(data: dict, boot: int, seed: int, exclude_other: bool, force: bool = False):
    common = sorted(set.intersection(*[set(d) for d in data.values()]))
    if not common:
        sys.exit("[ERROR] Tidak ada review_id yang sama di semua file issue.")
    check_overlap(data, common, "review", force)
    labels = sorted({l for d in data.values() for r in common for s in d[r] for l in s})
    if exclude_other:
        labels = [l for l in labels if l != "OTHER"]
    li = {l: i for i, l in enumerate(labels)}
    n, L = len(common), len(labels)
    print(f"\n=== ISSUE (multi-label) | {n} review sama di semua model | {L} label ===")
    print("Jumlah review per file:", {m: len(d) for m, d in data.items()})
    W = bootstrap_weights(n, boot, seed)
    point, boots = {}, {}
    for m, d in data.items():
        tp, fp, fn = (np.zeros((n, L)) for _ in range(3))
        for i, r in enumerate(common):
            true, pred = d[r]
            for l in labels:
                t, p = l in true, l in pred
                tp[i, li[l]], fp[i, li[l]], fn[i, li[l]] = t and p, (not t) and p, t and (not p)
        point[m] = f1_from_counts(tp.sum(0), fp.sum(0), fn.sum(0)).mean()
        boots[m] = f1_from_counts(W @ tp, W @ fp, W @ fn).mean(axis=1)
    ci_df = pd.DataFrame([{"model": m, "n_review": n, "macro_F1": point[m],
                           "CI95_bawah": ci(boots[m])[0], "CI95_atas": ci(boots[m])[1]} for m in data])
    pair_df = summarize_pairwise(boots, point, "macroF1")
    return ci_df, pair_df


# ---------------------------------------------------------------- severity
def run_severity(data: dict, boot: int, seed: int, force: bool = False):
    common = sorted(set.intersection(*[set(d) for d in data.values()]))
    if not common:
        sys.exit("[ERROR] Tidak ada pasangan (review, kategori) yang sama di semua file severity.")
    check_overlap(data, common, "pasangan review-kategori", force)
    print(f"\n=== SEVERITY (per pasangan review-kategori) | {len(common)} pasangan sama di semua model ===")
    print("Jumlah pasangan per file:", {m: len(d) for m, d in data.items()})
    reviews = sorted({r for r, _ in common})
    rmap = {r: i for i, r in enumerate(reviews)}
    n, K = len(reviews), len(SEVERITY_LABELS)
    ki = {l: i for i, l in enumerate(SEVERITY_LABELS)}
    W = bootstrap_weights(n, boot, seed)
    point, boots, correct = {}, {}, {}
    for m, d in data.items():
        conf = np.zeros((n, K, K))  # per review: baris = sebenarnya, kolom = prediksi
        ok = np.zeros(len(common), dtype=bool)
        for j, key in enumerate(common):
            t, p = d[key]
            ok[j] = t == p
            if t in ki and p in ki:
                conf[rmap[key[0]], ki[t], ki[p]] += 1
        correct[m] = ok

        def macro(c):  # c: (..., K, K)
            tp = np.diagonal(c, axis1=-2, axis2=-1)
            return f1_from_counts(tp, c.sum(-2) - tp, c.sum(-1) - tp).mean(-1)

        point[m] = macro(conf.sum(0))
        boots[m] = macro(np.einsum("bn,nij->bij", W, conf))
    ci_df = pd.DataFrame([{"model": m, "n_pasangan": len(common), "macro_F1": point[m],
                           "accuracy": correct[m].mean(),
                           "CI95_bawah": ci(boots[m])[0], "CI95_atas": ci(boots[m])[1]} for m in data])
    pair_df = summarize_pairwise(boots, point, "macroF1")
    # McNemar eksak
    mc = []
    for a, b in itertools.combinations(data, 2):
        only_a = int((correct[a] & ~correct[b]).sum())
        only_b = int((~correct[a] & correct[b]).sum())
        p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        mc.append({"model_A": a, "model_B": b, "hanya_A_benar": only_a, "hanya_B_benar": only_b,
                   "p_McNemar": p})
    mc_df = pd.DataFrame(mc)
    order = mc_df["p_McNemar"].argsort().to_numpy()
    holm, running = np.zeros(len(mc_df)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(mc_df) - rank) * mc_df["p_McNemar"].iloc[i]))
        holm[i] = running
    mc_df["p_Holm"] = holm
    mc_df["signifikan_0.05_Holm"] = mc_df["p_Holm"] < 0.05
    return ci_df, pair_df, mc_df


# ---------------------------------------------------------------- main
def show(title: str, df: pd.DataFrame):
    print(f"\n--- {title} ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="folder dashboard (berisi folder results/)")
    ap.add_argument("--boot", type=int, default=2000, help="jumlah ulangan bootstrap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="lanjut walau test set tidak sama (hanya uji coba)")
    ap.add_argument("--exclude-other", action="store_true", help="keluarkan label OTHER dari Macro-F1 issue")
    args = ap.parse_args()

    root = Path(args.root)
    issue, sev = {}, {}
    for name, folder in MODELS.items():
        d = root / "results" / folder
        fi, fs = find_file(d, "issue"), find_file(d, "severity")
        print(f"{name:12s} issue: {fi.name if fi else '-- TIDAK ADA --'} | severity: {fs.name if fs else '-- TIDAK ADA --'}")
        if fi is None or fs is None:
            sys.exit(f"[ERROR] File prediksi uji {name} tidak lengkap di {d}")
        issue[name], sev[name] = load_issue(fi), load_severity(fs)

    out = Path("uji_statistik_output")
    out.mkdir(exist_ok=True)

    ci_i, pair_i = run_issue(issue, args.boot, args.seed, args.exclude_other, args.force)
    show("Issue: Macro-F1 dan interval kepercayaan 95%", ci_i)
    show("Issue: selisih Macro-F1 antar model", pair_i)
    ci_i.to_csv(out / "issue_ci.csv", index=False)
    pair_i.to_csv(out / "issue_selisih_antar_model.csv", index=False)

    ci_s, pair_s, mc_s = run_severity(sev, args.boot, args.seed, args.force)
    show("Severity: Macro-F1 dan interval kepercayaan 95%", ci_s)
    show("Severity: selisih Macro-F1 antar model (bootstrap per review)", pair_s)
    show("Severity: uji McNemar eksak (pasangan dianggap independen)", mc_s)
    ci_s.to_csv(out / "severity_ci.csv", index=False)
    pair_s.to_csv(out / "severity_selisih_antar_model.csv", index=False)
    mc_s.to_csv(out / "severity_mcnemar.csv", index=False)

    print(f"\nSelesai. Semua tabel disimpan di folder: {out.resolve()}")
    print("Cara membaca: kalau 'CI_mencakup_0' = True atau 'signifikan_0.05_Holm' = False, "
          "kedua model tidak berbeda secara signifikan.")


if __name__ == "__main__":
    main()