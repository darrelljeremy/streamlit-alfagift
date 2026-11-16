"""Model severity per (review, kategori) berbasis LinearSVC + TF-IDF.

File ini dipakai bersama oleh notebook training DAN app.py agar joblib bisa
me-load model yang sama (class harus bisa di-import dengan nama modul yang sama).

mode:
  shared       : satu LinearSVC; fitur = TF-IDF teks + one-hot kategori
  augmented    : shared + salinan TF-IDF khusus tiap kategori (feature augmentation)
  per_category : satu LinearSVC terpisah per kategori; kategori yang datanya terlalu
                 sedikit / hanya 1 kelas memakai model shared sebagai fallback
"""
import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC

SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "BLOCKER"]


class CategorySeverityModel:
    def __init__(self, mode="augmented", C=1.0, class_weight=None, categories=None,
                 min_samples=30, seed=42):
        self.mode = mode
        self.C = C
        self.class_weight = class_weight
        self.categories = list(categories) if categories is not None else None
        self.min_samples = min_samples
        self.seed = seed

    # ---------- fitur ----------
    def _svc(self):
        return LinearSVC(C=self.C, class_weight=self.class_weight,
                         random_state=self.seed, max_iter=10000, dual="auto")

    def _onehot(self, cats):
        idx = {c: i for i, c in enumerate(self.categories)}
        rows, cols = [], []
        for r, c in enumerate(cats):
            if c in idx:
                rows.append(r)
                cols.append(idx[c])
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                             shape=(len(cats), len(self.categories)))

    def _features(self, X, cats):
        cats = np.asarray(cats)
        if self.mode == "shared":
            return sp.hstack([X, self._onehot(cats)], format="csr")
        if self.mode == "augmented":
            blocks = [X, self._onehot(cats)]
            for c in self.categories:
                blocks.append(sp.diags((cats == c).astype(np.float32)) @ X)
            return sp.hstack(blocks, format="csr")
        raise ValueError(self.mode)

    # ---------- training ----------
    def fit(self, X, cats, y):
        cats, y = np.asarray(cats), np.asarray(y)
        if self.categories is None:
            self.categories = sorted(set(cats))
        if self.mode in ("shared", "augmented"):
            self.model_ = self._svc().fit(self._features(X, cats), y)
            self.classes_ = self.model_.classes_
            return self
        # per_category
        self.global_ = self._svc().fit(self._features_shared(X, cats), y)
        self.classes_ = self.global_.classes_
        self.cat_models_, self.cat_info_ = {}, {}
        for c in self.categories:
            m = cats == c
            labels = np.unique(y[m])
            if m.sum() == 0:
                self.cat_info_[c] = "fallback_global (tidak ada data)"
            elif len(labels) == 1:
                self.cat_models_[c] = str(labels[0])
                self.cat_info_[c] = f"konstan={labels[0]} (hanya 1 kelas)"
            elif m.sum() < self.min_samples:
                self.cat_info_[c] = f"fallback_global (n={int(m.sum())} < {self.min_samples})"
            else:
                self.cat_models_[c] = self._svc().fit(X[m], y[m])
                self.cat_info_[c] = f"model sendiri (n={int(m.sum())})"
        return self

    def _features_shared(self, X, cats):
        return sp.hstack([X, self._onehot(cats)], format="csr")

    # ---------- prediksi ----------
    def predict(self, X, cats):
        return self.predict_with_margin(X, cats)[0]

    def predict_with_margin(self, X, cats):
        """Return (label, margin_kelas_terpilih). Margin = decision_function (bukan probabilitas)."""
        cats = np.asarray(cats)
        n = X.shape[0]
        labels = np.empty(n, dtype=object)
        margins = np.zeros(n, dtype=float)
        if self.mode in ("shared", "augmented"):
            dec = self._decision(self.model_, self._features(X, cats))
            best = dec.argmax(1)
            labels[:] = self.model_.classes_[best]
            margins[:] = dec[np.arange(n), best]
            return labels, margins
        dec_g = self._decision(self.global_, self._features_shared(X, cats))
        best = dec_g.argmax(1)
        labels[:] = self.global_.classes_[best]
        margins[:] = dec_g[np.arange(n), best]
        for c, m in self.cat_models_.items():
            mask = cats == c
            if not mask.any():
                continue
            if isinstance(m, str):
                labels[mask] = m
                margins[mask] = 0.0
            else:
                d = self._decision(m, X[mask])
                b = d.argmax(1)
                labels[mask] = m.classes_[b]
                margins[mask] = d[np.arange(d.shape[0]), b]
        return labels, margins

    @staticmethod
    def _decision(model, X):
        d = model.decision_function(X)
        if d.ndim == 1:  # biner: kolom ke-0 = kelas pertama, kolom ke-1 = kelas kedua
            d = np.column_stack([-d, d])
        return d
