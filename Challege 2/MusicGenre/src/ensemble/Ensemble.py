"""Kết hợp model phân loại trên XÁC SUẤT out-of-fold: Voting, Blending, Stacking.

Mọi phương pháp nhận dict {tên_model: ma trận xác suất (n, n_lớp)}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class Blending:
    """Trung bình xác suất có trọng số; trọng số tối ưu log loss trên OOF."""

    def __init__(self, weights: dict | None = None):
        self.weights = self._normalize(weights) if weights else None

    @staticmethod
    def _normalize(w: dict) -> dict:
        total = sum(w.values())
        if total <= 0:
            raise ValueError("Tổng trọng số phải lớn hơn 0")
        return {k: v / total for k, v in w.items()}

    def fit(self, oof: dict, y) -> "Blending":
        names = list(oof)
        S = np.stack([oof[n] for n in names])
        y = np.asarray(y)
        labels = np.sort(np.unique(y))

        def loss(w):
            w = np.clip(w, 0, None)
            return log_loss(y, np.tensordot(w / w.sum(), S, 1), labels=labels)

        res = minimize(loss, np.full(len(names), 1 / len(names)), method="SLSQP",
                       bounds=[(0.0, 1.0)] * len(names),
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        self.weights = self._normalize(dict(zip(names, np.clip(res.x, 0, None))))
        return self

    def predict_proba(self, probas: dict) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Chưa có trọng số: gọi fit() hoặc truyền weights")
        missing = set(self.weights) - set(probas)
        if missing:
            raise KeyError(f"Thiếu dự đoán của: {sorted(missing)}")
        return sum(self.weights[k] * np.asarray(probas[k]) for k in self.weights)


class Voting(Blending):
    """Soft voting: trung bình đều xác suất. Mốc so sánh cho Blending."""

    def __init__(self, names: list[str]):
        super().__init__({n: 1.0 for n in names})

    def fit(self, oof=None, y=None):
        return self


class Stacking:
    """Hồi quy logistic cấp hai trên xác suất OOF ghép của mọi model."""

    def __init__(self, C: float = 1.0):
        self.meta = LogisticRegression(C=C, max_iter=2000)
        self.names_: list[str] = []
        self.weights = None

    def _stack(self, probas: dict) -> np.ndarray:
        return np.hstack([probas[n] for n in self.names_])

    def fit(self, oof: dict, y) -> "Stacking":
        self.names_ = list(oof)
        self.meta.fit(self._stack(oof), np.asarray(y))
        return self

    def predict_proba(self, probas: dict) -> np.ndarray:
        return self.meta.predict_proba(self._stack(probas))

    def oof_proba(self, oof: dict, y, n_folds: int = 5, seed: int = 42) -> np.ndarray:
        """Xác suất của meta model, tự nó cũng out-of-fold.

        Stacking có hàng chục tham số nên đánh giá ngay trên dữ liệu đã fit
        sẽ lạc quan. Hàm này cross-validate meta model để con số trung thực.
        """
        self.names_ = list(oof)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return cross_val_predict(self.meta, self._stack(oof), np.asarray(y), cv=skf, method="predict_proba")


def build(method: str, names: list[str], cfg: dict):
    method = method.lower()
    if method == "voting":
        return Voting(names)
    if method == "blending":
        return Blending()
    if method == "stacking":
        return Stacking()
    raise ValueError(f"Chưa hỗ trợ phương pháp: {method}")


def compare_methods(oof: dict, y, classes, seed: int = 42) -> pd.DataFrame:
    """So sánh ba cách kết hợp và từng model đơn lẻ trên OOF."""
    y = np.asarray(y); classes = np.asarray(classes)

    def row(name, P, weights=None):
        return {"method": name,
                "accuracy": round(accuracy_score(y, classes[P.argmax(1)]), 4),
                "log_loss": round(log_loss(y, P, labels=classes), 4),
                "weights": {k: round(float(v), 3) for k, v in (weights or {}).items()}}

    v = Voting(list(oof)); b = Blending().fit(oof, y); s = Stacking()
    rows = [row("voting", v.predict_proba(oof), v.weights),
            row("blending", b.predict_proba(oof), b.weights),
            row("stacking", s.oof_proba(oof, y, seed=seed))]
    rows += [row(f"(đơn lẻ) {n}", P) for n, P in oof.items()]
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
