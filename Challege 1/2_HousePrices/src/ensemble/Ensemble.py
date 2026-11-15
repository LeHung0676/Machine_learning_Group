"""Kết hợp nhiều model: Voting, Blending, Stacking, Bagging.

Mọi thứ ở đây làm việc trên dự đoán OOF (out-of-fold) do Evaluator sinh ra.
Dùng OOF là bắt buộc: nếu tìm trọng số trên cùng tập đã huấn luyện thì
trọng số sẽ nghiêng về model quá khớp nhất, và điểm Kaggle sẽ tệ hơn hẳn
so với điểm bạn thấy ở máy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge

from src.evaluation.Evaluation import rmse


class Blending:
    """Trung bình có trọng số. Trọng số do người dùng đặt, hoặc tự tối ưu."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = self._normalize(weights) if weights else None

    @staticmethod
    def _normalize(w: dict) -> dict:
        total = sum(w.values())
        if total <= 0:
            raise ValueError("Tổng trọng số phải lớn hơn 0")
        return {k: v / total for k, v in w.items()}

    def fit(self, oof: dict[str, np.ndarray], y) -> "Blending":
        """Tìm trọng số tối ưu trên OOF, ràng buộc: không âm và tổng bằng 1."""
        names = list(oof.keys())
        P = np.column_stack([oof[n] for n in names])
        y = np.asarray(y)

        def loss(w):
            return rmse(y, P @ w)

        x0 = np.full(len(names), 1 / len(names))
        res = minimize(
            loss, x0, method="SLSQP",
            bounds=[(0.0, 1.0)] * len(names),
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        w = np.clip(res.x, 0, None)
        self.weights = self._normalize(dict(zip(names, w)))
        return self

    def predict(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Chưa có trọng số: gọi fit() hoặc truyền weights")
        missing = set(self.weights) - set(preds)
        if missing:
            raise KeyError(f"Thiếu dự đoán của: {sorted(missing)}")
        return sum(self.weights[k] * np.asarray(preds[k]) for k in self.weights)


class Voting(Blending):
    """Trung bình đều, mọi model cùng trọng số. Mốc so sánh cho Blending."""

    def __init__(self, names: list[str]):
        super().__init__({n: 1.0 for n in names})

    def fit(self, oof=None, y=None):
        return self


class Stacking:
    """Dùng OOF làm đặc trưng đầu vào cho một model cấp hai (meta model).

    Khác Blending ở chỗ meta model có thể học hệ số âm và hệ số tự do,
    nên linh hoạt hơn nhưng cũng dễ quá khớp hơn khi ít dữ liệu.
    """

    def __init__(self, meta=None, alpha: float = 1.0):
        self.meta = meta if meta is not None else Ridge(alpha=alpha, positive=True)
        self.names_: list[str] = []

    def fit(self, oof: dict[str, np.ndarray], y) -> "Stacking":
        self.names_ = list(oof.keys())
        P = np.column_stack([oof[n] for n in self.names_])
        self.meta.fit(P, np.asarray(y))
        return self

    def predict(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        P = np.column_stack([preds[n] for n in self.names_])
        return self.meta.predict(P)

    @property
    def weights(self) -> dict:
        return dict(zip(self.names_, getattr(self.meta, "coef_", [])))


def bagging(estimator, n_estimators: int = 10, seed: int = 42, **kw):
    """Huấn luyện nhiều bản cùng model trên các mẫu bootstrap rồi lấy trung bình.

    Giảm phương sai, hữu ích khi một model đơn lẻ không ổn định giữa các fold.
    """
    return BaggingRegressor(
        estimator=estimator, n_estimators=n_estimators,
        random_state=seed, **kw
    )


def build(method: str, names: list[str], cfg: dict):
    """Tạo bộ kết hợp theo tên ghi trong config."""
    method = method.lower()
    if method == "voting":
        return Voting(names)
    if method == "blending":
        w = cfg.get("ensemble", {}).get("weights")
        # Có trọng số trong config thì dùng, không thì để fit() tự tìm
        if w and all(n in w for n in names):
            return Blending({n: w[n] for n in names})
        return Blending()
    if method == "stacking":
        return Stacking(alpha=cfg.get("ensemble", {}).get("meta_alpha", 1.0))
    raise ValueError(f"Chưa hỗ trợ phương pháp: {method}")


def compare_methods(oof: dict[str, np.ndarray], y) -> pd.DataFrame:
    """So sánh cả ba cách kết hợp trên OOF. Dùng cho báo cáo của người 4."""
    rows = []
    for name, ens in [
        ("voting", Voting(list(oof))),
        ("blending", Blending()),
        ("stacking", Stacking()),
    ]:
        ens.fit(oof, y)
        rows.append({
            "method": name,
            "rmse_oof": round(rmse(y, ens.predict(oof)), 5),
            "weights": {k: round(float(v), 3) for k, v in (ens.weights or {}).items()},
        })
    for name, arr in oof.items():
        rows.append({"method": f"(đơn lẻ) {name}",
                     "rmse_oof": round(rmse(y, arr), 5), "weights": {}})
    return pd.DataFrame(rows).sort_values("rmse_oof").reset_index(drop=True)