"""Tiền xử lý: phân nhóm cột và dựng bộ mã hóa dùng chung cho mọi model.

Khác Challenge 1 ở một điểm quan trọng: target encoding KHÔNG được tính
trước một lần cho toàn bộ train, vì như vậy nhãn của chính mỗi dòng sẽ rò
vào đặc trưng của nó. Bộ mã hóa được đặt BÊN TRONG pipeline của từng model,
nên trong mỗi fold nó chỉ học từ phần huấn luyện của fold đó.
"""
from __future__ import annotations

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder

TEXT_COLS = ["Artist Name", "Track Name", "pair"]

_SK_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])


def make_target_encoder(seed: int = 42) -> TargetEncoder:
    """TargetEncoder đa lớp, tương thích nhiều phiên bản scikit-learn.

    Từ bản 1.9, tham số random_state bị thay bằng việc truyền bộ chia fold
    qua `cv`. Bản cũ hơn chỉ nhận `cv` là số nguyên kèm random_state.
    """
    if _SK_VERSION >= (1, 9):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return TargetEncoder(target_type="multiclass", cv=cv)
    return TargetEncoder(target_type="multiclass", cv=5, shuffle=True, random_state=seed)


class Preprocessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.te_cols: list[str] = list(cfg.get("features", {}).get("target_encode", ["Artist Name", "pair"]))
        self.num_cols_: list[str] = []
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.num_cols_ = [c for c in X.columns if c not in TEXT_COLS]
        non_numeric = [c for c in self.num_cols_ if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            raise ValueError(f"Cột không phải số lọt vào nhóm số: {non_numeric}")
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Thiếu cột so với lúc fit: {missing}")
        X = X[self.columns_].copy()
        if X[self.num_cols_].isna().any().any():
            raise ValueError("Còn giá trị thiếu trong cột số sau khi làm sạch")
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def make_encoder(self, scale: bool = False, seed: int = 42) -> ColumnTransformer:
        """Bộ mã hóa đặt trong pipeline của model: cột số + target encoding."""
        steps = [("num", StandardScaler() if scale else "passthrough", self.num_cols_)]
        for i, c in enumerate(self.te_cols):
            steps.append((f"te{i}", make_target_encoder(seed), [c]))
        return ColumnTransformer(steps)
