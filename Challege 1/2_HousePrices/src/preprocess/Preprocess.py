"""Chuẩn hóa, xử lý lệch phân phối, mã hóa one-hot, scale.

Mọi tham số đều học từ train (fit_transform) rồi áp cho test (transform).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


class Preprocessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg["preprocess"]
        self.scaler = SCALERS[self.cfg["scaler"]]()
        self.skewed_cols_: list[str] = []
        self.num_cols_: list[str] = []
        self.cat_levels_: dict[str, list] = {}
        self.columns_: list[str] = []

    # ---------- fit + transform trên train ----------
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        num = X.select_dtypes("number")
        cat = X.select_dtypes(exclude="number")

        # 1. Chỉ xử lý lệch cho cột liên tục.
        #    Cột nhị phân (0/1) hay thứ bậc ít giá trị thì log1p vô nghĩa.
        min_unique = self.cfg.get("skew_min_unique", 15)
        candidates = [c for c in num.columns if num[c].nunique() > min_unique]
        s = num[candidates].apply(lambda col: skew(col.dropna()))
        self.skewed_cols_ = s[s.abs() > self.cfg["skew_threshold"]].index.tolist()

        # 2. Ghi nhớ danh sách giá trị của từng cột chữ,
        #    để test sinh ra đúng bộ cột one-hot như train.
        self.cat_levels_ = {c: sorted(cat[c].astype(str).unique()) for c in cat.columns}

        X = self._apply(X)
        self.columns_ = X.columns.tolist()

        # 3. Scale: chỉ fit trên train
        self.num_cols_ = [c for c in num.columns if c in X.columns]
        X[self.num_cols_] = self.scaler.fit_transform(X[self.num_cols_])
        return X

    # ---------- transform trên test ----------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._apply(X)
        X = X.reindex(columns=self.columns_, fill_value=0)
        X[self.num_cols_] = self.scaler.transform(X[self.num_cols_])
        return X

    # ---------- dùng chung ----------
    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Kéo đuôi phân phối lệch về gần chuẩn.
        # Lasso là model tuyến tính nên rất cần bước này; LightGBM thì không,
        # nhưng dùng chung một bộ dữ liệu cho dễ so sánh.
        for c in self.skewed_cols_:
            if c in X.columns:
                X[c] = np.log1p(X[c].clip(lower=0))

        # One-hot với bộ giá trị đã chốt từ train.
        # Nếu test có giá trị lạ, nó bị gán vào NaN rồi bỏ,
        # thay vì sinh ra một cột mới mà model chưa từng thấy.
        for c, levels in self.cat_levels_.items():
            if c in X.columns:
                X[c] = pd.Categorical(X[c].astype(str), categories=levels)
        cat_cols = list(self.cat_levels_.keys())
        cat_cols = [c for c in cat_cols if c in X.columns]
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, dummy_na=False, dtype=float)
        return X

    def feature_names(self) -> list[str]:
        return self.columns_