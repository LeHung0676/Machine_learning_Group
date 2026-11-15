"""Giao diện chung cho mọi model, để Pipeline và Ensemble dùng thay thế được nhau."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class BaseModel:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        """X_val/y_val là tùy chọn, chỉ model nào hỗ trợ early stopping mới dùng."""
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self, feature_names=None) -> pd.Series:
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_)
        else:
            raise AttributeError(f"{self.name} không cung cấp độ quan trọng đặc trưng")
        names = feature_names if feature_names is not None else range(len(imp))
        return pd.Series(imp, index=names).sort_values(ascending=False)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        return path

    def load(self, path):
        self.model = joblib.load(path)
        return self