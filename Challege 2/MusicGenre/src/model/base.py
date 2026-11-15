"""Giao diện chung cho mọi model phân loại.

Mỗi model là một sklearn Pipeline gồm hai bước: bộ mã hóa (cột số + target
encoding) rồi tới bộ phân loại. Nhờ bộ mã hóa nằm TRONG pipeline, mỗi lần
fit trên một fold thì target encoding chỉ thấy nhãn của fold đó.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline


class BaseModel:
    name = "base"
    scale = False          # model tuyến tính đặt True

    def __init__(self, cfg: dict, pre):
        self.cfg = cfg
        self.pre = pre
        self.seed = cfg["project"]["seed"]
        self.params = dict(cfg["models"].get(self.name, {}))
        self.model = Pipeline([
            ("enc", pre.make_encoder(scale=self.scale, seed=self.seed)),
            ("clf", self.build_estimator()),
        ])

    def build_estimator(self):
        raise NotImplementedError

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self.model.classes_[self.predict_proba(X).argmax(axis=1)]

    @property
    def classes_(self):
        return self.model.classes_

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        return path
