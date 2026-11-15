"""Model ElasticNet: kết hợp phạt L1 và L2.

Nằm giữa Lasso (chỉ L1) và Ridge (chỉ L2). Khi nhiều đặc trưng tương quan
với nhau, Lasso có xu hướng chọn bừa một cột và bỏ các cột còn lại;
ElasticNet giữ lại cả nhóm nên thường ổn định hơn.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV

from src.model.base import BaseModel


class ElasticNetModel(BaseModel):
    name = "elasticnet"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        p = cfg["models"].get("elasticnet", {})
        self.model = ElasticNet(
            alpha=p.get("alpha", 0.0005),
            l1_ratio=p.get("l1_ratio", 0.5),
            max_iter=p.get("max_iter", 50000),
            random_state=cfg["project"]["seed"],
        )

    @staticmethod
    def tune(X, y, n_folds: int = 5, seed: int = 42) -> dict:
        cv = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            alphas=np.logspace(-5, -1, 40),
            cv=n_folds, max_iter=50000, random_state=seed, n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cv.fit(X, y)
        return {"alpha": float(cv.alpha_), "l1_ratio": float(cv.l1_ratio_),
                "n_features_used": int((cv.coef_ != 0).sum())}
