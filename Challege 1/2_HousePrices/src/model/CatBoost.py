"""Model CatBoost. Import catboost đặt trong hàm để ai chưa cài vẫn chạy được."""
from __future__ import annotations

from src.model.base import BaseModel


class CatBoostModel(BaseModel):
    name = "catboost"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from catboost import CatBoostRegressor

        p = dict(cfg["models"].get("catboost", {}))
        self.early_stopping_rounds = p.pop("early_stopping_rounds", 200)
        self.model = CatBoostRegressor(
            iterations=p.get("iterations", 2000),
            learning_rate=p.get("learning_rate", 0.03),
            depth=p.get("depth", 4),
            l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
            random_seed=cfg["project"]["seed"],
            verbose=0,
            allow_writing_files=False,   # không rải catboost_info/ vào repo
        )

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(X, y, eval_set=(X_val, y_val),
                           early_stopping_rounds=self.early_stopping_rounds,
                           verbose=0)
        else:
            self.model.fit(X, y, verbose=0)
        return self
