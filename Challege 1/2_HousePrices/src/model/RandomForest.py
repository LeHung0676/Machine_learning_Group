"""Model Random Forest.

Riêng lẻ thì yếu hơn các model tuyến tính trên bộ này, nhưng nó sai theo
kiểu khác, nên vẫn góp phần khi trộn model.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from src.model.base import BaseModel


class RandomForestModel(BaseModel):
    name = "randomforest"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        p = cfg["models"].get("randomforest", {})
        self.model = RandomForestRegressor(
            n_estimators=p.get("n_estimators", 300),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            max_features=p.get("max_features", "sqrt"),
            random_state=cfg["project"]["seed"],
            n_jobs=-1,
        )
