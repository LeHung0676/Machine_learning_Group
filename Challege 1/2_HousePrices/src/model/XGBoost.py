"""Model XGBoost. Import xgboost đặt trong hàm để ai chưa cài vẫn chạy được."""
from __future__ import annotations

from src.model.base import BaseModel


class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from xgboost import XGBRegressor

        p = dict(cfg["models"].get("xgboost", {}))
        self.early_stopping_rounds = p.pop("early_stopping_rounds", 200)
        self.model = XGBRegressor(
            n_estimators=p.get("n_estimators", 2000),
            learning_rate=p.get("learning_rate", 0.02),
            max_depth=p.get("max_depth", 3),
            min_child_weight=p.get("min_child_weight", 2),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.6),
            reg_alpha=p.get("reg_alpha", 0.1),
            reg_lambda=p.get("reg_lambda", 1.0),
            random_state=cfg["project"]["seed"],
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            # Không có tập validation thì phải tắt early stopping,
            # nếu không XGBoost sẽ báo lỗi vì thiếu eval_set.
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X, y, verbose=False)
        return self
