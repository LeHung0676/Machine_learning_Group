"""Model LightGBM.

Cây quyết định tăng cường. Không cần scale, nhưng ta dùng chung bộ dữ liệu
đã xử lý với Lasso để hai model so sánh được công bằng.
Import lightgbm đặt trong hàm để ai chưa cài vẫn chạy được phần Lasso.
"""
from __future__ import annotations

import numpy as np

from src.model.base import BaseModel


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from lightgbm import LGBMRegressor

        p = dict(cfg["models"]["lightgbm"])
        self.early_stopping_rounds = p.pop("early_stopping_rounds", 200)
        self.model = LGBMRegressor(
            **p, random_state=cfg["project"]["seed"], verbose=-1
        )

    def fit(self, X, y, X_val=None, y_val=None):
        """Có tập validation thì dùng early stopping để không bị quá khớp."""
        if X_val is not None and y_val is not None:
            from lightgbm import early_stopping, log_evaluation

            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(self.early_stopping_rounds, verbose=False),
                    log_evaluation(0),
                ],
            )
        else:
            self.model.fit(X, y)
        return self

    @property
    def best_iteration(self):
        return getattr(self.model, "best_iteration_", None)

    @staticmethod
    def tune(X, y, n_iter: int = 30, n_folds: int = 5, seed: int = 42) -> dict:
        """Dò tham số ngẫu nhiên. Chạy khá lâu, nên để n_iter nhỏ lúc thử.

        Trả về dict 'params' tốt nhất và 'rmse_cv'.
        """
        from lightgbm import LGBMRegressor
        from sklearn.model_selection import KFold, RandomizedSearchCV

        space = {
            "n_estimators": [1000, 2000, 3000],
            "learning_rate": [0.005, 0.01, 0.02, 0.05],
            "num_leaves": [7, 15, 31, 63],
            "min_child_samples": [5, 10, 20, 30],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "reg_lambda": [0, 0.1, 0.5, 1.0],
        }
        search = RandomizedSearchCV(
            LGBMRegressor(random_state=seed, verbose=-1),
            param_distributions=space,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed),
            random_state=seed,
            n_jobs=-1,
        )
        search.fit(X, y)
        return {
            "params": search.best_params_,
            "rmse_cv": float(-search.best_score_),
        }