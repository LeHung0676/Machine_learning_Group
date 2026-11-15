"""Model Lasso Regression.

Lasso là model tuyến tính có phạt L1, nên nó tự đẩy hệ số của các đặc trưng
vô dụng về 0. Rất phù hợp ở đây vì sau one-hot ta có hàng trăm cột.
Bắt buộc phải scale dữ liệu trước, Preprocessor đã làm việc đó.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV

from src.model.base import BaseModel


class LassoModel(BaseModel):
    name = "lasso"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        p = cfg["models"]["lasso"]
        self.model = Lasso(
            alpha=p["alpha"],
            max_iter=p.get("max_iter", 50000),
            random_state=cfg["project"]["seed"],
        )

    @staticmethod
    def tune(X, y, n_folds: int = 5, seed: int = 42, alphas=None) -> dict:
        """Dò alpha bằng LassoCV. Người 3 chạy hàm này rồi đưa kết quả cho người 4.

        Trả về dict có 'alpha' tốt nhất và 'rmse_cv' tương ứng.
        """
        if alphas is None:
            alphas = np.logspace(-5, -1, 60)
        cv = LassoCV(alphas=alphas, cv=n_folds, max_iter=50000, random_state=seed)
        # Các alpha rất nhỏ ở đầu dải luôn sinh ConvergenceWarning và không
        # bao giờ được chọn, nên ta tắt cảnh báo để kết quả dễ đọc.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cv.fit(X, y)
        # mse_path_ có dạng (n_alphas, n_folds); LassoCV sắp alphas giảm dần
        idx = list(cv.alphas_).index(cv.alpha_)
        rmse = float(np.sqrt(cv.mse_path_[idx].mean()))
        n_used = int((cv.coef_ != 0).sum())
        return {
            "alpha": float(cv.alpha_),
            "rmse_cv": rmse,
            "n_features_used": n_used,
            "n_features_total": X.shape[1],
        }