"""Đánh giá model hồi quy: MAE, RMSE, R², RMSLE + cross-validation."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


class Evaluator:
    @staticmethod
    def regression_metrics(y_true, y_pred) -> dict:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def rmsle(y_true, y_pred) -> float:
        """Metric chính thức của Kaggle (trên giá gốc)."""
        return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))))

    @staticmethod
    def cross_validate(model_factory, X, y, n_folds: int = 5, seed: int = 42) -> dict:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for tr, va in kf.split(X):
            m = model_factory().fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[va])
            scores.append(np.sqrt(mean_squared_error(y.iloc[va], pred)))
        return {"cv_rmse_mean": float(np.mean(scores)), "cv_rmse_std": float(np.std(scores))}
