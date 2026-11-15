"""Đánh giá model hồi quy: metrics, cross-validation sinh OOF, biểu đồ.

Mọi đánh giá làm trên thang log(SalePrice), vì đó là metric Kaggle dùng.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class Evaluator:
    # ---------- metrics ----------
    @staticmethod
    def regression_metrics(y_true, y_pred) -> dict:
        return {
            "rmse": rmse(y_true, y_pred),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def rmsle(y_true_price, y_pred_price) -> float:
        """Metric chính thức của Kaggle, tính trên giá gốc chưa lấy log."""
        return rmse(np.log1p(y_true_price), np.log1p(np.clip(y_pred_price, 0, None)))

    # ---------- cross-validation ----------
    @staticmethod
    def oof_predict(model_factory, X, y, X_test=None, n_folds: int = 5,
                    seed: int = 42) -> dict:
        """Chia K fold, trả về dự đoán out-of-fold và dự đoán test bình quân.

        OOF nghĩa là mỗi dòng của train được dự đoán bởi model CHƯA từng
        thấy dòng đó. Đây là cách duy nhất để tìm trọng số ensemble mà
        không rò rỉ dữ liệu.

        Trả về dict: oof, test_pred, fold_scores, rmse_mean, rmse_std.
        """
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        oof = np.zeros(len(X))
        test_parts = []
        scores = []

        for tr_idx, va_idx in kf.split(X):
            m = model_factory()
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx])
            oof[va_idx] = m.predict(X.iloc[va_idx])
            scores.append(rmse(y.iloc[va_idx], oof[va_idx]))
            if X_test is not None:
                test_parts.append(m.predict(X_test))

        return {
            "oof": oof,
            "test_pred": np.mean(test_parts, axis=0) if test_parts else None,
            "fold_scores": [float(s) for s in scores],
            "rmse_mean": float(np.mean(scores)),
            "rmse_std": float(np.std(scores)),
            "rmse_oof": rmse(y, oof),
        }

    # ---------- bảng so sánh ----------
    @staticmethod
    def compare(results: dict) -> pd.DataFrame:
        """results: {ten_model: ket_qua_tu_oof_predict}. Dùng cho báo cáo."""
        rows = []
        for name, r in results.items():
            rows.append({
                "model": name,
                "rmse_oof": round(r["rmse_oof"], 5),
                "rmse_cv_mean": round(r["rmse_mean"], 5),
                "rmse_cv_std": round(r["rmse_std"], 5),
            })
        return pd.DataFrame(rows).sort_values("rmse_oof").reset_index(drop=True)

    # ---------- biểu đồ cho báo cáo ----------
    @staticmethod
    def plot_pred_vs_actual(y_true, y_pred, title="", fig_dir=None, fname=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.4, s=18)
        lim = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("Thực tế (log)")
        ax.set_ylabel("Dự đoán (log)")
        ax.set_title(title or f"Dự đoán vs thực tế, RMSE={rmse(y_true, y_pred):.4f}")
        Evaluator._save(fig, fig_dir, fname or "pred_vs_actual.png")

    @staticmethod
    def plot_residuals(y_true, y_pred, fig_dir=None, fname=None):
        """Phần dư nên rải đều quanh 0. Nếu có hình phễu thì model còn thiếu sót."""
        import matplotlib.pyplot as plt
        resid = np.asarray(y_true) - np.asarray(y_pred)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].scatter(y_pred, resid, alpha=0.4, s=18)
        ax[0].axhline(0, color="r", ls="--", lw=1)
        ax[0].set_xlabel("Dự đoán (log)")
        ax[0].set_ylabel("Phần dư")
        ax[1].hist(resid, bins=40)
        ax[1].set_title("Phân phối phần dư")
        fig.tight_layout()
        Evaluator._save(fig, fig_dir, fname or "residuals.png")

    @staticmethod
    def plot_feature_importance(importance: pd.Series, k: int = 25,
                                title="", fig_dir=None, fname=None):
        import matplotlib.pyplot as plt
        top = importance.head(k)[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(top))))
        ax.barh(top.index.astype(str), top.values, color="steelblue")
        ax.set_title(title or f"{k} đặc trưng quan trọng nhất")
        fig.tight_layout()
        Evaluator._save(fig, fig_dir, fname or "feature_importance.png")

    @staticmethod
    def _save(fig, fig_dir, name):
        import matplotlib.pyplot as plt
        if fig_dir is not None:
            from pathlib import Path
            d = Path(fig_dir); d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / name, bbox_inches="tight", dpi=120)
        plt.close(fig)