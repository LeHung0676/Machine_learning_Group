"""Độ đo đánh giá và các mô hình baseline.

Độ đo (tính trên đơn vị Wh gốc):
- MAE  : sai số tuyệt đối trung bình – dễ hiểu ("trung bình lệch bao nhiêu Wh"), ít nhạy với ngoại lệ.
- RMSE : căn của sai số bình phương trung bình – phạt nặng các lỗi lớn (các đỉnh tiêu thụ). Độ đo chính,
         cũng là độ đo bài báo gốc dùng → so sánh được.
- R²   : tỷ lệ phương sai được giải thích; 0 = không hơn dự đoán trung bình, 1 = hoàn hảo.
- MAPE : sai số phần trăm trung bình – dùng được vì Appliances luôn ≥ 10 Wh (không chia cho 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Trả về dict gồm MAE, RMSE, R², MAPE (%)."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": 100 * mean_absolute_percentage_error(y_true, y_pred),
    }


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Gộp nhiều kết quả {tên mô hình: dict độ đo} thành bảng, sắp theo RMSE tăng dần."""
    return pd.DataFrame(results).T.sort_values("RMSE").round(3)


# ---------------------------------------------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------------------------------------------
class MeanBaseline(BaseEstimator, RegressorMixin):
    """Luôn dự đoán trung bình của tập train. Mốc thấp nhất: R² ≈ 0."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class HourMeanBaseline(BaseEstimator, RegressorMixin):
    """Dự đoán trung bình tiêu thụ theo **giờ trong ngày** (học từ train).

    Baseline "thông minh" hơn: tận dụng chu kỳ ngày đã thấy trong EDA (đỉnh 18h). Mô hình học máy phải vượt
    mốc này mới chứng tỏ dữ liệu cảm biến/thời tiết có thêm giá trị.
    Cần cột `nsm` (giây tính từ nửa đêm) trong X.
    """

    def fit(self, X, y):
        hours = (np.asarray(X["nsm"]) // 3600).astype(int)
        s = pd.Series(np.asarray(y)).groupby(hours).mean()
        self.global_mean_ = float(np.mean(y))
        self.hour_means_ = s.to_dict()
        return self

    def predict(self, X):
        hours = (np.asarray(X["nsm"]) // 3600).astype(int)
        return np.array([self.hour_means_.get(h, self.global_mean_) for h in hours])
