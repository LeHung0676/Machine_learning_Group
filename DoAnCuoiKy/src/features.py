"""Xây dựng đặc trưng (feature engineering).

Hiện tại: đặc trưng thời gian trích xuất từ cột `date`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATE_COL

TIME_FEATURES = ["hour", "day_of_week", "is_weekend", "nsm", "hour_sin", "hour_cos"]


def add_time_features(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """Thêm các đặc trưng thời gian. Trả về bản sao, không sửa df gốc.

    - hour        : giờ trong ngày (0–23)
    - day_of_week : thứ trong tuần (0 = Thứ Hai, ..., 6 = Chủ Nhật)
    - is_weekend  : 1 nếu Thứ Bảy/Chủ Nhật
    - nsm         : số giây tính từ nửa đêm (Number of Seconds from Midnight) – như bài báo gốc
    - hour_sin/cos: mã hóa chu kỳ của giờ, để mô hình hiểu 23h và 0h là "gần nhau"

    Không dùng `month`: dữ liệu chỉ kéo dài ~4,5 tháng và sẽ chia train/test theo thời gian,
    nên tháng trong tập test không xuất hiện trong tập train → đặc trưng này không tổng quát hóa được.
    """
    out = df.copy()
    dt = out[date_col]
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["nsm"] = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
    angle = 2 * np.pi * out["nsm"] / 86400
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    return out


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    data = add_time_features(load_raw_data())
    print(data[["date", *TIME_FEATURES]].head(8))
