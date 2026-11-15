<<<<<<< HEAD
"""Xây dựng và lựa chọn đặc trưng (feature engineering & selection).

- add_time_features   : đặc trưng thời gian trích xuất từ cột `date`
- get_feature_columns : danh sách đặc trưng đưa vào mô hình theo từng bộ (feature set)
=======
"""Xây dựng đặc trưng (feature engineering).

Hiện tại: đặc trưng thời gian trích xuất từ cột `date`.
>>>>>>> 4d1608ea44602e52ed03a5da141d94c452ca596c
"""
from __future__ import annotations

import numpy as np
import pandas as pd

<<<<<<< HEAD
from src.config import DATE_COL, HUMIDITY_COLS, OTHER_COLS, TARGET, TEMP_COLS, WEATHER_COLS

TIME_FEATURES = ["hour", "day_of_week", "is_weekend", "nsm", "hour_sin", "hour_cos"]

# `hour` trùng thông tin với `nsm` (nsm chi tiết hơn tới từng 10 phút) → chỉ đưa nsm + sin/cos vào mô hình.
MODEL_TIME_FEATURES = ["nsm", "hour_sin", "hour_cos", "day_of_week", "is_weekend"]

# Các bộ đặc trưng dùng để so sánh trong thực nghiệm.
#  - "sensors"   : chỉ cảm biến trong nhà (T1–T9, RH_1–RH_9)
#  - "no_time"   : cảm biến + thời tiết + lights (gần với thiết lập của bài báo gốc)
#  - "full"      : no_time + đặc trưng thời gian  ← bộ chính của nhóm
# `rv1` được giữ trong mọi bộ như "biến đối chứng" (xem ghi chú bên dưới); `rv2` bị loại vì trùng hoàn toàn rv1.
FEATURE_SETS = {
    "sensors": TEMP_COLS + HUMIDITY_COLS,
    "no_time": TEMP_COLS + HUMIDITY_COLS + WEATHER_COLS + OTHER_COLS,
    "full": TEMP_COLS + HUMIDITY_COLS + WEATHER_COLS + OTHER_COLS + MODEL_TIME_FEATURES,
}
CONTROL_FEATURE = "rv1"
DROPPED_FEATURES = {
    "rv2": "trùng hoàn toàn với rv1",
    "hour": "trùng thông tin với nsm",
}

=======
from src.config import DATE_COL

TIME_FEATURES = ["hour", "day_of_week", "is_weekend", "nsm", "hour_sin", "hour_cos"]

>>>>>>> 4d1608ea44602e52ed03a5da141d94c452ca596c

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


<<<<<<< HEAD
def get_feature_columns(feature_set: str = "full", include_control: bool = True) -> list[str]:
    """Trả về danh sách cột đặc trưng của một bộ.

    include_control=True giữ `rv1` – một biến hoàn toàn ngẫu nhiên. Khi phân tích độ quan trọng đặc trưng, mọi
    đặc trưng xếp hạng thấp hơn rv1 được xem là không mang thông tin hữu ích (ý tưởng của bài báo gốc).
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Không có bộ đặc trưng '{feature_set}'. Chọn trong: {list(FEATURE_SETS)}")
    cols = list(FEATURE_SETS[feature_set])
    if include_control:
        cols.append(CONTROL_FEATURE)
    return cols


def make_xy(df: pd.DataFrame, feature_set: str = "full", include_control: bool = True):
    """Tách X (đặc trưng) và y (mục tiêu). Tự thêm đặc trưng thời gian nếu cần."""
    if "nsm" not in df.columns:
        df = add_time_features(df)
    cols = get_feature_columns(feature_set, include_control)
    return df[cols], df[TARGET]


=======
>>>>>>> 4d1608ea44602e52ed03a5da141d94c452ca596c
if __name__ == "__main__":
    from src.data_loader import load_raw_data

    data = add_time_features(load_raw_data())
    print(data[["date", *TIME_FEATURES]].head(8))
<<<<<<< HEAD
    print()
    for name in FEATURE_SETS:
        print(f"{name:>8}: {len(get_feature_columns(name))} đặc trưng")
    print("Loại bỏ:", DROPPED_FEATURES)
=======
>>>>>>> 4d1608ea44602e52ed03a5da141d94c452ca596c
