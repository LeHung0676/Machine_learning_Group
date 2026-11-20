"""Xây dựng và lựa chọn đặc trưng (feature engineering & selection).

- add_time_features   : đặc trưng thời gian trích xuất từ cột `date`
- add_lag_features    : đặc trưng quá khứ gần của biến mục tiêu (10–60 phút trước) – phục vụ RQ4
- get_feature_columns : danh sách đặc trưng đưa vào mô hình theo từng bộ (feature set)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATE_COL, HUMIDITY_COLS, OTHER_COLS, TARGET, TEMP_COLS, WEATHER_COLS

TIME_FEATURES = ["hour", "day_of_week", "is_weekend", "nsm", "hour_sin", "hour_cos"]

# `hour` trùng thông tin với `nsm` (nsm chi tiết hơn tới từng 10 phút) → chỉ đưa nsm + sin/cos vào mô hình.
MODEL_TIME_FEATURES = ["nsm", "hour_sin", "hour_cos", "day_of_week", "is_weekend"]

# Đặc trưng quá khứ gần (RQ4): giá trị tiêu thụ 10, 20, 30, 60 phút trước, trung bình trượt 30 và 60 phút
# của các giá trị ĐÃ QUAN SÁT (không gồm thời điểm hiện tại) và mức thay đổi gần nhất.
LAG_STEPS = (1, 2, 3, 6)            # 1 bước = 10 phút
ROLL_WINDOWS = (3, 6)
LAG_FEATURES = (
    [f"lag_{k * 10}m" for k in LAG_STEPS]
    + [f"roll_mean_{w * 10}m" for w in ROLL_WINDOWS]
    + ["diff_10m"]
)

# Các bộ đặc trưng dùng để so sánh trong thực nghiệm.
#  - "sensors"   : chỉ cảm biến trong nhà (T1–T9, RH_1–RH_9)
#  - "no_time"   : cảm biến + thời tiết + lights (gần với thiết lập của bài báo gốc)
#  - "full"      : no_time + đặc trưng thời gian  ← bộ chính khi KHÔNG có số đo điện quá khứ
#  - "full_lag"  : full + đặc trưng quá khứ gần (notebook 08, RQ4)
#  - "lag_time"  : chỉ thời gian + quá khứ gần – mô hình dự báo 10 phút tới gọn nhất (notebook 08, 09, app/)
# `rv1` được giữ trong mọi bộ như "biến đối chứng" (xem ghi chú bên dưới); `rv2` bị loại vì trùng hoàn toàn rv1.
FEATURE_SETS = {
    "sensors": TEMP_COLS + HUMIDITY_COLS,
    "no_time": TEMP_COLS + HUMIDITY_COLS + WEATHER_COLS + OTHER_COLS,
    "full": TEMP_COLS + HUMIDITY_COLS + WEATHER_COLS + OTHER_COLS + MODEL_TIME_FEATURES,
    "full_lag": TEMP_COLS + HUMIDITY_COLS + WEATHER_COLS + OTHER_COLS + MODEL_TIME_FEATURES + LAG_FEATURES,
    "lag_time": MODEL_TIME_FEATURES + LAG_FEATURES,
}
CONTROL_FEATURE = "rv1"
DROPPED_FEATURES = {
    "rv2": "trùng hoàn toàn với rv1",
    "hour": "trùng thông tin với nsm",
}


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


def add_lag_features(df: pd.DataFrame, target: str = TARGET, date_col: str = DATE_COL) -> pd.DataFrame:
    """Thêm đặc trưng quá khứ gần của biến mục tiêu. Trả về bản sao, không sửa df gốc.

    - lag_10m, lag_20m, lag_30m, lag_60m : tiêu thụ 1, 2, 3, 6 bước (10 phút/bước) trước
    - roll_mean_30m, roll_mean_60m       : trung bình 3 và 6 giá trị gần nhất TRƯỚC thời điểm hiện tại
    - diff_10m                           : lag_10m − lag_20m (xu hướng tăng/giảm gần nhất)

    Chống rò rỉ: mọi đặc trưng chỉ dùng giá trị ở thời điểm t−1 trở về trước (`shift(1)` trước khi lấy trung bình
    trượt), nên không bao giờ chứa y(t). Có thể tính trên toàn bộ chuỗi TRƯỚC khi chia train/val/test: các dòng đầu
    tập val/test dùng giá trị cuối tập trước – đó là quá khứ thật, trong thực tế luôn đo được.

    Dữ liệu phải liên tục mỗi 10 phút (đã kiểm tra trong `basic_report`). 6 dòng đầu chuỗi sẽ có NaN → bỏ khi huấn luyện.
    """
    out = df.sort_values(date_col).reset_index(drop=True).copy()
    y = out[target]
    for k in LAG_STEPS:
        out[f"lag_{k * 10}m"] = y.shift(k)
    past = y.shift(1)
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w * 10}m"] = past.rolling(w).mean()
    out["diff_10m"] = y.shift(1) - y.shift(2)
    return out


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
    """Tách X (đặc trưng) và y (mục tiêu). Tự thêm đặc trưng thời gian nếu cần.

    Với bộ có lag ("full_lag", "lag_time"): df phải được thêm đặc trưng lag bằng `add_lag_features` TRƯỚC khi chia
    train/val/test; các dòng còn NaN (đầu chuỗi) bị bỏ.
    """
    if "nsm" not in df.columns:
        df = add_time_features(df)
    cols = get_feature_columns(feature_set, include_control)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột {missing}. Với bộ có lag, gọi add_lag_features(df) trước khi time_split.")
    lag_cols = [c for c in cols if c in LAG_FEATURES]
    if lag_cols:
        df = df.dropna(subset=lag_cols)
    return df[cols], df[TARGET]


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    data = add_time_features(load_raw_data())
    print(data[["date", *TIME_FEATURES]].head(8))
    print()
    for name in FEATURE_SETS:
        print(f"{name:>8}: {len(get_feature_columns(name))} đặc trưng")
    lagged = add_lag_features(data)
    print(lagged[["date", "Appliances", *LAG_FEATURES]].head(8))
    print("Loại bỏ:", DROPPED_FEATURES)
