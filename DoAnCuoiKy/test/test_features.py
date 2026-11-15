"""Kiểm thử hàm đặc trưng thời gian.

Chạy:  python -m pytest test/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import TIME_FEATURES, add_time_features  # noqa: E402


def _sample():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2016-01-11 00:00:00",  # Thứ Hai, nửa đêm
            "2016-01-16 12:30:00",  # Thứ Bảy, giữa trưa
            "2016-01-17 23:50:00",  # Chủ Nhật, gần nửa đêm
        ]),
        "Appliances": [50, 60, 70],
    })


def test_columns_added_and_original_untouched():
    df = _sample()
    out = add_time_features(df)
    for col in TIME_FEATURES:
        assert col in out.columns
    assert "hour" not in df.columns  # không sửa dataframe gốc


def test_values():
    out = add_time_features(_sample())
    assert out["hour"].tolist() == [0, 12, 23]
    assert out["day_of_week"].tolist() == [0, 5, 6]
    assert out["is_weekend"].tolist() == [0, 1, 1]
    assert out["nsm"].tolist() == [0, 12 * 3600 + 30 * 60, 23 * 3600 + 50 * 60]


def test_cyclic_encoding_midnight_close_to_late_night():
    out = add_time_features(_sample())
    p0 = np.array([out.loc[0, "hour_sin"], out.loc[0, "hour_cos"]])
    p2 = np.array([out.loc[2, "hour_sin"], out.loc[2, "hour_cos"]])
    p1 = np.array([out.loc[1, "hour_sin"], out.loc[1, "hour_cos"]])
    # 23:50 phải gần 00:00 hơn nhiều so với 12:30
    assert np.linalg.norm(p0 - p2) < np.linalg.norm(p0 - p1)
