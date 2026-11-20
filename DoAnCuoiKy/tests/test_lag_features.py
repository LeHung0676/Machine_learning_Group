"""Kiểm thử đặc trưng quá khứ gần (RQ4).

Chạy:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import LAG_FEATURES, LAG_STEPS, add_lag_features, add_time_features, make_xy  # noqa: E402

N_HISTORY = max(LAG_STEPS)


def _series(n=20):
    dates = pd.date_range("2016-01-11 17:00", periods=n, freq="10min")
    y = np.arange(10, 10 * (n + 1), 10, dtype=float)       # 10, 20, 30, ... dễ kiểm tra bằng tay
    return pd.DataFrame({"date": dates, "Appliances": y})


def test_lag_chi_dung_qua_khu():
    out = add_lag_features(_series())
    row = out.iloc[10]                                        # y(t) = 110
    assert row["Appliances"] == 110
    assert row["lag_10m"] == 100 and row["lag_20m"] == 90 and row["lag_30m"] == 80 and row["lag_60m"] == 50
    assert row["roll_mean_30m"] == pytest.approx((100 + 90 + 80) / 3)
    assert row["roll_mean_60m"] == pytest.approx((100 + 90 + 80 + 70 + 60 + 50) / 6)
    assert row["diff_10m"] == 10
    # Không đặc trưng nào được chứa giá trị hiện tại y(t)
    assert all((out[c] != out["Appliances"]).all() for c in LAG_FEATURES)


def test_lag_nan_chi_o_dau_chuoi_va_make_xy_bo_nan():
    out = add_lag_features(add_time_features(_series()))
    assert out[LAG_FEATURES].isna().any(axis=1).sum() == N_HISTORY
    X, y = make_xy(out, "lag_time", include_control=False)
    assert len(X) == len(out) - N_HISTORY
    assert not X.isna().any().any()


def test_full_lag_thieu_cot_bao_loi_ro_rang():
    with pytest.raises(KeyError, match="add_lag_features"):
        make_xy(add_time_features(_series()), "full_lag")
