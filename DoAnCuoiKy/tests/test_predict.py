"""Kiểm thử hàm dự báo dùng cho demo `app/` (src/predict.py).

Chạy:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import LAG_FEATURES, add_lag_features, add_time_features  # noqa: E402
from src.predict import MODEL_PATH, N_HISTORY, features_from_history, load_model, predict_next  # noqa: E402


def _series(n=20):
    dates = pd.date_range("2016-01-11 17:00", periods=n, freq="10min")
    y = np.arange(10, 10 * (n + 1), 10, dtype=float)
    return pd.DataFrame({"date": dates, "Appliances": y})


def test_features_from_history_khop_voi_ca_chuoi():
    df = _series()
    full = add_lag_features(add_time_features(df))
    t = df["date"].iloc[12]
    one = features_from_history(t, df["Appliances"].iloc[12 - N_HISTORY:12])
    cols = LAG_FEATURES + ["nsm", "hour_sin", "hour_cos", "day_of_week", "is_weekend"]
    np.testing.assert_allclose(one[cols].to_numpy(dtype=float), full.iloc[[12]][cols].to_numpy(dtype=float))


def test_features_from_history_sai_so_luong():
    with pytest.raises(ValueError):
        features_from_history("2016-05-10 18:00", [60, 70])


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Chưa có models/best_model.joblib (chạy notebook 09)")
def test_mo_hinh_da_luu_du_bao_hop_ly():
    model, info = load_model()
    flat = predict_next(model, info["features"], "2016-05-10 03:00", [50] * N_HISTORY)
    peak = predict_next(model, info["features"], "2016-05-10 18:00", [60, 60, 70, 90, 300, 500])
    assert 0 < flat < 120          # đêm, tiêu thụ đều 50 Wh → dự báo mức nền
    assert peak > flat             # vừa có đỉnh → dự báo cao hơn
