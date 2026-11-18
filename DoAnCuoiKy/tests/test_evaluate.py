"""Kiểm thử độ đo, baseline và hàm cross-validation.

Chạy:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluate import HourMeanBaseline, MeanBaseline, regression_metrics  # noqa: E402
from src.train import cv_evaluate, fit_and_validate  # noqa: E402


def test_metrics_perfect_prediction():
    y = np.array([10.0, 50.0, 100.0])
    m = regression_metrics(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1 and m["MAPE_%"] == 0


def test_metrics_known_values():
    y = np.array([10.0, 20.0])
    p = np.array([12.0, 16.0])          # sai số 2 và 4
    m = regression_metrics(y, p)
    assert np.isclose(m["MAE"], 3.0)
    assert np.isclose(m["RMSE"], np.sqrt((4 + 16) / 2))
    assert np.isclose(m["MAPE_%"], 100 * (0.2 + 0.2) / 2)


def test_mean_baseline_r2_zero_on_train():
    X = pd.DataFrame({"a": range(10)})
    y = np.arange(10, dtype=float)
    pred = MeanBaseline().fit(X, y).predict(X)
    assert np.allclose(pred, 4.5)
    assert np.isclose(regression_metrics(y, pred)["R2"], 0.0)


def test_hour_mean_baseline():
    X = pd.DataFrame({"nsm": [0, 600, 3600, 3600 + 600]})   # 2 mẫu giờ 0, 2 mẫu giờ 1
    y = np.array([10.0, 30.0, 100.0, 200.0])
    model = HourMeanBaseline().fit(X, y)
    assert np.allclose(model.predict(X), [20, 20, 150, 150])
    # giờ chưa thấy khi train → dùng trung bình chung
    assert np.isclose(model.predict(pd.DataFrame({"nsm": [5 * 3600]}))[0], y.mean())


def _linear_data(n=200):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = pd.Series(3 * X["x1"] - 2 * X["x2"] + rng.normal(scale=0.1, size=n))
    return X, y


def test_cv_evaluate_keys_and_quality():
    X, y = _linear_data()
    r = cv_evaluate(LinearRegression(), X, y, n_splits=4)
    for k in ("cv_RMSE", "cv_RMSE_std", "train_RMSE", "cv_R2"):
        assert k in r
    assert r["cv_R2"] > 0.95


def test_fit_and_validate_does_not_modify_input_model():
    X, y = _linear_data()
    model = LinearRegression()
    fitted, r = fit_and_validate(model, X[:150], y[:150], X[150:], y[150:])
    assert not hasattr(model, "coef_")      # mô hình gốc không bị fit (dùng clone)
    assert hasattr(fitted, "coef_")
    assert r["val_R2"] > 0.95
