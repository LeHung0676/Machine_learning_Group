"""Test chung cho mọi model: cùng giao diện, xác suất hợp lệ."""
import numpy as np
import pytest

from src.pipeline.Pipeline import MODELS, available_models


@pytest.mark.parametrize("name", ["histgb", "randomforest", "extratrees", "logreg", "text"])
def test_model_fit_va_predict_proba(name, cfg, prepared):
    X, y, Xt, pre = prepared
    m = MODELS[name](cfg, pre).fit(X, y)
    P = m.predict_proba(Xt)
    assert P.shape == (len(Xt), 11)
    assert np.allclose(P.sum(axis=1), 1)
    assert list(m.classes_) == list(range(11))
    assert set(m.predict(Xt)) <= set(range(11))


def test_model_thieu_thu_vien_khong_duoc_dang_ky():
    """Model nào có mặt trong bảng đăng ký thì phải dùng được ngay."""
    import importlib.util
    for name, lib in [("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("catboost", "catboost")]:
        assert (name in available_models()) == (importlib.util.find_spec(lib) is not None)
