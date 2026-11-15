"""Test cho Preprocessor và tính không rò rỉ của target encoding."""
import numpy as np
import pytest

from src.model.HistGB import HistGBModel
from src.preprocess.Preprocess import TEXT_COLS, Preprocessor


def test_train_va_test_cung_bo_cot(prepared):
    X, _, Xt, _ = prepared
    assert list(X.columns) == list(Xt.columns)


def test_cot_so_khong_chua_cot_chu(prepared):
    X, _, _, pre = prepared
    assert not set(pre.num_cols_) & set(TEXT_COLS)
    assert X[pre.num_cols_].select_dtypes(exclude="number").empty


def test_bao_loi_khi_thieu_cot(prepared):
    _, _, Xt, pre = prepared
    with pytest.raises(ValueError):
        pre.transform(Xt.drop(columns=["energy"]))


def test_bao_loi_khi_con_missing(prepared):
    _, _, Xt, pre = prepared
    bad = Xt.copy()
    bad.loc[bad.index[0], "energy"] = np.nan
    with pytest.raises(ValueError):
        pre.transform(bad)


def test_target_encoding_nam_trong_pipeline_cua_model(cfg, prepared):
    """Bộ mã hóa phải là một bước của model, để mỗi fold tự học lại từ đầu."""
    _, _, _, pre = prepared
    m = HistGBModel(cfg, pre)
    assert [name for name, _ in m.model.steps] == ["enc", "clf"]


def test_target_encoding_chi_hoc_tu_du_lieu_duoc_fit(cfg, prepared):
    """Đổi nhãn của các dòng NGOÀI tập fit không được làm đổi dự đoán.

    Nếu target encoding lén dùng nhãn toàn bộ train thì phép thử này thất bại.
    """
    X, y, Xt, pre = prepared
    half = np.arange(len(X)) % 2 == 0
    p1 = HistGBModel(cfg, pre).fit(X[half], y[half]).predict_proba(Xt)
    y2 = y.copy()
    y2[~half] = (y2[~half] + 5) % 11          # phá nhãn của nửa không được fit
    p2 = HistGBModel(cfg, pre).fit(X[half], y2[half]).predict_proba(Xt)
    assert np.allclose(p1, p2)
