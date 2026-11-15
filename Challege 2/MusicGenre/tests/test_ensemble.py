"""Test cho Ensemble."""
import numpy as np
import pytest
from sklearn.metrics import log_loss

from src.ensemble.Ensemble import Blending, Stacking, Voting, build, compare_methods


@pytest.fixture
def oof_data():
    """Ba model trên 4 lớp: tốt, khá, và nhiễu thuần."""
    rng = np.random.default_rng(0)
    n, k = 600, 4
    y = rng.integers(0, k, n)

    def make(strength):
        logits = rng.normal(0, 1, (n, k))
        logits[np.arange(n), y] += strength
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    return {"tot": make(2.5), "kha": make(1.5), "nhieu": make(0.0)}, y


def test_trong_so_duoc_chuan_hoa():
    assert Blending({"a": 2.0, "b": 2.0}).weights == {"a": 0.5, "b": 0.5}


def test_tong_trong_so_bang_0_thi_bao_loi():
    with pytest.raises(ValueError):
        Blending({"a": 0.0})


def test_fit_tim_trong_so_hop_le(oof_data):
    oof, y = oof_data
    b = Blending().fit(oof, y)
    assert abs(sum(b.weights.values()) - 1) < 1e-6
    assert all(w >= 0 for w in b.weights.values())


def test_model_nhieu_nhan_trong_so_thap_nhat(oof_data):
    oof, y = oof_data
    w = Blending().fit(oof, y).weights
    assert w["nhieu"] < w["kha"] < w["tot"]


def test_blending_khong_te_hon_voting_ve_log_loss(oof_data):
    """Voting là trường hợp riêng của Blending nên Blending phải bằng hoặc hơn."""
    oof, y = oof_data
    v = log_loss(y, Voting(list(oof)).predict_proba(oof))
    b = log_loss(y, Blending().fit(oof, y).predict_proba(oof))
    assert b <= v + 1e-9


def test_ket_qua_van_la_xac_suat(oof_data):
    oof, y = oof_data
    for ens in (Voting(list(oof)), Blending().fit(oof, y), Stacking().fit(oof, y)):
        P = ens.predict_proba(oof)
        assert P.shape == (len(y), 4)
        assert np.allclose(P.sum(axis=1), 1)


def test_chua_fit_thi_bao_loi(oof_data):
    oof, _ = oof_data
    with pytest.raises(RuntimeError):
        Blending().predict_proba(oof)


def test_thieu_du_doan_thi_bao_loi(oof_data):
    oof, y = oof_data
    b = Blending().fit(oof, y)
    with pytest.raises(KeyError):
        b.predict_proba({"tot": oof["tot"]})


def test_stacking_oof_khong_dung_nhan_cua_chinh_dong_do(oof_data):
    """oof_proba phải là cross-validation: đổi nhãn một dòng không đổi dự đoán của dòng đó
    nhiều bằng việc fit trực tiếp. Kiểm tra gián tiếp: điểm OOF không cao hơn điểm fit-rồi-chấm."""
    oof, y = oof_data
    s = Stacking()
    honest = log_loss(y, s.oof_proba(oof, y))
    optimistic = log_loss(y, Stacking().fit(oof, y).predict_proba(oof))
    assert honest >= optimistic - 1e-9


def test_build_dispatch():
    assert isinstance(build("voting", ["a"], {}), Voting)
    assert isinstance(build("blending", ["a"], {}), Blending)
    assert isinstance(build("stacking", ["a"], {}), Stacking)
    with pytest.raises(ValueError):
        build("khong_ton_tai", ["a"], {})


def test_compare_methods_co_du_dong(oof_data):
    oof, y = oof_data
    df = compare_methods(oof, y, [0, 1, 2, 3])
    assert len(df) == 3 + len(oof)
    assert df["accuracy"].is_monotonic_decreasing
