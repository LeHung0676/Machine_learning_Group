"""Test cho Ensemble (người 4)."""
import numpy as np
import pytest

from src.ensemble.Ensemble import Blending, Stacking, Voting, build, compare_methods
from src.evaluation.Evaluation import rmse


@pytest.fixture
def oof_data():
    """Hai model: một lệch lên, một lệch xuống. Trộn lại phải tốt hơn cả hai."""
    rng = np.random.default_rng(0)
    y = rng.normal(12, 0.4, 200)
    return {"a": y + rng.normal(0.1, 0.15, 200),
            "b": y + rng.normal(-0.1, 0.15, 200)}, y


def test_trong_so_duoc_chuan_hoa_ve_tong_1():
    b = Blending({"a": 2.0, "b": 2.0})
    assert b.weights == {"a": 0.5, "b": 0.5}


def test_tong_trong_so_bang_0_thi_bao_loi():
    with pytest.raises(ValueError):
        Blending({"a": 0.0, "b": 0.0})


def test_fit_tim_trong_so_hop_le(oof_data):
    oof, y = oof_data
    b = Blending().fit(oof, y)
    assert abs(sum(b.weights.values()) - 1) < 1e-6
    assert all(w >= 0 for w in b.weights.values())


def test_blending_tot_hon_tung_model_don_le(oof_data):
    oof, y = oof_data
    b = Blending().fit(oof, y)
    tron = rmse(y, b.predict(oof))
    assert tron <= min(rmse(y, v) for v in oof.values())


def test_blending_khong_te_hon_voting(oof_data):
    """Voting là trường hợp riêng của Blending, nên Blending phải bằng hoặc hơn."""
    oof, y = oof_data
    v = rmse(y, Voting(list(oof)).predict(oof))
    b = rmse(y, Blending().fit(oof, y).predict(oof))
    assert b <= v + 1e-9


def test_chua_fit_thi_bao_loi(oof_data):
    oof, _ = oof_data
    with pytest.raises(RuntimeError):
        Blending().predict(oof)


def test_thieu_du_doan_thi_bao_loi(oof_data):
    oof, y = oof_data
    b = Blending().fit(oof, y)
    with pytest.raises(KeyError):
        b.predict({"a": oof["a"]})


def test_voting_chia_deu():
    v = Voting(["a", "b", "c"])
    assert all(abs(w - 1 / 3) < 1e-9 for w in v.weights.values())


def test_stacking_chay_duoc(oof_data):
    oof, y = oof_data
    s = Stacking().fit(oof, y)
    pred = s.predict(oof)
    assert pred.shape == y.shape
    assert rmse(y, pred) <= min(rmse(y, v) for v in oof.values())


def test_build_dispatch():
    cfg = {"ensemble": {"weights": {"a": 0.7, "b": 0.3}}}
    assert isinstance(build("voting", ["a", "b"], cfg), Voting)
    assert isinstance(build("stacking", ["a", "b"], cfg), Stacking)
    bl = build("blending", ["a", "b"], cfg)
    assert bl.weights == {"a": 0.7, "b": 0.3}
    with pytest.raises(ValueError):
        build("khong_ton_tai", ["a"], cfg)


def test_build_blending_khong_co_trong_so_thi_de_fit_tu_tim():
    bl = build("blending", ["a", "b"], {"ensemble": {}})
    assert bl.weights is None


def test_compare_methods_co_du_dong(oof_data):
    oof, y = oof_data
    df = compare_methods(oof, y)
    assert len(df) == 3 + len(oof)          # 3 cách kết hợp + từng model đơn lẻ
    assert df["rmse_oof"].is_monotonic_increasing
