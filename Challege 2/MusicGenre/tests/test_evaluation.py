"""Test cho Evaluator."""
import numpy as np

from src.evaluation.Evaluation import Evaluator
from src.model.LogReg import LogRegModel


def test_du_doan_hoan_hao():
    y = np.array([0, 1, 2, 1])
    proba = np.eye(3)[y]
    m = Evaluator.metrics(y, proba, [0, 1, 2])
    assert m["accuracy"] == 1 and m["f1_macro"] == 1 and m["log_loss"] < 1e-9


def test_oof_la_xac_suat_hop_le(cfg, prepared):
    X, y, Xt, pre = prepared
    r = Evaluator.oof_predict(lambda: LogRegModel(cfg, pre), X, y, Xt, n_folds=3)
    assert r["oof"].shape == (len(X), 11)
    assert np.allclose(r["oof"].sum(axis=1), 1)        # mỗi dòng cộng lại bằng 1
    assert (r["oof"] >= 0).all()
    assert r["test_proba"].shape == (len(Xt), 11)
    assert np.allclose(r["test_proba"].sum(axis=1), 1)
    assert len(r["fold_acc"]) == 3


def test_oof_phu_moi_dong(cfg, prepared):
    """Không dòng nào bị bỏ sót: dòng chưa được dự đoán sẽ có tổng xác suất 0."""
    X, y, _, pre = prepared
    r = Evaluator.oof_predict(lambda: LogRegModel(cfg, pre), X, y, n_folds=3)
    assert (r["oof"].sum(axis=1) > 0.99).all()
    assert r["test_proba"] is None


def test_model_hoc_duoc_tin_hieu(cfg, prepared):
    """Dữ liệu giả có tín hiệu rõ, model phải hơn hẳn đoán mò (1/11)."""
    X, y, _, pre = prepared
    r = Evaluator.oof_predict(lambda: LogRegModel(cfg, pre), X, y, n_folds=3)
    assert r["accuracy"] > 0.5


def test_ma_tran_nham_lan_chuan_hoa_theo_hang():
    y = np.array([0, 0, 1, 1])
    proba = np.array([[.9, .1], [.2, .8], [.1, .9], [.3, .7]])
    cm = Evaluator.confusion(y, proba, [0, 1])
    assert np.allclose(cm, [[0.5, 0.5], [0.0, 1.0]])


def test_bang_theo_lop(cfg):
    y = np.array([0, 0, 1, 2])
    proba = np.eye(3)[[0, 1, 1, 2]]
    df = Evaluator.per_class(y, proba, [0, 1, 2])
    assert list(df["support"]) == [2, 1, 1]
    assert df.loc[df["class"] == 0, "recall"].iloc[0] == 0.5


def test_compare_sap_theo_accuracy_giam_dan():
    a = {"accuracy": 0.7, "acc_std": 0.01, "f1_macro": 0.7, "log_loss": 0.8}
    b = {"accuracy": 0.6, "acc_std": 0.01, "f1_macro": 0.6, "log_loss": 0.9}
    df = Evaluator.compare({"kem": b, "tot": a})
    assert df.iloc[0]["model"] == "tot"
