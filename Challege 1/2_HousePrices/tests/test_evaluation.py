"""Test cho Evaluator (người 4)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluation.Evaluation import Evaluator, rmse


class TinyModel:
    """Model giả tối giản, đúng giao diện mà oof_predict cần."""
    def __init__(self):
        self.m = LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(2 * X["a"] - X["b"] + rng.normal(0, 0.1, n))
    return X, y


def test_du_doan_hoan_hao():
    y = np.array([1.0, 2.0, 3.0])
    m = Evaluator.regression_metrics(y, y)
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1


def test_rmsle_tinh_tren_gia_goc():
    y = np.array([100000.0, 200000.0])
    assert Evaluator.rmsle(y, y) == 0
    assert Evaluator.rmsle(y, np.array([-5.0, 200000.0])) > 0  # giá âm bị kẹp về 0


def test_oof_phu_het_moi_dong_dung_mot_lan():
    """Mỗi dòng train phải được dự đoán bởi model chưa từng thấy nó."""
    X, y = _data()
    r = Evaluator.oof_predict(TinyModel, X, y, n_folds=5)
    assert len(r["oof"]) == len(X)
    assert not np.isnan(r["oof"]).any()
    assert len(r["fold_scores"]) == 5


def test_oof_khop_voi_rmse_tinh_tay():
    X, y = _data()
    r = Evaluator.oof_predict(TinyModel, X, y, n_folds=5)
    assert abs(r["rmse_oof"] - rmse(y, r["oof"])) < 1e-12


def test_test_pred_dung_do_dai():
    X, y = _data()
    X_test = X.head(10)
    r = Evaluator.oof_predict(TinyModel, X, y, X_test, n_folds=5)
    assert r["test_pred"].shape == (10,)


def test_khong_co_x_test_thi_tra_none():
    X, y = _data()
    assert Evaluator.oof_predict(TinyModel, X, y, n_folds=3)["test_pred"] is None


def test_compare_sap_theo_rmse_tang_dan():
    X, y = _data()
    a = Evaluator.oof_predict(TinyModel, X, y, n_folds=5)
    b = dict(a, rmse_oof=a["rmse_oof"] + 1, rmse_mean=a["rmse_mean"] + 1)
    df = Evaluator.compare({"xau": b, "tot": a})
    assert df.iloc[0]["model"] == "tot"
    assert list(df.columns) == ["model", "rmse_oof", "rmse_cv_mean", "rmse_cv_std"]