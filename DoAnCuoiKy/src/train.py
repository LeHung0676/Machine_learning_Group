"""Huấn luyện và cross-validation dùng chung cho mọi mô hình.

Dùng `TimeSeriesSplit` trên tập train: mỗi fold validation luôn nằm **sau** phần dữ liệu dùng để huấn luyện,
giống tình huống thực tế (học quá khứ → dự đoán tương lai), không rò rỉ thông tin.

    fold 1: [train ][val]
    fold 2: [train      ][val]
    fold 3: [train           ][val]
    ...
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from src.evaluate import regression_metrics

N_SPLITS = 5
SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}


def time_cv(n_splits: int = N_SPLITS) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def cv_evaluate(model, X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> dict:
    """Cross-validation theo thời gian. Trả về trung bình và độ lệch chuẩn của từng độ đo trên các fold,
    kèm điểm trên phần train của fold để phát hiện overfitting (train tốt hơn val quá nhiều)."""
    res = cross_validate(model, X, y, cv=time_cv(n_splits), scoring=SCORING,
                         return_train_score=True, n_jobs=1)
    out = {}
    for m in SCORING:
        sign = -1 if m in ("MAE", "RMSE") else 1
        val = sign * res[f"test_{m}"]
        tr = sign * res[f"train_{m}"]
        out[f"cv_{m}"] = val.mean()
        out[f"cv_{m}_std"] = val.std()
        out[f"train_{m}"] = tr.mean()
    out["fit_time_s"] = res["fit_time"].mean()
    return out


def fit_and_validate(model, X_train, y_train, X_val, y_val) -> tuple[object, dict]:
    """Huấn luyện trên toàn bộ train, đánh giá trên validation (và train để so sánh overfitting)."""
    model = clone(model)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    tr = regression_metrics(y_train, model.predict(X_train))
    va = regression_metrics(y_val, model.predict(X_val))
    result = {f"train_{k}": v for k, v in tr.items()} | {f"val_{k}": v for k, v in va.items()}
    result["fit_time_s"] = fit_time
    return model, result


def summarize(results: dict[str, dict], cols: list[str] | None = None) -> pd.DataFrame:
    """Bảng tóm tắt kết quả nhiều mô hình."""
    df = pd.DataFrame(results).T
    if cols:
        df = df[cols]
    return df.round(3)


if __name__ == "__main__":
    from sklearn.linear_model import Ridge

    from src.data_loader import load_raw_data
    from src.features import make_xy
    from src.preprocessing import make_pipeline
    from src.split import time_split

    train, val, _ = time_split(load_raw_data())
    X_tr, y_tr = make_xy(train)
    pipe = make_pipeline(Ridge(alpha=1.0), list(X_tr.columns), "linear")
    r = cv_evaluate(pipe, X_tr, y_tr)
    print({k: round(v, 3) for k, v in r.items()})
