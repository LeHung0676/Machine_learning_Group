"""Nạp mô hình cuối cùng và dự báo tiêu thụ 10 phút tới (dùng cho demo `app/` và kiểm thử).

Mô hình được lưu bởi notebook 09: `models/best_model.joblib` (HGB, bộ đặc trưng `lag_time`) kèm
`models/best_model_info.json` (danh sách đặc trưng, siêu tham số, kết quả test).

Chạy thử:  python -m src.predict
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import DATE_COL, MODELS_DIR, TARGET
from src.features import LAG_STEPS, add_lag_features, add_time_features

MODEL_PATH = MODELS_DIR / "best_model.joblib"
INFO_PATH = MODELS_DIR / "best_model_info.json"
N_HISTORY = max(LAG_STEPS)          # cần 6 giá trị gần nhất (60 phút)


def load_model(model_path: str | Path = MODEL_PATH, info_path: str | Path = INFO_PATH):
    """Trả về (mô hình, thông tin mô hình). Báo lỗi rõ ràng nếu chưa chạy notebook 09."""
    model_path, info_path = Path(model_path), Path(info_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Chưa có {model_path}. Chạy notebooks/09_final_evaluation.ipynb để huấn luyện và lưu mô hình.")
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    return joblib.load(model_path), info


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm đặc trưng thời gian + quá khứ gần cho cả một chuỗi (có cột `date`, `Appliances`)."""
    return add_lag_features(add_time_features(df))


def features_from_history(timestamp, recent_values) -> pd.DataFrame:
    """Tạo 1 dòng đặc trưng để dự báo tiêu thụ tại `timestamp`.

    recent_values: 6 giá trị tiêu thụ (Wh) gần nhất, theo thứ tự CŨ → MỚI, tức tại
    timestamp−60, −50, −40, −30, −20, −10 phút.
    """
    recent = np.asarray(recent_values, dtype=float)
    if recent.shape != (N_HISTORY,):
        raise ValueError(f"Cần đúng {N_HISTORY} giá trị gần nhất (60 phút), nhận được {recent.size}")
    timestamp = pd.Timestamp(timestamp)
    dates = pd.date_range(end=timestamp, periods=N_HISTORY + 1, freq="10min")
    hist = pd.DataFrame({DATE_COL: dates, TARGET: np.append(recent, np.nan)})
    return build_features(hist).iloc[[-1]]


def predict_next(model, features: list[str], timestamp, recent_values) -> float:
    """Dự báo tiêu thụ (Wh) tại `timestamp` từ 60 phút lịch sử."""
    X = features_from_history(timestamp, recent_values)[features]
    return float(model.predict(X)[0])


def predict_series(model, features: list[str], df: pd.DataFrame) -> pd.Series:
    """Dự báo cho mọi dòng của một chuỗi liên tục (mỗi dòng dùng giá trị thực của 60 phút trước nó).

    df nên chứa thêm ≥ 6 dòng trước khoảng cần xem; các dòng đầu chưa đủ lịch sử trả về NaN.
    """
    feats = build_features(df)
    out = pd.Series(np.nan, index=feats.index, name="prediction")
    ok = feats[features].notna().all(axis=1)
    if ok.any():
        out[ok] = model.predict(feats.loc[ok, features])
    out.index = feats[DATE_COL]
    return out


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    model, info = load_model()
    feats = info.get("features")
    df = load_raw_data()
    t = df[DATE_COL].iloc[-1]
    recent = df[TARGET].iloc[-1 - N_HISTORY:-1].tolist()
    print("Mô hình:", info.get("model"), "| đặc trưng:", len(feats))
    print(f"60 phút trước {t}: {recent}")
    print(f"Dự báo tại {t}: {predict_next(model, feats, t, recent):.1f} Wh | thực tế: {df[TARGET].iloc[-1]} Wh")
