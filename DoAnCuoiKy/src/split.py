"""Chia dữ liệu theo thời gian (không xáo trộn) để tránh rò rỉ thông tin từ tương lai.

Chạy thử:  python -m src.split
"""
from __future__ import annotations

import pandas as pd

from src.config import DATE_COL

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = phần còn lại (0.15)


def time_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia df thành train / validation / test theo thứ tự thời gian.

    - Train     : 70% đầu  → huấn luyện, cross-validation, tinh chỉnh siêu tham số
    - Validation: 15% giữa → so sánh và chọn mô hình
    - Test      : 15% cuối → đánh giá cuối cùng, chỉ dùng MỘT lần

    Vì sao không chia ngẫu nhiên? Các mẫu cách nhau 10 phút gần như giống hệt nhau. Nếu xáo trộn, mẫu lúc 10:00
    nằm trong train còn 10:10 nằm trong test → mô hình "nhìn trộm" tương lai, kết quả đẹp ảo.
    """
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Tỷ lệ chia không hợp lệ")

    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    return df.iloc[:i_train].copy(), df.iloc[i_train:i_val].copy(), df.iloc[i_val:].copy()


def describe_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   target: str = "Appliances", date_col: str = DATE_COL) -> pd.DataFrame:
    """Bảng tóm tắt khoảng thời gian, số mẫu và phân phối biến mục tiêu của từng tập."""
    rows = []
    for name, part in (("train", train), ("validation", val), ("test", test)):
        rows.append({
            "tập": name,
            "số mẫu": len(part),
            "từ": part[date_col].min(),
            "đến": part[date_col].max(),
            "TB mục tiêu": round(part[target].mean(), 1),
            "trung vị": part[target].median(),
        })
    return pd.DataFrame(rows).set_index("tập")


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    tr, va, te = time_split(load_raw_data())
    print(describe_split(tr, va, te).to_string())
