"""Nạp và kiểm tra dữ liệu gốc.

Chạy thử:  python -m src.data_loader
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATE_COL, RAW_DATA_PATH, TARGET


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc file CSV gốc, chuyển cột thời gian sang datetime và sắp xếp theo thời gian.

    File gốc lưu số dạng chuỗi có khoảng trắng (vd: "  60"), nên ép kiểu số cho mọi cột trừ `date`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Tải dữ liệu tại "
            "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction "
            "và đặt vào data/raw/."
        )

    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y-%m-%d %H:%M:%S")
    num_cols = [c for c in df.columns if c != DATE_COL]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def basic_report(df: pd.DataFrame) -> dict:
    """Thống kê nhanh để kiểm tra chất lượng dữ liệu."""
    diffs = df[DATE_COL].diff().dropna()
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "start": df[DATE_COL].min(),
        "end": df[DATE_COL].max(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "duplicated_timestamps": int(df[DATE_COL].duplicated().sum()),
        "irregular_intervals": int((diffs != pd.Timedelta(minutes=10)).sum()),
        "target_mean": round(df[TARGET].mean(), 2),
        "target_median": float(df[TARGET].median()),
        "target_max": float(df[TARGET].max()),
    }


if __name__ == "__main__":
    data = load_raw_data()
    print(data.head())
    print()
    for k, v in basic_report(data).items():
        print(f"{k:>22}: {v}")
