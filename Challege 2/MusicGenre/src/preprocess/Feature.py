"""Tạo đặc trưng cho Music Genre.

Hai nhóm:
- Khóa định danh `pair` = nghệ sĩ + tên bài, dùng cho target encoding.
  Cùng một bài hát xuất hiện nhiều lần trong dữ liệu với nhãn khác nhau,
  nên nhãn của bản ghi "anh em" là tín hiệu rất mạnh.
- Đặc trưng rút từ tên bài: độ dài, số từ, dấu hiệu feat/remix/live,
  tỉ lệ ký tự ngoài ASCII (gợi ý ngôn ngữ, ví dụ nhạc Bollywood).
"""
from __future__ import annotations

import pandas as pd

PAIR_SEP = " || "


class FeatureEngineer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.use_name_features = cfg.get("features", {}).get("name_features", True)

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["pair"] = df["Artist Name"] + PAIR_SEP + df["Track Name"]

        if self.use_name_features:
            t = df["Track Name"]
            low = t.str.lower()
            df["track_len"] = t.str.len()
            df["track_words"] = t.str.split().str.len().fillna(0).astype(int)
            df["has_feat"] = low.str.contains(r"\bfeat\b|\bft\b|featuring|\(with ", regex=True).astype(int)
            df["has_version"] = low.str.contains(r"remaster|remix|live|version|edit|acoustic|from \"", regex=True).astype(int)
            df["non_ascii_ratio"] = t.apply(
                lambda s: sum(ord(ch) > 127 for ch in s) / len(s) if len(s) else 0.0)
            df["artist_multi"] = df["Artist Name"].str.contains(r",|&| x ", regex=True).astype(int)

        # Tương tác đơn giản giữa các thuộc tính âm thanh
        df["energy_x_loud"] = df["energy"] * (df["loudness"] + 60) / 60
        df["acoustic_minus_energy"] = df["acousticness"] - df["energy"]
        df["duration_min"] = df["duration_ms"] / 60_000
        return df
