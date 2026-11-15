"""Làm sạch dữ liệu Music Genre theo đúng ngữ nghĩa của từng cột.

Ba phát hiện từ EDA quyết định cách xử lý:

1. `duration_in min/ms` trộn HAI đơn vị: khoảng 14% dòng ghi bằng phút
   (giá trị < 30), còn lại ghi bằng mili-giây. Phải quy về một đơn vị.
2. `key` không có giá trị 0 nào trong khi thang Spotify là 0..11, và tỉ lệ
   thiếu (11%) xấp xỉ tần suất của một cung nhạc. NA ở đây chính là key = 0.
3. `instrumentalness` không có giá trị 0 nào (nhỏ nhất là 1e-6) nhưng thiếu
   tới 25%, và tỉ lệ thiếu khác hẳn nhau giữa các lớp. NA ở đây là giá trị 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Cleaner:
    """Học tham số từ train, áp dụng cho cả train và test."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        c = cfg.get("clean", {})
        self.duration_col = c.get("duration_col", "duration_in min/ms")
        self.minute_threshold = c.get("minute_threshold", 30)
        self.popularity_median_: float = 0.0
        self.report_: dict = {}

    def fit(self, train: pd.DataFrame) -> "Cleaner":
        self.popularity_median_ = float(train["Popularity"].median())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Thời lượng: quy về mili-giây, giữ lại cờ đơn vị gốc.
        #    Cờ này mang thông tin: các thể loại được thu thập từ nguồn khác nhau
        #    nên đơn vị ghi cũng khác nhau.
        d = df[self.duration_col]
        is_min = d < self.minute_threshold
        df["duration_is_min"] = is_min.astype(int)
        df["duration_ms"] = np.where(is_min, d * 60_000, d)
        df = df.drop(columns=[self.duration_col])

        # 2. key: NA nghĩa là cung Đô (key = 0), không phải thiếu.
        df["key"] = df["key"].fillna(0).astype(int)

        # 3. instrumentalness: NA nghĩa là 0. Giữ cờ vì tỉ lệ khác nhau theo lớp.
        df["instr_missing"] = df["instrumentalness"].isna().astype(int)
        df["instrumentalness"] = df["instrumentalness"].fillna(0.0)

        # 4. Popularity: thiếu thật -> trung vị học từ train, kèm cờ.
        df["pop_missing"] = df["Popularity"].isna().astype(int)
        df["Popularity"] = df["Popularity"].fillna(self.popularity_median_)

        # 5. Chuỗi: bỏ khoảng trắng thừa, tránh NaN.
        for c in ("Artist Name", "Track Name"):
            df[c] = df[c].fillna("").astype(str).str.strip()
        return df

    def run(self, train: pd.DataFrame, test: pd.DataFrame, target: str):
        n = len(train)
        train = train.drop_duplicates().reset_index(drop=True)   # chỉ bỏ dòng trùng HOÀN TOÀN
        self.report_["duplicates_removed"] = n - len(train)
        self.report_["rows_train"] = len(train)
        self.report_["rows_test"] = len(test)
        self.report_["minute_rows_train"] = int((train[self.duration_col] < self.minute_threshold).sum())
        self.report_["na_before"] = train.isna().sum()[lambda s: s > 0].to_dict()
        self.fit(train)
        return self.transform(train), self.transform(test)

    def summary(self) -> str:
        r = self.report_
        return (
            f"- Dòng train: {r.get('rows_train')} (bỏ {r.get('duplicates_removed')} dòng trùng hoàn toàn)\n"
            f"- Dòng test: {r.get('rows_test')}\n"
            f"- Dòng train ghi thời lượng bằng phút: {r.get('minute_rows_train')}\n"
            f"- Giá trị thiếu trước khi xử lý: {r.get('na_before')}\n"
        )
